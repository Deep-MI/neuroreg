"""MRI_coreg-style brute-force and Powell image-registration backend."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import minimize
from scipy.signal import convolve2d
from scipy.spatial.transform import Rotation

from neuroreg.image import reslice_r2r_image
from neuroreg.imreg.reg_model import RegModel
from neuroreg.transforms import LINEAR_RAS_TO_RAS, LINEAR_VOX_TO_VOX, LTA, convert_transform_type

from .device import resolve_cpu_only_device
from .init import InitType, get_init_vox2vox, resolve_init_type

logger = logging.getLogger(__name__)

_HIST_SIZE = 256
_HIST_FWHM = 7.0
_EPS = np.finfo(np.float32).eps


@dataclass(frozen=True, slots=True)
class PowellCostResult:
    """Result of a single Powell cost evaluation."""

    cost: float
    nhits: int
    pcthits: float


@dataclass(frozen=True, slots=True)
class PowellOptimizationStep:
    """One recorded brute-force or Powell refinement step."""

    iteration: int
    cost: float
    r2r: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True, slots=True)
class PowellOptimizationResult:
    """Summary of a completed Powell optimization run."""

    success: bool
    message: str
    nit: int
    nfev: int
    initial_cost: float
    final_cost: float
    initial_r2r: np.ndarray
    final_r2r: np.ndarray
    weights: np.ndarray
    history: tuple[PowellOptimizationStep, ...]


@dataclass(frozen=True, slots=True)
class PowellBruteForceResult:
    """Summary of the coarse brute-force sweep preceding Powell refinement."""

    params: np.ndarray
    cost: float
    history: tuple[PowellOptimizationStep, ...]


@dataclass(frozen=True, slots=True)
class _PreparedVolume:
    data: np.ndarray
    affine: np.ndarray
    shape: tuple[int, int, int]
    voxel_sizes: np.ndarray


class PowellCostEvaluator:
    """Evaluate MRI_coreg-style NMI costs for moving/reference image pairs."""

    def __init__(
            self,
            mov_img: nib.spatialimages.SpatialImage,
            ref_img: nib.spatialimages.SpatialImage,
            *,
            ref_mask_img: nib.spatialimages.SpatialImage | None = None,
            sep: int = 4,
            saturation_pct: float = 99.99,
            seed: int = 53,
            coord_dither: bool = True,
            intensity_dither: bool = True,
            smooth_images: bool = True,
            smooth_histogram: bool = True,
            include_oob: bool = False,
    ) -> None:
        """Prepare cached moving/reference volumes for repeated NMI evaluation."""
        if sep <= 0:
            raise ValueError("sep must be positive")
        self.sep = int(sep)
        self.seed = int(seed)
        self.coord_dither = coord_dither
        self.include_oob = include_oob
        ref_data = np.asarray(ref_img.get_fdata(dtype=np.float32), dtype=np.float32)
        if ref_mask_img is not None:
            ref_mask = np.asarray(ref_mask_img.get_fdata(dtype=np.float32), dtype=np.float32)
            ref_data = np.where(ref_mask > 0, ref_data, 0.0).astype(np.float32, copy=False)
        self.ref = _prepare_volume(
            ref_data,
            affine=np.asarray(ref_img.affine, dtype=np.float64),
            sep=self.sep,
            saturation_pct=saturation_pct,
            seed=self.seed + 1,
            intensity_dither=intensity_dither,
            smooth_images=smooth_images,
        )
        self.mov = _prepare_volume(
            np.asarray(mov_img.get_fdata(dtype=np.float32), dtype=np.float32),
            affine=np.asarray(mov_img.affine, dtype=np.float64),
            sep=self.sep,
            saturation_pct=saturation_pct,
            seed=self.seed + 2,
            intensity_dither=intensity_dither,
            smooth_images=smooth_images,
        )
        self.hist_kernel = (
            _gaussian_kernel(_HIST_FWHM / np.sqrt(np.log(_HIST_SIZE)), int(np.ceil(2.0 * _HIST_FWHM)))
            if smooth_histogram
            else None
        )
        self._initialize_cached_state()

    def evaluate_v2v(self, mov_to_ref_v2v: np.ndarray) -> PowellCostResult:
        """Evaluate a voxel-to-voxel transform by converting it to RAS space first."""
        mov_to_ref_r2r = convert_transform_type(
            mov_to_ref_v2v,
            src_affine=self.mov.affine,
            dst_affine=self.ref.affine,
            from_type=LINEAR_VOX_TO_VOX,
            to_type=LINEAR_RAS_TO_RAS,
        )
        return self.evaluate_r2r(mov_to_ref_r2r)

    def evaluate_r2r(self, mov_to_ref_r2r: np.ndarray) -> PowellCostResult:
        """Evaluate a moving-to-reference RAS transform with the cached NMI surrogate."""
        mov_to_ref_r2r = np.asarray(mov_to_ref_r2r, dtype=np.float64)
        if mov_to_ref_r2r.shape != (4, 4):
            raise ValueError("mov_to_ref_r2r must be 4x4")
        ref_to_mov_r2r = np.linalg.inv(mov_to_ref_r2r)
        ref_to_mov_v2v = self._inv_mov_affine @ ref_to_mov_r2r @ self.ref.affine

        if self._ref_sample_count == 0:
            return PowellCostResult(cost=float("inf"), nhits=0, pcthits=0.0)

        ref_vals = self._ref_vals
        mov_coords = ref_to_mov_v2v @ self._ref_coords_hom
        in_bounds = _coords_in_bounds(mov_coords[:3], self.mov.shape)
        nhits = int(np.count_nonzero(in_bounds))
        if not self.include_oob:
            ref_vals = ref_vals[in_bounds]
            mov_coords = mov_coords[:3, in_bounds]
        else:
            mov_coords = mov_coords[:3]
        if ref_vals.size == 0:
            return PowellCostResult(cost=float("inf"), nhits=0, pcthits=0.0)

        mov_vals = map_coordinates(self.mov.data, mov_coords, order=1, mode="constant", cval=0.0)
        hist = _joint_histogram(ref_vals, mov_vals)
        if self.hist_kernel is not None:
            hist = convolve2d(hist, self.hist_kernel[:, None], mode="full")
            hist = convolve2d(hist, self.hist_kernel[None, :], mode="full")
        hist += _EPS
        hist /= hist.sum()
        cost = _nmi_cost(hist)
        pcthits = (100.0 * self.sep ** 3 * float(nhits)) / self._ref_voxel_count
        return PowellCostResult(cost=float(cost), nhits=nhits, pcthits=pcthits)

    def evaluate_powell_params(self, params: np.ndarray, *, include_oob: bool | None = None) -> PowellCostResult:
        """Evaluate Powell parameter vectors in the canonical MRI_coreg parameterization."""
        mov_to_ref_r2r = powell_params_to_mov_to_ref_r2r(params)
        if include_oob is None or include_oob == self.include_oob:
            return self.evaluate_r2r(mov_to_ref_r2r)
        alt = self._alternate_evaluators.get(bool(include_oob))
        if alt is None:
            alt = PowellCostEvaluator.from_prepared(
                self.mov,
                self.ref,
                sep=self.sep,
                seed=self.seed,
                coord_dither=self.coord_dither,
                include_oob=include_oob,
                hist_kernel=self.hist_kernel,
            )
            self._alternate_evaluators[bool(include_oob)] = alt
        return alt.evaluate_r2r(mov_to_ref_r2r)

    @classmethod
    def from_prepared(
            cls,
            mov: _PreparedVolume,
            ref: _PreparedVolume,
            *,
            sep: int,
            seed: int,
            coord_dither: bool,
            include_oob: bool,
            hist_kernel: np.ndarray | None,
    ) -> PowellCostEvaluator:
        """Construct an evaluator from already prepared volumes and cached settings."""
        obj = cls.__new__(cls)
        obj.sep = int(sep)
        obj.seed = int(seed)
        obj.coord_dither = bool(coord_dither)
        obj.include_oob = include_oob
        obj.mov = mov
        obj.ref = ref
        obj.hist_kernel = hist_kernel
        obj._initialize_cached_state()
        return obj

    def _initialize_cached_state(self) -> None:
        """Build cached sampled reference coordinates and reference intensities."""
        self._inv_mov_affine = np.linalg.inv(self.mov.affine)
        self._ref_coords = _sample_reference_grid(self.ref.shape, self.sep, self.seed, self.coord_dither)
        self._ref_sample_count = int(self._ref_coords.shape[1])
        self._ref_coords_hom = np.vstack(
            [self._ref_coords.astype(np.float64, copy=False), np.ones((1, self._ref_sample_count), dtype=np.float64)]
        )
        self._ref_vals = (
            map_coordinates(self.ref.data, self._ref_coords, order=1, mode="nearest")
            if self._ref_sample_count > 0
            else np.empty((0,), dtype=np.float32)
        )
        self._ref_voxel_count = float(np.prod(self.ref.shape))
        self._alternate_evaluators: dict[bool, PowellCostEvaluator] = {}

    def brute_force_search(
            self,
            init_params: np.ndarray,
            *,
            limit: float = 30.0,
            niters: int = 1,
            n1d: int = 30,
            callback: Callable[[PowellOptimizationStep], None] | None = None,
    ) -> PowellBruteForceResult:
        """Run the coarse axis-aligned parameter sweep used before Powell refinement."""
        params = np.asarray(init_params, dtype=np.float64).copy()
        dof = min(6, params.shape[0])
        history: list[PowellOptimizationStep] = []
        mincost = float("inf")
        iteration = 0
        lim = float(limit)
        for _ in range(int(niters)):
            pdelta = 0.0
            for nthp in range(dof):
                pmin = params[nthp] - lim
                pmax = params[nthp] + lim
                pdelta = (pmax - pmin) / float(n1d)
                best_param = params[nthp]
                for nth1d in range(int(n1d) + 1):
                    p = pmin + nth1d * pdelta
                    params[nthp] = p
                    result = self.evaluate_powell_params(params, include_oob=True)
                    if result.cost < mincost:
                        mincost = result.cost
                        best_param = p
                    step = PowellOptimizationStep(
                        iteration=iteration,
                        cost=float(result.cost),
                        r2r=powell_params_to_mov_to_ref_r2r(params),
                        weights=params.copy(),
                    )
                    history.append(step)
                    if callback is not None:
                        callback(step)
                    iteration += 1
                params[nthp] = best_param
            lim /= float(n1d)
        final = self.evaluate_powell_params(params, include_oob=True)
        return PowellBruteForceResult(params=params.copy(), cost=float(final.cost), history=tuple(history))

    def optimize_rigid(
            self,
            init_r2r: np.ndarray,
            *,
            method: str = "Powell",
            maxiter: int = 20,
            x0: np.ndarray | None = None,
            device: str | torch.device = "cpu",
            callback: Callable[[PowellOptimizationStep], None] | None = None,
            options: dict[str, object] | None = None,
    ) -> PowellOptimizationResult:
        """Optimize a rigid transform through the shared :class:`RegModel` parameterization."""
        init_r2r = np.asarray(init_r2r, dtype=np.float64)
        run_device = resolve_cpu_only_device(device, backend_name="Powell rigid optimization")
        init_v2v = convert_transform_type(
            init_r2r,
            src_affine=self.mov.affine,
            dst_affine=self.ref.affine,
            from_type=LINEAR_RAS_TO_RAS,
            to_type=LINEAR_VOX_TO_VOX,
        )
        model = RegModel(
            dof=6,
            v2v_init=torch.from_numpy(init_v2v).float(),
            source_shape=self.mov.shape,
            target_shape=self.ref.shape,
            device=run_device,
        )
        start_weights = np.zeros(6, dtype=np.float64) if x0 is None else np.asarray(x0, dtype=np.float64)
        if start_weights.shape != (6,):
            raise ValueError("x0 must have shape (6,)")

        def evaluate_weights(weights: np.ndarray) -> tuple[float, np.ndarray]:
            with torch.no_grad():
                model.weights.data = torch.from_numpy(np.asarray(weights, dtype=np.float32)).to(run_device)
                v2v = np.asarray(model.get_v2v_from_weights(self.mov.shape, self.ref.shape), dtype=np.float64)
            r2r = convert_transform_type(
                v2v,
                src_affine=self.mov.affine,
                dst_affine=self.ref.affine,
                from_type=LINEAR_VOX_TO_VOX,
                to_type=LINEAR_RAS_TO_RAS,
            )
            cost = self.evaluate_r2r(r2r).cost
            return float(cost), np.asarray(r2r, dtype=np.float64)

        initial_cost = self.evaluate_r2r(init_r2r).cost
        initial_r2r = init_r2r.copy()
        history: list[PowellOptimizationStep] = []

        def objective(weights: np.ndarray) -> float:
            cost, _ = evaluate_weights(weights)
            return cost

        def scipy_callback(weights: np.ndarray) -> None:
            cost, r2r = evaluate_weights(weights)
            step = PowellOptimizationStep(
                iteration=len(history),
                cost=float(cost),
                r2r=r2r,
                weights=np.asarray(weights, dtype=np.float64).copy(),
            )
            history.append(step)
            if callback is not None:
                callback(step)

        resolved_options = {"maxiter": int(maxiter)}
        if options is not None:
            resolved_options.update(options)
        result = minimize(objective, start_weights, method=method, callback=scipy_callback, options=resolved_options)
        final_cost, final_r2r = evaluate_weights(result.x)
        return PowellOptimizationResult(
            success=bool(result.success),
            message=str(result.message),
            nit=int(getattr(result, "nit", 0)),
            nfev=int(getattr(result, "nfev", 0)),
            initial_cost=float(initial_cost),
            final_cost=float(final_cost),
            initial_r2r=initial_r2r,
            final_r2r=final_r2r,
            weights=np.asarray(result.x, dtype=np.float64).copy(),
            history=tuple(history),
        )

    def optimize_powell_params(
            self,
            init_params: np.ndarray,
            *,
            brute_force_limit: float = 30.0,
            brute_force_iters: int = 1,
            brute_force_samples: int = 30,
            powell_maxiter: int = 4,
            callback: Callable[[PowellOptimizationStep], None] | None = None,
            options: dict[str, object] | None = None,
    ) -> PowellOptimizationResult:
        """Run brute-force initialization followed by SciPy Powell refinement."""
        init_params = np.asarray(init_params, dtype=np.float64)
        init_r2r = powell_params_to_mov_to_ref_r2r(init_params)
        initial_cost = self.evaluate_r2r(init_r2r).cost
        brute = self.brute_force_search(
            init_params,
            limit=brute_force_limit,
            niters=brute_force_iters,
            n1d=brute_force_samples,
            callback=callback,
        )
        history: list[PowellOptimizationStep] = list(brute.history)

        def objective(params: np.ndarray) -> float:
            return self.evaluate_powell_params(params, include_oob=False).cost

        def scipy_callback(params: np.ndarray) -> None:
            result = self.evaluate_powell_params(params, include_oob=False)
            step = PowellOptimizationStep(
                iteration=len(history),
                cost=float(result.cost),
                r2r=powell_params_to_mov_to_ref_r2r(params),
                weights=np.asarray(params, dtype=np.float64).copy(),
            )
            history.append(step)
            if callback is not None:
                callback(step)

        resolved_options = {"maxiter": int(powell_maxiter), "xtol": 1e-2, "ftol": 1e-4}
        if options is not None:
            resolved_options.update(options)
        result = minimize(objective, brute.params, method="Powell", callback=scipy_callback, options=resolved_options)
        final_params = np.asarray(result.x, dtype=np.float64)
        final_r2r = powell_params_to_mov_to_ref_r2r(final_params)
        final_cost = self.evaluate_r2r(final_r2r).cost
        return PowellOptimizationResult(
            success=bool(result.success),
            message=str(result.message),
            nit=int(getattr(result, "nit", 0)),
            nfev=int(getattr(result, "nfev", 0)),
            initial_cost=float(initial_cost),
            final_cost=float(final_cost),
            initial_r2r=init_r2r,
            final_r2r=final_r2r,
            weights=final_params.copy(),
            history=tuple(history),
        )


def powell_params_to_ref_to_mov_r2r(params: np.ndarray) -> np.ndarray:
    """Convert Powell parameters into a reference-to-moving RAS transform."""
    params = np.asarray(params, dtype=np.float64)
    if params.shape[0] < 6:
        raise ValueError("powell params must have at least 6 entries")

    rotation = Rotation.from_euler("XYZ", [-params[3], params[4], -params[5]], degrees=True).as_matrix()
    scale = np.eye(3, dtype=np.float64)
    shear = np.eye(3, dtype=np.float64)
    if params.shape[0] >= 9:
        scale = np.diag(np.asarray(params[6:9], dtype=np.float64))
    if params.shape[0] >= 12:
        shear[0, 1] = float(params[9])
        shear[0, 2] = float(params[10])
        shear[1, 2] = float(params[11])

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation @ scale @ shear
    matrix[:3, 3] = params[:3]
    return matrix


def powell_params_to_mov_to_ref_r2r(params: np.ndarray) -> np.ndarray:
    """Convert Powell parameters into a moving-to-reference RAS transform."""
    return np.linalg.inv(powell_params_to_ref_to_mov_r2r(params))


def powell_mov_to_ref_r2r_to_params(mov_to_ref_r2r: np.ndarray, *, dof: int = 6) -> np.ndarray:
    """Convert a moving-to-reference RAS transform back into Powell parameters."""
    mov_to_ref_r2r = np.asarray(mov_to_ref_r2r, dtype=np.float64)
    if mov_to_ref_r2r.shape != (4, 4):
        raise ValueError("mov_to_ref_r2r must be 4x4")
    if dof not in (6, 9, 12):
        raise ValueError("dof must be one of 6, 9, or 12")

    ref_to_mov_r2r = np.linalg.inv(mov_to_ref_r2r)
    params = np.zeros(dof, dtype=np.float64)
    params[:3] = ref_to_mov_r2r[:3, 3]

    linear = ref_to_mov_r2r[:3, :3]
    rotation, upper = np.linalg.qr(linear)
    signs = np.sign(np.diag(upper))
    signs[signs == 0.0] = 1.0
    sign_matrix = np.diag(signs)
    rotation = rotation @ sign_matrix
    upper = sign_matrix @ upper
    if np.linalg.det(rotation) < 0.0:
        fix = np.diag([1.0, 1.0, -1.0])
        rotation = rotation @ fix
        upper = fix @ upper

    euler = Rotation.from_matrix(rotation).as_euler("XYZ", degrees=True)
    params[3] = -float(euler[0])
    params[4] = float(euler[1])
    params[5] = -float(euler[2])

    if dof >= 9:
        scale = np.diag(upper).copy()
        if np.any(np.isclose(scale, 0.0)):
            raise ValueError("Cannot convert transform with zero scale component to powell params")
        params[6:9] = scale
        if dof == 12:
            params[9] = upper[0, 1] / scale[0]
            params[10] = upper[0, 2] / scale[0]
            params[11] = upper[1, 2] / scale[1]
    return params


def optimize_powell_from_rigid(
        evaluator: PowellCostEvaluator,
        init_r2r: np.ndarray,
        **kwargs: object,
) -> PowellOptimizationResult:
    """Convenience wrapper around :meth:`PowellCostEvaluator.optimize_rigid`."""
    return evaluator.optimize_rigid(init_r2r, **kwargs)


def optimize_powell_from_params(
        evaluator: PowellCostEvaluator,
        init_params: np.ndarray,
        **kwargs: object,
) -> PowellOptimizationResult:
    """Convenience wrapper around :meth:`PowellCostEvaluator.optimize_powell_params`."""
    return evaluator.optimize_powell_params(init_params, **kwargs)


def _prepare_volume(
        data: np.ndarray,
        *,
        affine: np.ndarray,
        sep: int,
        saturation_pct: float,
        seed: int,
        intensity_dither: bool,
        smooth_images: bool,
) -> _PreparedVolume:
    """Prepare a volume for repeated MRI_coreg-style cost evaluation."""
    data = np.asarray(data, dtype=np.float32)
    affine = np.asarray(affine, dtype=np.float64)
    voxel_sizes = np.asarray(nib.affines.voxel_sizes(affine), dtype=np.float64)
    data = _rescale_to_uchar_range(data, saturation_pct)
    if smooth_images:
        sigma = _smooth_sigma(voxel_sizes, sep)
        if np.any(sigma > 0):
            data = gaussian_filter(data, sigma=sigma, mode="nearest")
    data = _quantize_to_uchar(data, seed=seed, intensity_dither=intensity_dither)
    return _PreparedVolume(data=data, affine=affine, shape=tuple(int(v) for v in data.shape), voxel_sizes=voxel_sizes)


def _rescale_to_uchar_range(data: np.ndarray, saturation_pct: float) -> np.ndarray:
    """Rescale intensities into the 8-bit range used by the histogram evaluator."""
    data = np.asarray(data, dtype=np.float32)
    lo = float(np.min(data))
    hi = _percentile_value(data, saturation_pct)
    hi = max(hi, lo)
    if hi <= lo:
        return np.zeros_like(data, dtype=np.float32)
    scaled = (data - lo) * (255.0 / (hi - lo))
    return np.clip(scaled, 0.0, 255.0).astype(np.float32, copy=False)


def _percentile_value(data: np.ndarray, pct: float) -> float:
    """Return a percentile value without sorting the full array."""
    flat = np.asarray(data, dtype=np.float32).ravel()
    if flat.size == 0:
        return 0.0
    pct = float(np.clip(pct, 0.0, 100.0))
    if pct <= 0.0:
        return float(np.min(flat))
    if pct >= 100.0:
        return float(np.max(flat))
    idx = pct * 0.01 * (flat.size - 1)
    lo = int(np.floor(idx))
    hi = int(np.ceil(idx))
    part = np.partition(flat, (lo, hi))
    if lo == hi:
        return float(part[lo])
    frac = idx - lo
    return float((1.0 - frac) * part[lo] + frac * part[hi])


def _smooth_sigma(voxel_sizes: np.ndarray, sep: int) -> np.ndarray:
    """Compute FreeSurfer-style pre-smoothing sigmas for the evaluator grid spacing."""
    sep = float(sep)
    val = np.maximum(sep * sep - voxel_sizes * voxel_sizes, 0.0)
    fwhm = 1.15 * np.sqrt(val)
    return fwhm / np.sqrt(np.log(_HIST_SIZE))


def _quantize_to_uchar(data: np.ndarray, *, seed: int, intensity_dither: bool) -> np.ndarray:
    """Quantize float intensities into the evaluator's 8-bit histogram domain."""
    rng = np.random.default_rng(seed)
    if intensity_dither:
        data = data + rng.random(size=data.shape, dtype=np.float32)
        data = np.floor(data)
    else:
        data = np.rint(data)
    return np.clip(data, 0.0, 255.0).astype(np.float32, copy=False)


def _sample_reference_grid(shape: tuple[int, int, int], sep: int, seed: int, coord_dither: bool) -> np.ndarray:
    """Sample the reference lattice on a coarse grid with optional coordinate dithering."""
    axes = [np.arange(0, dim, sep, dtype=np.float32) for dim in shape]
    mesh = np.meshgrid(*axes, indexing="ij")
    coords = np.stack([m.reshape(-1) for m in mesh], axis=0)
    if coord_dither:
        rng = np.random.default_rng(seed)
        coords += rng.random(size=coords.shape, dtype=np.float32) * float(sep)
        for axis, dim in enumerate(shape):
            np.clip(coords[axis], 0.0, float(dim - 1), out=coords[axis])
    return coords


def _coords_in_bounds(coords: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Return a mask indicating which sampled coordinates lie inside a volume."""
    lower_ok = coords >= 0.0
    upper_ok = coords <= (np.asarray(shape, dtype=np.float64)[:, None] - 1.0)
    return np.all(lower_ok & upper_ok, axis=0)


def _joint_histogram(ref_vals: np.ndarray, mov_vals: np.ndarray) -> np.ndarray:
    """Build the soft 8-bit joint histogram used by the NMI cost."""
    ref_vals = np.clip(np.asarray(ref_vals, dtype=np.float32), 0.0, 255.0)
    mov_vals = np.clip(np.asarray(mov_vals, dtype=np.float32), 0.0, 255.0)
    mov_floor = np.floor(mov_vals).astype(np.int32)
    mov_frac = mov_vals - mov_floor
    ref_bin = np.clip(np.floor(ref_vals + 0.5).astype(np.int32), 0, 255)
    mov_floor = np.clip(mov_floor, 0, 255)

    hist = np.zeros((_HIST_SIZE, _HIST_SIZE), dtype=np.float64)
    lin0 = ref_bin * _HIST_SIZE + mov_floor
    hist += np.bincount(lin0, weights=1.0 - mov_frac, minlength=_HIST_SIZE * _HIST_SIZE).reshape(_HIST_SIZE, _HIST_SIZE)

    upper = mov_floor < (_HIST_SIZE - 1)
    if np.any(upper):
        lin1 = ref_bin[upper] * _HIST_SIZE + (mov_floor[upper] + 1)
        hist += np.bincount(lin1, weights=mov_frac[upper], minlength=_HIST_SIZE * _HIST_SIZE).reshape(
            _HIST_SIZE, _HIST_SIZE
        )
    return hist


def _gaussian_kernel(std: float, lim: int) -> np.ndarray:
    """Return a normalized 1-D Gaussian kernel."""
    x = np.arange(-lim, lim + 1, dtype=np.float64)
    kernel = np.exp(-(x * x) / (2.0 * std * std))
    kernel /= kernel.sum()
    return kernel


def _nmi_cost(hist: np.ndarray) -> float:
    """Compute the negative normalized mutual information cost from a joint histogram."""
    row = hist.sum(axis=1)
    col = hist.sum(axis=0)
    hab = np.sum(hist * np.log2(hist))
    ha = np.sum(row * np.log2(row))
    hb = np.sum(col * np.log2(col))
    return -((ha + hb) / hab)


def _shape3(shape: torch.Size | tuple[int, ...]) -> tuple[int, int, int]:
    """Return the leading three spatial dimensions as an explicit 3-tuple."""
    return int(shape[0]), int(shape[1]), int(shape[2])


def register_powell_coreg(
        src: str | nib.Nifti1Image,
        trg: str | nib.Nifti1Image,
        lta_name: str | None = None,
        mapped_name: str | None = None,
        return_v2v: bool = False,
        init_type: InitType = "image_center",
        dof: int = 6,
        brute_force_limit: float = 30.0,
        brute_force_iters: int = 1,
        brute_force_samples: int = 30,
        powell_maxiter: int = 4,
        sep: int = 4,
        device: str | torch.device = "cpu",
        trace_fn=None,
) -> torch.Tensor:
    """Run the MRI_coreg-style Powell registration path.

    This backend derives a starting transform from the requested initialization
    mode, converts it into the Powell parameterization, performs a coarse
    brute-force sweep over the leading pose parameters, and then refines the
    result with SciPy's Powell optimizer against the cached NMI evaluator.

    Parameters
    ----------
    src, trg : str or nibabel image
        Moving and reference images.
    lta_name, mapped_name : str or None, optional
        Optional output paths for the final LTA and mapped moving volume.
    return_v2v : bool, default=False
        Return voxel-to-voxel instead of RAS-to-RAS when requested.
    init_type : {"header", "centroid", "image_center"}, default="image_center"
        Initialization mode used before the brute-force sweep.
    dof : {6, 9, 12}, default=6
        Powell parameterization to optimize.
    brute_force_limit, brute_force_iters, brute_force_samples
        Coarse search settings.
    powell_maxiter : int, default=4
        Maximum Powell refinement iterations.
    sep : int, default=4
        Sampling spacing for the evaluator grid.
    device : str or torch.device, default="cpu"
        Requested device. The Powell backend currently resolves this to CPU.
    trace_fn : callable, optional
        Optional callback receiving run and iteration events.

    Returns
    -------
    torch.Tensor
        Final RAS-to-RAS transform by default, or voxel-to-voxel when
        ``return_v2v=True``.
    """
    if dof not in (6, 9, 12):
        raise ValueError("method='powell' currently supports dof=6, 9, or 12")

    start = time.perf_counter()
    run_device = resolve_cpu_only_device(device, backend_name="Powell coreg")
    resolved_init_type = resolve_init_type(init_type=init_type, default_init_type="image_center")
    logger.debug("Powell coreg running on %s", run_device)
    if isinstance(src, str):
        src = nib.load(src)
    if isinstance(trg, str):
        trg = nib.load(trg)

    src_affine_t = torch.from_numpy(np.asarray(src.affine, dtype=np.float64))
    trg_affine_t = torch.from_numpy(np.asarray(trg.affine, dtype=np.float64))
    sdata_full = torch.from_numpy(np.asarray(src.get_fdata(dtype=np.float32), dtype=np.float32))
    tdata_full = torch.from_numpy(np.asarray(trg.get_fdata(dtype=np.float32), dtype=np.float32))
    init_v2v = get_init_vox2vox(
        sdata_full,
        tdata_full,
        saffine=src_affine_t.float(),
        taffine=trg_affine_t.float(),
        init_type=resolved_init_type,
    )
    init_r2r = trg_affine_t @ init_v2v.double() @ torch.inverse(src_affine_t)
    init_params = powell_mov_to_ref_r2r_to_params(np.asarray(init_r2r.cpu(), dtype=np.float64), dof=dof)

    evaluator = PowellCostEvaluator(src, trg, sep=sep)

    def _step_trace(step):
        if trace_fn is None:
            return
        v2v_iter = convert_transform_type(
            step.r2r,
            src_affine=np.asarray(src.affine, dtype=np.float64),
            dst_affine=np.asarray(trg.affine, dtype=np.float64),
            from_type=LINEAR_RAS_TO_RAS,
            to_type=LINEAR_VOX_TO_VOX,
        )
        trace_fn(
            event="iter_end",
            iteration=step.iteration,
            loss=step.cost,
            v2v=torch.from_numpy(np.asarray(v2v_iter, dtype=np.float64)),
        )

    if trace_fn is not None:
        trace_fn(event="run_start", method="powell", dof=dof)
    result = evaluator.optimize_powell_params(
        init_params,
        brute_force_limit=brute_force_limit,
        brute_force_iters=brute_force_iters,
        brute_force_samples=brute_force_samples,
        powell_maxiter=powell_maxiter,
        callback=_step_trace,
    )

    Mr2r = torch.from_numpy(np.asarray(result.final_r2r, dtype=np.float64)).double()
    Mv2v_orig = torch.from_numpy(
        convert_transform_type(
            result.final_r2r,
            src_affine=np.asarray(src.affine, dtype=np.float64),
            dst_affine=np.asarray(trg.affine, dtype=np.float64),
            from_type=LINEAR_RAS_TO_RAS,
            to_type=LINEAR_VOX_TO_VOX,
        )
    ).double()

    if lta_name is not None:
        logger.info("Writing final LTA file: %s", lta_name)
        LTA.from_matrix(Mr2r.numpy(), src.get_filename(), src, trg.get_filename(), trg).write(lta_name)
    if mapped_name is not None:
        logger.info("Writing mapped image: %s", mapped_name)
        mapped_img = reslice_r2r_image(
            src,
            Mr2r.numpy(),
            target_affine=trg.affine,
            target_shape=_shape3(trg.shape),
            mode="linear",
        )
        mapped_img.to_filename(mapped_name)

    logger.info("register_powell_coreg total time: %.2f s", time.perf_counter() - start)
    if return_v2v:
        return Mv2v_orig
    return Mr2r


__all__ = [
    "PowellBruteForceResult",
    "PowellCostEvaluator",
    "PowellCostResult",
    "PowellOptimizationResult",
    "PowellOptimizationStep",
    "powell_mov_to_ref_r2r_to_params",
    "powell_params_to_mov_to_ref_r2r",
    "powell_params_to_ref_to_mov_r2r",
    "optimize_powell_from_params",
    "optimize_powell_from_rigid",
    "register_powell_coreg",
]
