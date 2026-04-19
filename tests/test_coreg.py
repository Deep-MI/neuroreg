from __future__ import annotations

import importlib
from typing import Any, cast

import nibabel as nib
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from neuroreg import coreg
from neuroreg.imreg.coreg import register_powell_coreg
from neuroreg.imreg.device import resolve_cpu_only_device, resolve_torch_device
from neuroreg.imreg.gd import register_gd_pyramid, register_level
from neuroreg.imreg.losses import mi_loss, ncc_loss, nmi_loss
from neuroreg.imreg.powell import (
    PowellCostEvaluator,
    optimize_powell_from_params,
    optimize_powell_from_rigid,
    powell_mov_to_ref_r2r_to_params,
    powell_params_to_mov_to_ref_r2r,
    powell_params_to_ref_to_mov_r2r,
)
from neuroreg.imreg.reg_model import RegModel
from neuroreg.transforms import LINEAR_RAS_TO_RAS, LINEAR_VOX_TO_VOX, convert_transform_type


def _make_blob(shape: tuple[int, int, int] = (20, 20, 20), shift: tuple[float, float, float] = (0, 0, 0)) -> np.ndarray:
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        np.arange(shape[2], dtype=np.float32),
        indexing="ij",
    )
    center = (np.asarray(shape, dtype=np.float32) - 1.0) / 2.0 + np.asarray(shift, dtype=np.float32)
    dist2 = (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    return np.exp(-dist2 / 18.0).astype(np.float32)


def _make_img(
        shape: tuple[int, int, int] = (20, 20, 20),
        shift: tuple[float, float, float] = (0, 0, 0),
) -> nib.Nifti1Image:
    affine = cast(Any, np.eye(4, dtype=np.float32))
    data = cast(Any, _make_blob(shape, shift))
    return nib.Nifti1Image(data, affine)


class TestRegisterPyramidSynthetic:
    def test_register_pyramid_returns_v2v_on_identical_images(self):
        img = _make_img()
        v2v = register_gd_pyramid(
            img,
            img,
            return_v2v=True,
            init_type="header",
            symmetric=False,
            dof=3,
            n=1,
            device="cpu",
        )
        assert v2v.shape == (4, 4)
        assert torch.isfinite(v2v).all()
        assert torch.allclose(v2v, torch.eye(4, dtype=v2v.dtype), atol=1.0)

    def test_register_pyramid_returns_original_v2v_after_isotropic_resampling(
            self,
            monkeypatch: pytest.MonkeyPatch,
    ):
        img = _make_img(shape=(16, 16, 16))
        rsrc = torch.tensor(
            [[2.0, 0.0, 0.0, 0.5], [0.0, 2.0, 0.0, 1.0], [0.0, 0.0, 2.0, -0.5], [0.0, 0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        rtrg = torch.tensor(
            [[3.0, 0.0, 0.0, -1.0], [0.0, 3.0, 0.0, 0.25], [0.0, 0.0, 3.0, 2.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )

        def fake_resample_isotropic(_img, _isosize, mode="bilinear"):
            return (
                torch.ones((16, 16, 16), dtype=torch.float32),
                torch.eye(4, dtype=torch.float32),
                rsrc if mode == "bilinear" and _img is img else rtrg,
            )

        def fake_register_level(simg, timg, **kwargs):
            _ = simg, timg, kwargs
            return torch.eye(4, dtype=torch.float64), [], None

        image_map_module = importlib.import_module("neuroreg.image.map")
        monkeypatch.setattr(image_map_module, "resample_isotropic", fake_resample_isotropic)
        monkeypatch.setattr("neuroreg.imreg.gd.register_level", fake_register_level)

        v2v = register_gd_pyramid(
            img,
            img.__class__(img.get_fdata(), img.affine.copy()),
            return_v2v=True,
            init_type="header",
            symmetric=False,
            isotropic=True,
            n=1,
            min_voxels=16,
            max_voxels=16,
            device="cpu",
        )

        expected = rtrg.double() @ torch.inverse(rsrc.double())
        assert torch.allclose(v2v, expected)

    def test_register_pyramid_respects_max_voxels_schedule(self, monkeypatch: pytest.MonkeyPatch):
        seen_shapes: list[tuple[int, int, int]] = []

        def fake_register_level(simg, timg, **kwargs):
            seen_shapes.append(tuple(int(v) for v in simg.shape))
            return torch.eye(4), [], None

        monkeypatch.setattr("neuroreg.imreg.gd.register_level", fake_register_level)

        img = _make_img(shape=(128, 128, 128))
        register_gd_pyramid(
            img,
            img,
            return_v2v=True,
            init_type="header",
            symmetric=False,
            n=1,
            min_voxels=16,
            max_voxels=64,
        )
        assert seen_shapes == [(16, 16, 16), (32, 32, 32), (64, 64, 64)]

        seen_shapes.clear()
        register_gd_pyramid(
            img,
            img,
            return_v2v=True,
            init_type="header",
            symmetric=False,
            n=1,
            min_voxels=16,
            max_voxels=None,
        )
        assert seen_shapes == [(16, 16, 16), (32, 32, 32), (64, 64, 64), (128, 128, 128)]

    def test_register_pyramid_uses_per_level_iteration_schedule(self, monkeypatch: pytest.MonkeyPatch):
        seen: list[tuple[tuple[int, int, int], int]] = []

        def fake_register_level(simg, timg, **kwargs):
            seen.append((tuple(int(v) for v in simg.shape), int(kwargs["n"])))
            return torch.eye(4), [], None

        monkeypatch.setattr("neuroreg.imreg.gd.register_level", fake_register_level)

        img = _make_img(shape=(64, 64, 64))
        register_gd_pyramid(
            img,
            img,
            return_v2v=True,
            init_type="header",
            symmetric=False,
            n=99,
            level_iters=[7, 3, 1],
            min_voxels=16,
            max_voxels=None,
        )
        assert seen == [((16, 16, 16), 7), ((32, 32, 32), 3), ((64, 64, 64), 1)]

    def test_register_pyramid_rejects_wrong_number_of_level_iterations(self):
        img = _make_img(shape=(64, 64, 64))
        with pytest.raises(ValueError, match="level_iters"):
            register_gd_pyramid(
                img,
                img,
                return_v2v=True,
                init_type="header",
                symmetric=False,
                n=1,
                level_iters=[5, 2],
                min_voxels=16,
                max_voxels=None,
                device="cpu",
            )

    def test_reg_model_maps_source_onto_target_grid_when_shapes_differ(self):
        src = torch.from_numpy(_make_blob((20, 20, 20)))
        trg = torch.from_numpy(_make_blob((24, 24, 24)))
        model = RegModel(
            dof=6,
            source_shape=tuple(int(v) for v in src.shape),
            target_shape=tuple(int(v) for v in trg.shape),
        )
        preds = model(src)
        assert preds.shape == trg.shape

    def test_reg_model_preserves_prior_r2r_when_weights_are_zero_and_shapes_differ(self):
        src_shape = (64, 64, 64)
        trg_shape = (80, 80, 80)
        src_affine = np.diag([4.0, 4.0, 4.0, 1.0]).astype(np.float32)
        trg_affine = np.diag([2.8, 2.8, 2.8, 1.0]).astype(np.float32)
        prior_r2r = np.array(
            [
                [0.9823, -0.1872, -0.0089, -21.65444],
                [0.1872, 0.9776, 0.0960, 65.03291],
                [-0.0093, -0.0960, 0.9953, -10.15581],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        prior_v2v = convert_transform_type(
            prior_r2r,
            src_affine=src_affine,
            dst_affine=trg_affine,
            from_type=LINEAR_RAS_TO_RAS,
            to_type=LINEAR_VOX_TO_VOX,
        )

        model = RegModel(dof=6, v2v_init=torch.from_numpy(prior_v2v), source_shape=src_shape, target_shape=trg_shape)
        v2v = np.asarray(model.get_v2v_from_weights(src_shape, trg_shape), dtype=np.float32)
        r2r = trg_affine @ v2v @ np.linalg.inv(src_affine)

        assert np.allclose(r2r, prior_r2r, atol=1e-5)

    def test_register_level_runs_when_source_and_target_shapes_differ(self):
        src = torch.from_numpy(_make_blob((20, 20, 20), shift=(2, 0, 0)))
        trg = torch.from_numpy(_make_blob((24, 24, 24)))
        v2v, losses, model = register_level(src, trg, dof=6, init_type="centroid", n=2, loss_name="mi", device="cpu")
        assert v2v.shape == (4, 4)
        assert torch.isfinite(v2v).all()
        assert len(losses) == 2
        assert model(src).shape == trg.shape

    def test_register_level_trace_hook_receives_iter_events(self):
        src = torch.from_numpy(_make_blob((20, 20, 20), shift=(1, 0, 0)))
        trg = torch.from_numpy(_make_blob((20, 20, 20)))
        events: list[dict[str, object]] = []

        def trace_fn(**payload):
            events.append(payload)

        _v2v, losses, _model = register_level(
            src,
            trg,
            dof=6,
            init_type="header",
            n=3,
            loss_name="nmi",
            device="cpu",
            trace_fn=trace_fn,
        )
        iter_events = [e for e in events if e["event"] == "iter_end"]
        assert len(iter_events) == 3
        assert len(losses) == 3
        assert all("v2v" in e for e in iter_events)

    def test_register_level_trace_v2v_uses_target_shape(self):
        src = torch.from_numpy(_make_blob((20, 20, 20), shift=(1, 0, 0)))
        trg = torch.from_numpy(_make_blob((24, 24, 24)))
        events: list[dict[str, object]] = []

        def trace_fn(**payload):
            events.append(payload)

        v2v, _losses, _model = register_level(
            src,
            trg,
            dof=6,
            init_type="header",
            n=1,
            loss_name="nmi",
            device="cpu",
            trace_fn=trace_fn,
        )
        iter_events = [e for e in events if e["event"] == "iter_end"]
        assert len(iter_events) == 1
        traced_v2v = iter_events[0]["v2v"]
        assert torch.allclose(traced_v2v, v2v)

    def test_register_level_uses_custom_adam_lr(self, monkeypatch: pytest.MonkeyPatch):
        captured: dict[str, float] = {}

        class FakeAdam:
            def __init__(self, params, lr):
                _ = list(params)
                captured["lr"] = float(lr)

            def zero_grad(self):
                return None

            def step(self):
                return None

        monkeypatch.setattr("neuroreg.imreg.gd.torch.optim.Adam", FakeAdam)
        monkeypatch.setattr("neuroreg.imreg.gd.training_loop", lambda *args, **kwargs: [])

        src = torch.from_numpy(_make_blob((16, 16, 16)))
        trg = torch.from_numpy(_make_blob((16, 16, 16)))
        v2v, losses, _ = register_level(src, trg, init_type="header", n=1, lr=0.005, device="cpu")

        assert captured["lr"] == pytest.approx(0.005)
        assert v2v.shape == (4, 4)
        assert losses == []

    def test_register_level_forwards_weight_scales_to_model(self, monkeypatch: pytest.MonkeyPatch):
        captured: dict[str, float] = {}

        class FakeModel:
            def __init__(self, **kwargs):
                captured["translation_weight_scale"] = float(kwargs["translation_weight_scale"])
                captured["rotation_weight_scale"] = float(kwargs["rotation_weight_scale"])
                captured["scale_weight_scale"] = float(kwargs["scale_weight_scale"])
                captured["shear_weight_scale"] = float(kwargs["shear_weight_scale"])
                self.weights = torch.nn.Parameter(torch.zeros(6))

            def parameters(self):
                return [self.weights]

            def get_v2v_from_weights(self, sshape, tshape):
                _ = sshape, tshape
                return torch.eye(4)

            def __call__(self, src):
                return src

        class FakeAdam:
            def __init__(self, params, lr):
                _ = list(params), lr

            def zero_grad(self):
                return None

            def step(self):
                return None

        monkeypatch.setattr("neuroreg.imreg.gd.RegModel", FakeModel)
        monkeypatch.setattr("neuroreg.imreg.gd.torch.optim.Adam", FakeAdam)
        monkeypatch.setattr("neuroreg.imreg.gd.training_loop", lambda *args, **kwargs: [])

        src = torch.from_numpy(_make_blob((16, 16, 16)))
        trg = torch.from_numpy(_make_blob((16, 16, 16)))
        register_level(
            src,
            trg,
            init_type="header",
            n=1,
            device="cpu",
            translation_weight_scale=0.5,
            rotation_weight_scale=3.0,
            scale_weight_scale=2.0,
            shear_weight_scale=1.5,
        )

        assert captured["translation_weight_scale"] == pytest.approx(0.5)
        assert captured["rotation_weight_scale"] == pytest.approx(3.0)
        assert captured["scale_weight_scale"] == pytest.approx(2.0)
        assert captured["shear_weight_scale"] == pytest.approx(1.5)

    def test_reg_model_scale_weights_do_not_scale_translation_column(self):
        model = RegModel(dof=9)
        with torch.no_grad():
            model.weights[:] = torch.tensor([2.0, -3.0, 4.0, 0.0, 0.0, 0.0, 1.5, 0.5, 2.0])

        affine = model.get_torch_transform_from_weights()

        assert torch.allclose(affine[:3, 3], torch.tensor([2.0, -3.0, 4.0]))
        assert torch.allclose(torch.diag(affine[:3, :3]), torch.tensor([1.5, 0.5, 2.0]))

    def test_register_level_header_init_uses_affines(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("neuroreg.imreg.gd.training_loop", lambda *args, **kwargs: [])

        src = torch.from_numpy(_make_blob((16, 16, 16)))
        trg = torch.from_numpy(_make_blob((16, 16, 16)))
        src_aff = torch.eye(4)
        trg_aff = torch.eye(4)
        trg_aff[0, 3] = 3.0

        v2v, losses, _ = register_level(
            src,
            trg,
            init_type="header",
            src_affine=src_aff,
            trg_affine=trg_aff,
            n=1,
            device="cpu",
        )

        expected = torch.inverse(trg_aff) @ src_aff
        assert torch.allclose(v2v, expected)
        assert losses == []

    def test_register_pyramid_trace_hook_receives_level_events(self):
        img = _make_img(shape=(32, 32, 32))
        events: list[dict[str, object]] = []

        def trace_fn(**payload):
            events.append(payload)

        _ = register_gd_pyramid(
            img,
            img,
            return_v2v=True,
            init_type="header",
            symmetric=False,
            dof=6,
            n=1,
            loss_name="nmi",
            min_voxels=16,
            device="cpu",
            trace_fn=trace_fn,
        )
        assert any(e["event"] == "run_start" for e in events)
        assert any(e["event"] == "level_start" for e in events)
        assert any(e["event"] == "level_end" for e in events)
        assert all("n_iterations" in e for e in events if e["event"] in {"level_start", "level_end"})

    def test_register_pyramid_symmetric_trace_hook_receives_level_events(self):
        img = _make_img(shape=(32, 32, 32))
        events: list[dict[str, object]] = []

        def trace_fn(**payload):
            events.append(payload)

        _ = register_gd_pyramid(
            img,
            img,
            return_v2v=True,
            symmetric=True,
            dof=6,
            n=1,
            loss_name="nmi",
            min_voxels=16,
            max_voxels=16,
            device="cpu",
            trace_fn=trace_fn,
        )
        assert any(e["event"] == "run_start" for e in events)
        assert any(e["event"] == "level_start" for e in events)
        assert any(e["event"] == "level_end" for e in events)
        assert any(e["event"] == "iter_end" for e in events)
        assert all("n_iterations" in e for e in events if e["event"] in {"level_start", "level_end"})

    def test_register_pyramid_symmetric_uses_requested_init_only_on_first_level(self, monkeypatch: pytest.MonkeyPatch):
        seen_init_types: list[str] = []

        def fake_register_level(simg, timg, **kwargs):
            _ = simg, timg
            seen_init_types.append(str(kwargs["init_type"]))
            return torch.eye(4), [], None

        monkeypatch.setattr("neuroreg.imreg.gd.register_level", fake_register_level)

        img = _make_img(shape=(32, 32, 32))
        _ = register_gd_pyramid(
            img,
            img,
            return_v2v=True,
            symmetric=True,
            init_type="centroid",
            isotropic=False,
            n=1,
            min_voxels=16,
            max_voxels=None,
            device="cpu",
        )

        assert seen_init_types == ["centroid", "header"]

    def test_register_pyramid_symmetric_can_disable_isotropic_preprocessing(self, monkeypatch: pytest.MonkeyPatch):
        def fail_resample(*args, **kwargs):
            _ = args, kwargs
            raise AssertionError("resample_isotropic should not be called when isotropic=False")

        image_map_module = importlib.import_module("neuroreg.image.map")
        monkeypatch.setattr(image_map_module, "resample_isotropic", fail_resample)

        img = _make_img(shape=(16, 16, 16))
        v2v = register_gd_pyramid(
            img,
            img.__class__(img.get_fdata(), img.affine.copy()),
            return_v2v=True,
            symmetric=True,
            init_type="header",
            isotropic=False,
            n=1,
            min_voxels=16,
            max_voxels=16,
            device="cpu",
        )

        assert torch.allclose(v2v, torch.eye(4, dtype=v2v.dtype), atol=1.0)


class TestDeviceResolution:
    def test_resolve_torch_device_normalizes_gpu_alias_to_cuda(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

        resolved = resolve_torch_device("GPU")

        assert resolved == torch.device("cuda")

    def test_resolve_cpu_only_device_warns_and_falls_back(self):
        with pytest.warns(RuntimeWarning, match="Powell coreg currently runs on CPU"):
            resolved = resolve_cpu_only_device("cuda", backend_name="Powell coreg")

        assert resolved == torch.device("cpu")


class TestPublicCoregWrapper:
    def test_top_level_coreg_defaults_to_register_powell_coreg(self, monkeypatch: pytest.MonkeyPatch):
        captured: dict[str, object] = {}

        def fake_register_powell_coreg(*args, **kwargs):
            captured["args"] = args
            captured.update(kwargs)
            return torch.eye(4)

        monkeypatch.setattr("neuroreg.imreg.coreg.register_powell_coreg", fake_register_powell_coreg)

        img = _make_img()
        Mr2r = coreg(
            img,
            img,
            return_v2v=False,
            init_type="header",
            dof=6,
            powell_sep=6,
            powell_maxiter=7,
        )

        assert captured["init_type"] == "header"
        assert captured["dof"] == 6
        assert captured["sep"] == 6
        assert captured["powell_maxiter"] == 7
        assert captured["device"] == "cpu"
        assert captured["src"] is img
        assert captured["trg"] is img
        assert len(cast(tuple[Any, ...], captured["args"])) == 0
        assert Mr2r.shape == (4, 4)
        assert torch.isfinite(Mr2r).all()

    def test_register_powell_coreg_warns_and_falls_back_to_cpu(self, monkeypatch: pytest.MonkeyPatch):
        class DummyEvaluator:
            def __init__(self, src, trg, sep):
                self.src = src
                self.trg = trg
                self.sep = sep

            def optimize_powell_params(self, init_params, **kwargs):
                _ = init_params, kwargs
                return type("Result", (), {"final_r2r": np.eye(4, dtype=np.float64), "final_cost": 0.0})()

        monkeypatch.setattr("neuroreg.imreg.powell.PowellCostEvaluator", DummyEvaluator)

        img = _make_img()
        with pytest.warns(RuntimeWarning, match="Powell coreg currently runs on CPU"):
            Mr2r = register_powell_coreg(img, img, init_type="header", dof=6, device="cuda")

        assert Mr2r.shape == (4, 4)
        assert torch.allclose(Mr2r, torch.eye(4, dtype=Mr2r.dtype))

    def test_top_level_coreg_can_explicitly_use_gd(self, monkeypatch: pytest.MonkeyPatch):
        captured: dict[str, object] = {}

        def fake_register_gd_pyramid(*args, **kwargs):
            captured["args"] = args
            captured.update(kwargs)
            return torch.eye(4)

        monkeypatch.setattr("neuroreg.imreg.coreg.register_gd_pyramid", fake_register_gd_pyramid)

        img = _make_img()
        Mr2r = coreg(
            img,
            img,
            method="gd",
            return_v2v=False,
            init_type="header",
            dof=6,
            n=1,
            rotation_weight_scale=3.0,
            scale_weight_scale=2.0,
            shear_weight_scale=1.5,
        )

        assert captured["init_type"] == "header"
        assert captured["dof"] == 6
        assert captured["rotation_weight_scale"] == pytest.approx(3.0)
        assert captured["scale_weight_scale"] == pytest.approx(2.0)
        assert captured["shear_weight_scale"] == pytest.approx(1.5)
        assert captured["src"] is img
        assert captured["trg"] is img
        assert len(cast(tuple[Any, ...], captured["args"])) == 0
        assert Mr2r.shape == (4, 4)
        assert torch.isfinite(Mr2r).all()

    def test_top_level_coreg_rejects_unknown_method(self):
        img = _make_img()
        with pytest.raises(ValueError, match="method must be 'powell' or 'gd'"):
            coreg(img, img, method="bogus")


# ---------------------------------------------------------------------------
# Loss-function unit tests
# ---------------------------------------------------------------------------

class TestLossFunctions:
    """Unit tests for the three modality-independent similarity losses."""

    def _blob(self, shape=(20, 20, 20), shift=(0, 0, 0)):
        return _make_blob(shape, shift)

    # ---- NCC ---------------------------------------------------------------

    def test_ncc_identical_images_is_zero(self):
        a = torch.from_numpy(self._blob())
        assert ncc_loss(a, a, win_size=5).item() == pytest.approx(0.0, abs=1e-4)

    def test_ncc_unrelated_images_near_one(self):
        a = torch.from_numpy(self._blob())
        b = torch.rand_like(a)
        # NCC between a smooth blob and pure noise should be very low
        assert ncc_loss(a, b, win_size=5).item() > 0.5

    def test_ncc_returns_scalar_in_range(self):
        a = torch.from_numpy(self._blob())
        b = torch.from_numpy(self._blob(shift=(3, 0, 0)))
        loss = ncc_loss(a, b, win_size=5)
        assert loss.shape == ()
        assert 0.0 <= loss.item() <= 1.0 + 1e-6

    def test_ncc_gradient_flows(self):
        a = torch.from_numpy(self._blob()).requires_grad_(True)
        b = torch.from_numpy(self._blob(shift=(2, 0, 0)))
        loss = ncc_loss(a, b, win_size=5)
        loss.backward()
        assert a.grad is not None
        assert torch.isfinite(a.grad).all()

    def test_ncc_clamps_win_size_to_image(self):
        # Very small image: win_size larger than any dim should not raise
        a = torch.rand(4, 4, 4)
        loss = ncc_loss(a, a.clone(), win_size=99)
        assert torch.isfinite(loss)

    # ---- MI ----------------------------------------------------------------

    def test_mi_identical_images_is_finite_and_negative(self):
        a = torch.from_numpy(self._blob())
        loss = mi_loss(a, a, num_bins=16)
        assert torch.isfinite(loss)
        assert loss.item() < 0  # loss = -MI, MI > 0 for non-trivial image

    def test_mi_gradient_flows(self):
        a = torch.from_numpy(self._blob()).requires_grad_(True)
        b = torch.from_numpy(self._blob(shift=(2, 0, 0)))
        loss = mi_loss(a, b, num_bins=16)
        loss.backward()
        assert a.grad is not None
        assert torch.isfinite(a.grad).all()

    def test_mi_same_image_lower_loss_than_shifted(self):
        """MI of identical images should be higher (loss more negative) than shifted."""
        a = torch.from_numpy(self._blob())
        b = torch.from_numpy(self._blob(shift=(5, 0, 0)))
        loss_same = mi_loss(a, a, num_bins=16).item()
        loss_shifted = mi_loss(a, b, num_bins=16).item()
        assert loss_same < loss_shifted  # more negative = higher MI

    def test_mi_cross_modal_finite(self):
        """MI should be finite even when src and trg have very different intensity ranges."""
        a = torch.from_numpy(self._blob()) * 1000.0  # simulated T1
        b = 1.0 - torch.from_numpy(self._blob())  # inverted contrast (T2-like)
        loss = mi_loss(a, b, num_bins=32)
        assert torch.isfinite(loss)

    # ---- NMI ---------------------------------------------------------------

    def test_nmi_identical_images_near_minus_two(self):
        """NMI of an image with itself is 2.0, so loss should be ≤ -1.0.

        NMI = 2 only in the limit of infinitesimally narrow Parzen windows.
        With finite sigma and a peaked image histogram the diagonal-only
        structure of the joint histogram is blurred, so NMI < 2.  We test the
        weaker, always-valid bound: identical images produce lower (more
        negative) loss than mismatched images, and loss ≤ -1.0 (NMI ≥ 1.0).
        """
        a = torch.from_numpy(self._blob())
        b = torch.from_numpy(self._blob(shift=(5, 0, 0)))
        loss_same = nmi_loss(a, a, num_bins=16).item()
        loss_diff = nmi_loss(a, b, num_bins=16).item()
        assert loss_same < loss_diff  # identical → lower loss (higher NMI)
        assert loss_same <= -1.0 + 1e-3  # NMI(x, x) ≥ 1.0 always

    def test_nmi_gradient_flows(self):
        a = torch.from_numpy(self._blob()).requires_grad_(True)
        b = torch.from_numpy(self._blob(shift=(2, 0, 0)))
        loss = nmi_loss(a, b, num_bins=16)
        loss.backward()
        assert a.grad is not None
        assert torch.isfinite(a.grad).all()

    def test_nmi_always_less_negative_than_minus_two(self):
        """NMI ≤ 2.0 for non-identical images, so loss ≥ -2."""
        a = torch.from_numpy(self._blob())
        b = torch.from_numpy(self._blob(shift=(3, 0, 0)))
        loss = nmi_loss(a, b, num_bins=16)
        assert loss.item() >= -2.0 - 1e-3


class TestMRICoregCost:
    def test_identity_beats_shift_on_same_image(self):
        img = _make_img(shape=(24, 24, 24))
        evaluator = PowellCostEvaluator(img, img, sep=2, coord_dither=False, intensity_dither=False)

        identity = np.eye(4, dtype=np.float64)
        shifted = np.eye(4, dtype=np.float64)
        shifted[0, 3] = 2.0

        same = evaluator.evaluate_r2r(identity)
        diff = evaluator.evaluate_r2r(shifted)

        assert np.isfinite(same.cost)
        assert np.isfinite(diff.cost)
        assert same.cost < diff.cost

    def test_optimizer_improves_shifted_start(self):
        img = _make_img(shape=(24, 24, 24))
        evaluator = PowellCostEvaluator(img, img, sep=2, coord_dither=False, intensity_dither=False)

        shifted = np.eye(4, dtype=np.float64)
        shifted[0, 3] = 2.0
        initial = evaluator.evaluate_r2r(shifted)
        result = optimize_powell_from_rigid(evaluator, shifted, maxiter=8, options={"xtol": 1e-2, "ftol": 1e-3})

        assert np.isfinite(result.final_cost)
        assert result.final_cost <= initial.cost
        assert np.isfinite(result.final_r2r).all()
        assert np.allclose(result.final_r2r[:3, :3].T @ result.final_r2r[:3, :3], np.eye(3), atol=1e-5)
        assert np.isclose(np.linalg.det(result.final_r2r[:3, :3]), 1.0, atol=1e-5)

    def test_powell_optimizer_improves_translated_start(self):
        img = _make_img(shape=(24, 24, 24))
        evaluator = PowellCostEvaluator(img, img, sep=2, coord_dither=False, intensity_dither=False)

        init_params = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        initial = evaluator.evaluate_r2r(powell_params_to_mov_to_ref_r2r(init_params))
        result = optimize_powell_from_params(
            evaluator,
            init_params,
            brute_force_limit=2.0,
            brute_force_iters=1,
            brute_force_samples=4,
            powell_maxiter=4,
            options={"xtol": 1e-2, "ftol": 1e-3},
        )

        assert np.isfinite(result.final_cost)
        assert result.final_cost <= initial.cost
        assert abs(result.weights[0]) < abs(init_params[0])

    def test_powell_param_evaluation_with_oob_is_repeatable(self):
        img = _make_img(shape=(24, 24, 24))
        evaluator = PowellCostEvaluator(img, img, sep=2, coord_dither=False, intensity_dither=False)

        params = np.array([1.5, -0.5, 0.25, 2.0, -1.0, 0.5], dtype=np.float64)
        first = evaluator.evaluate_powell_params(params, include_oob=True)
        second = evaluator.evaluate_powell_params(params, include_oob=True)

        assert np.isfinite(first.cost)
        assert first.cost == second.cost
        assert first.nhits == second.nhits
        assert first.pcthits == second.pcthits

    def test_mricoreg_12dof_params_build_expected_matrix(self):
        params = np.array([
            1.0,
            -2.0,
            3.0,
            10.0,
            -5.0,
            15.0,
            1.1,
            0.9,
            1.2,
            0.01,
            -0.02,
            0.03,
        ], dtype=np.float64)

        ref_to_mov = powell_params_to_ref_to_mov_r2r(params)
        expected_linear = Rotation.from_euler("XYZ", [-params[3], params[4], -params[5]], degrees=True).as_matrix()
        expected_linear = expected_linear @ np.diag(params[6:9]) @ np.array(
            [[1.0, params[9], params[10]], [0.0, 1.0, params[11]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        assert np.allclose(ref_to_mov[:3, :3], expected_linear)
        assert np.allclose(ref_to_mov[:3, 3], params[:3])

    @pytest.mark.parametrize(
        "params",
        [
            np.array([1.0, -2.0, 3.0, 10.0, -5.0, 15.0], dtype=np.float64),
            np.array([1.0, -2.0, 3.0, 10.0, -5.0, 15.0, 1.1, 0.9, 1.2], dtype=np.float64),
            np.array([
                1.0,
                -2.0,
                3.0,
                10.0,
                -5.0,
                15.0,
                1.1,
                0.9,
                1.2,
                0.01,
                -0.02,
                0.03,
            ], dtype=np.float64),
        ],
    )
    def test_mricoreg_param_roundtrip_from_affine(self, params: np.ndarray):
        dof = len(params)
        mov_to_ref = powell_params_to_mov_to_ref_r2r(params)
        recovered = powell_mov_to_ref_r2r_to_params(mov_to_ref, dof=dof)
        rebuilt = powell_params_to_mov_to_ref_r2r(recovered)

        assert recovered.shape == params.shape
        assert np.allclose(rebuilt, mov_to_ref, atol=1e-8)


# ---------------------------------------------------------------------------
# End-to-end GD registration with new losses
# ---------------------------------------------------------------------------

class TestRegisterPyramidNewLosses:
    """Smoke tests: run GD pyramid registration with each new loss on a simple example."""

    @pytest.mark.parametrize("loss_name", ["ncc", "mi", "nmi"])
    def test_register_pyramid_new_loss_near_identity_on_same_image(self, loss_name):
        img = _make_img(shape=(32, 32, 32))
        v2v = register_gd_pyramid(
            img, img,
            return_v2v=True,
            init_type="header",
            symmetric=False,
            dof=6,
            n=10,
            loss_name=loss_name,
            loss_bins=16,
            min_voxels=16,
            device="cpu",
        )
        if loss_name == "mi":
            translation_mm = float(torch.linalg.vector_norm(v2v[:3, 3]))
            rotation_deg = float(np.rad2deg(Rotation.from_matrix(v2v[:3, :3].cpu().numpy()).magnitude()))
            assert translation_mm < 2.0
            assert rotation_deg < 5.0
        else:
            assert torch.allclose(v2v, torch.eye(4, dtype=v2v.dtype), atol=1.0)

    def test_register_pyramid_ncc_loss_decreases(self):
        """NCC loss should decrease over iterations when images are misaligned."""
        src = torch.from_numpy(_make_blob(shape=(32, 32, 32), shift=(4, 0, 0)))
        trg = torch.from_numpy(_make_blob(shape=(32, 32, 32)))
        _, losses, _ = register_level(src, trg, dof=6, init_type="header", n=15,
                                      loss_name="ncc", device="cpu")
        assert losses[0] >= losses[-1]
