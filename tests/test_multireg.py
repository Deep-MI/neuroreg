from __future__ import annotations

import importlib
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from neuroreg.image import cast_image_dtype
from neuroreg.multireg import choose_initial_target, compute_seed, multireg
from neuroreg.transforms import LTA


def _make_blob(shape: tuple[int, int, int] = (21, 21, 21), shift: tuple[float, float, float] = (0, 0, 0)) -> np.ndarray:
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        np.arange(shape[2], dtype=np.float32),
        indexing="ij",
    )
    center = (np.asarray(shape, dtype=np.float32) - 1.0) / 2.0 + np.asarray(shift, dtype=np.float32)
    dist2 = (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    return np.exp(-dist2 / 10.0).astype(np.float32)


def _make_img(
    shape: tuple[int, int, int] = (21, 21, 21),
    shift: tuple[float, float, float] = (0, 0, 0),
    *,
    affine: np.ndarray | None = None,
) -> nib.Nifti1Image:
    img_affine = np.eye(4, dtype=np.float32) if affine is None else affine.astype(np.float32)
    return nib.Nifti1Image(_make_blob(shape, shift), img_affine)


def _fake_tensor(matrix: np.ndarray):
    class _FakeTensor:
        def __init__(self, value: np.ndarray):
            self._value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._value

    return _FakeTensor(matrix)


def test_seed_and_initial_target_choice_are_deterministic():
    images = [_make_img(shift=(0, 0, 0)), _make_img(shift=(2, 0, 0)), _make_img(shift=(-2, 0, 0))]
    seed1 = compute_seed(images)
    seed2 = compute_seed(images)
    choice1 = choose_initial_target(images)
    choice2 = choose_initial_target(images)
    assert seed1 == seed2
    assert choice1 == choice2
    assert 0 <= choice1[0] < len(images)
    assert choice1[1] == seed1


def test_multireg_warns_on_voxel_size_mismatch(monkeypatch: pytest.MonkeyPatch):
    images = [
        _make_img(affine=np.diag([1.0, 1.0, 1.0, 1.0])),
        _make_img(affine=np.diag([1.2, 1.0, 1.0, 1.0])),
    ]
    register_module = importlib.import_module("neuroreg.multireg.register")

    def fake_robreg(src, trg, **kwargs):
        return _fake_tensor(np.eye(4, dtype=np.float64))

    monkeypatch.setattr(register_module, "robreg", fake_robreg)

    with pytest.warns(RuntimeWarning, match="Input voxel sizes differ"):
        result = multireg(images, init_target_index=0, nmax=1, template_iterations=0)

    assert len(result.transforms_r2r) == 2


def test_multireg_builds_template_and_expected_transforms(monkeypatch: pytest.MonkeyPatch):
    images = [_make_img(shift=(0, 0, 0)), _make_img(shift=(2, 0, 0)), _make_img(shift=(-2, 0, 0))]
    register_module = importlib.import_module("neuroreg.multireg.register")

    def fake_robreg(src, trg, **kwargs):
        src_data = np.asarray(src.dataobj)
        trg_data = np.asarray(trg.dataobj)
        src_peak = np.unravel_index(np.argmax(src_data), src_data.shape)
        trg_peak = np.unravel_index(np.argmax(trg_data), trg_data.shape)
        translation = np.subtract(trg_peak, src_peak, dtype=np.float64)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = translation
        return _fake_tensor(matrix)

    monkeypatch.setattr(register_module, "robreg", fake_robreg)
    result = multireg(images, init_target_index=0, nmax=1, template_iterations=0, return_mapped=True)
    translations = [matrix[:3, 3] for matrix in result.transforms_r2r]
    assert translations[0] == pytest.approx([0.0, 0.0, 0.0])
    assert translations[1] == pytest.approx([-2.0, 0.0, 0.0])
    assert translations[2] == pytest.approx([2.0, 0.0, 0.0])
    assert result.template_image.shape[:3] == images[0].shape[:3]
    template_peak = np.unravel_index(np.argmax(result.template_image.get_fdata()), result.template_image.shape)
    assert all(abs(int(coord) - 10) <= 1 for coord in template_peak)
    assert result.mapped_images is not None
    assert len(result.mapped_images) == 3
    assert result.template_iterations_run == 0
    assert result.iteration_distances == []


def test_multireg_iterative_refinement_reuses_previous_transforms(monkeypatch: pytest.MonkeyPatch):
    images = [_make_img(shift=(0, 0, 0)), _make_img(shift=(2, 0, 0)), _make_img(shift=(-2, 0, 0))]
    register_module = importlib.import_module("neuroreg.multireg.register")
    captured_init_transforms: list[np.ndarray | None] = []
    captured_isotropic_sizes: list[float | None] = []
    captured_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def fake_robreg(src, trg, **kwargs):
        init_transform = kwargs.get("init_transform")
        captured_isotropic_sizes.append(kwargs.get("isotropic_size"))
        captured_shapes.append((tuple(src.shape), tuple(trg.shape)))
        if init_transform is not None:
            matrix = np.asarray(init_transform, dtype=np.float64)
            captured_init_transforms.append(matrix.copy())
            return _fake_tensor(matrix)
        captured_init_transforms.append(None)
        src_data = np.asarray(src.dataobj)
        trg_data = np.asarray(trg.dataobj)
        src_peak = np.unravel_index(np.argmax(src_data), src_data.shape)
        trg_peak = np.unravel_index(np.argmax(trg_data), trg_data.shape)
        translation = np.subtract(trg_peak, src_peak, dtype=np.float64)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = translation
        return _fake_tensor(matrix)

    monkeypatch.setattr(register_module, "robreg", fake_robreg)
    result = multireg(images, init_target_index=0, nmax=1, template_iterations=3, template_eps=0.03)

    assert len(captured_init_transforms) == 5
    assert captured_init_transforms[:2] == [None, None]
    assert captured_isotropic_sizes == [None, None, None, None, None]
    # Template iterations pass the previously accumulated transform as init_transform.
    # image[0] is the initial target so it holds identity; image[1]/[2] carry pairwise offsets.
    t1 = np.eye(4, dtype=np.float64)
    t1[0, 3] = -2.0
    t2 = np.eye(4, dtype=np.float64)
    t2[0, 3] = 2.0
    assert captured_init_transforms[2] == pytest.approx(np.eye(4, dtype=np.float64))
    assert captured_init_transforms[3] == pytest.approx(t1)
    assert captured_init_transforms[4] == pytest.approx(t2)
    assert captured_shapes[2:] == [((21, 21, 21), (21, 21, 21))] * 3
    assert result.template_iterations_run == 1
    assert result.iteration_distances == pytest.approx([0.0])


def test_multireg_accepts_file_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    paths = []
    register_module = importlib.import_module("neuroreg.multireg.register")
    for index, shift in enumerate(((0, 0, 0), (1, 0, 0))):
        path = tmp_path / f"tp{index + 1}.nii.gz"
        nib.save(_make_img(shift=shift), path)
        paths.append(path)

    monkeypatch.setattr(
        register_module,
        "robreg",
        lambda *args, **kwargs: _fake_tensor(np.eye(4, dtype=np.float64)),
    )

    result = multireg(paths, init_target_index=0, nmax=1, template_iterations=0)
    assert result.template_image.shape[:3] == (21, 21, 21)
    assert len(result.ltas) == 2


def test_multireg_supports_median_aggregation(monkeypatch: pytest.MonkeyPatch):
    images = [
        nib.Nifti1Image(np.full((5, 5, 5), 1.0, dtype=np.float32), np.eye(4, dtype=np.float32)),
        nib.Nifti1Image(np.full((5, 5, 5), 10.0, dtype=np.float32), np.eye(4, dtype=np.float32)),
        nib.Nifti1Image(np.full((5, 5, 5), 100.0, dtype=np.float32), np.eye(4, dtype=np.float32)),
    ]
    register_module = importlib.import_module("neuroreg.multireg.register")
    monkeypatch.setattr(
        register_module,
        "robreg",
        lambda *args, **kwargs: _fake_tensor(np.eye(4, dtype=np.float64)),
    )

    result = multireg(images, init_target_index=0, nmax=1, template_iterations=0, average="median", fix_target=True)

    assert np.asarray(result.template_image.dataobj).mean() == pytest.approx(10.0)


def test_multireg_rebuilds_template_from_precomputed_ltas(monkeypatch: pytest.MonkeyPatch):
    images = [_make_img(shift=(0, 0, 0)), _make_img(shift=(2, 0, 0))]
    register_module = importlib.import_module("neuroreg.multireg.register")
    template_image = _make_img()
    matrices = [
        np.eye(4, dtype=np.float64),
        np.array(
            [
                [1.0, 0.0, 0.0, -2.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
    ]
    init_ltas = [
        LTA.from_matrix(matrix, f"tp{index + 1}.nii.gz", image, "template.nii.gz", template_image, lta_type=1)
        for index, (matrix, image) in enumerate(zip(matrices, images, strict=False))
    ]
    monkeypatch.setattr(
        register_module,
        "robreg",
        lambda *args, **kwargs: pytest.fail("robreg should not be called when reusing init_ltas with no iterations"),
    )

    result = multireg(images, init_target_index=0, init_ltas=init_ltas, template_iterations=0, return_mapped=True)

    assert len(result.transforms_r2r) == 2
    assert result.transforms_r2r[0] == pytest.approx(matrices[0])
    assert result.transforms_r2r[1] == pytest.approx(matrices[1])
    assert np.asarray(result.template_image.affine) == pytest.approx(template_image.affine)
    assert result.mapped_images is not None
    assert len(result.mapped_images) == 2


def test_multireg_rejects_fix_target_with_init_ltas(monkeypatch: pytest.MonkeyPatch):
    # Both determine the template space, so accepting the pair would silently
    # discard one. A caller who passes both should be told, not have to find it
    # in a log.
    images = [_make_img(shift=(0, 0, 0)), _make_img(shift=(2, 0, 0))]
    register_module = importlib.import_module("neuroreg.multireg.register")
    template_image = _make_img()
    identity = np.eye(4, dtype=np.float64)
    init_ltas = [
        LTA.from_matrix(identity, f"tp{index + 1}.nii.gz", image, "template.nii.gz", template_image, lta_type=1)
        for index, image in enumerate(images)
    ]
    monkeypatch.setattr(
        register_module,
        "robreg",
        lambda *args, **kwargs: pytest.fail("multireg must reject the pair before registering"),
    )

    with pytest.raises(ValueError, match="pass only one"):
        multireg(images, init_target_index=0, init_ltas=init_ltas, template_iterations=0, fix_target=True)


def test_multireg_accepts_fix_target_without_init_ltas(monkeypatch: pytest.MonkeyPatch):
    # The rejection must be specific to the combination, not to fix_target.
    images = [_make_img(shift=(0, 0, 0)), _make_img(shift=(2, 0, 0))]
    register_module = importlib.import_module("neuroreg.multireg.register")
    monkeypatch.setattr(
        register_module,
        "robreg",
        lambda *args, **kwargs: _fake_tensor(np.eye(4, dtype=np.float64)),
    )

    result = multireg(images, init_target_index=0, nmax=1, template_iterations=0, fix_target=True)

    assert result.template_image.shape == images[0].shape


def _geom_affine(voxel_size: float, corner: tuple[float, float, float]) -> np.ndarray:
    affine = np.diag([voxel_size, voxel_size, voxel_size, 1.0]).astype(np.float32)
    affine[:3, 3] = corner
    return affine


def test_multireg_template_geom_overrides_derived_geometry(monkeypatch: pytest.MonkeyPatch):
    images = [_make_img(shift=(0, 0, 0)), _make_img(shift=(2, 0, 0)), _make_img(shift=(-2, 0, 0))]
    register_module = importlib.import_module("neuroreg.multireg.register")
    monkeypatch.setattr(
        register_module,
        "robreg",
        lambda *args, **kwargs: _fake_tensor(np.eye(4, dtype=np.float64)),
    )
    geom_image = _make_img(shape=(12, 13, 14), affine=_geom_affine(1.0, (2.0, 3.0, 4.0)))

    derived = multireg(images, init_target_index=0, nmax=1, template_iterations=0)
    result = multireg(images, init_target_index=0, nmax=1, template_iterations=0, template_geometry=geom_image)

    assert result.template_image.shape[:3] == (12, 13, 14)
    assert np.asarray(result.template_image.affine) == pytest.approx(geom_image.affine)
    # The point of the flag: this is not the geometry multireg would have derived.
    assert derived.template_image.shape[:3] != (12, 13, 14)


def test_multireg_template_geom_accepts_a_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    images = [_make_img(shift=(0, 0, 0)), _make_img(shift=(2, 0, 0))]
    register_module = importlib.import_module("neuroreg.multireg.register")
    monkeypatch.setattr(
        register_module,
        "robreg",
        lambda *args, **kwargs: _fake_tensor(np.eye(4, dtype=np.float64)),
    )
    geom_path = tmp_path / "std.nii.gz"
    geom_affine = _geom_affine(1.0, (2.0, 3.0, 4.0))
    nib.save(_make_img(shape=(12, 13, 14), affine=geom_affine), geom_path)

    result = multireg(images, init_target_index=0, nmax=1, template_iterations=0, template_geometry=str(geom_path))

    assert result.template_image.shape[:3] == (12, 13, 14)
    assert np.asarray(result.template_image.affine) == pytest.approx(geom_affine)


def test_multireg_template_geom_skips_deriving_a_geometry_it_would_discard(monkeypatch: pytest.MonkeyPatch):
    # Orientations that no axis permutation relates make the derived geometry
    # path raise. Supplying the grid is exactly how a caller opts out of that
    # derivation, so it must not run at all.
    register_module = importlib.import_module("neuroreg.multireg.register")
    monkeypatch.setattr(
        register_module,
        "robreg",
        lambda *args, **kwargs: _fake_tensor(np.eye(4, dtype=np.float64)),
    )
    cos, sin = np.cos(np.pi / 4), np.sin(np.pi / 4)
    rotated = np.eye(4, dtype=np.float32)
    rotated[:3, :3] = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    images = [_make_img(), _make_img(shift=(2, 0, 0), affine=rotated)]
    geom_image = _make_img(shape=(12, 13, 14), affine=_geom_affine(1.0, (2.0, 3.0, 4.0)))

    with pytest.raises(ValueError, match="not compatible with FreeSurfer-style axis reordering"):
        multireg(images, init_target_index=0, nmax=1, template_iterations=0)

    with warnings.catch_warnings():
        # Nor may it claim the orientations will be averaged for the template.
        warnings.simplefilter("error", RuntimeWarning)
        result = multireg(images, init_target_index=0, nmax=1, template_iterations=0, template_geometry=geom_image)

    assert result.template_image.shape[:3] == (12, 13, 14)


def test_multireg_template_geom_rejects_cras_center(monkeypatch: pytest.MonkeyPatch):
    # use_cras_center only selects how a derived geometry is centered, and with
    # template_geom nothing is derived.
    images = [_make_img(shift=(0, 0, 0)), _make_img(shift=(2, 0, 0))]
    register_module = importlib.import_module("neuroreg.multireg.register")
    monkeypatch.setattr(
        register_module,
        "robreg",
        lambda *args, **kwargs: pytest.fail("multireg must reject the pair before registering"),
    )

    with pytest.raises(ValueError, match="pass only one"):
        multireg(
            images,
            init_target_index=0,
            template_iterations=0,
            template_geometry=_make_img(),
            use_cras_center=True,
        )


def test_multireg_rejects_cras_center_with_init_ltas(monkeypatch: pytest.MonkeyPatch):
    # Same reasoning as for template_geometry: init_ltas supply a complete grid,
    # placement included, so there is nothing left for use_cras_center to center.
    images = [_make_img(shift=(0, 0, 0)), _make_img(shift=(2, 0, 0))]
    register_module = importlib.import_module("neuroreg.multireg.register")
    template_image = _make_img()
    identity = np.eye(4, dtype=np.float64)
    init_ltas = [
        LTA.from_matrix(identity, f"tp{index + 1}.nii.gz", image, "template.nii.gz", template_image, lta_type=1)
        for index, image in enumerate(images)
    ]
    monkeypatch.setattr(
        register_module,
        "robreg",
        lambda *args, **kwargs: pytest.fail("multireg must reject the pair before registering"),
    )

    with pytest.raises(ValueError, match="init_ltas supplies the geometry"):
        multireg(images, init_target_index=0, init_ltas=init_ltas, template_iterations=0, use_cras_center=True)


def test_multireg_rejects_a_template_grid_that_misses_a_timepoint(monkeypatch: pytest.MonkeyPatch):
    # A grid that is supplied rather than derived can be placed anywhere, and a
    # wrong c_ras otherwise shows up only as an empty template.
    images = [_make_img(shift=(0, 0, 0)), _make_img(shift=(2, 0, 0))]
    register_module = importlib.import_module("neuroreg.multireg.register")
    monkeypatch.setattr(
        register_module,
        "robreg",
        lambda *args, **kwargs: _fake_tensor(np.eye(4, dtype=np.float64)),
    )
    far_away = _make_img(shape=(12, 13, 14), affine=_geom_affine(1.0, (500.0, 500.0, 500.0)))

    with pytest.raises(ValueError, match="maps entirely outside the template grid"):
        multireg(images, init_target_index=0, nmax=1, template_iterations=0, template_geometry=far_away)


def test_multireg_warns_when_the_template_grid_barely_overlaps_a_timepoint(monkeypatch: pytest.MonkeyPatch):
    # Only a corner in common: possible, so not an error, but almost always a
    # placement mistake and worth saying out loud.
    images = [_make_img(shift=(0, 0, 0)), _make_img(shift=(2, 0, 0))]
    register_module = importlib.import_module("neuroreg.multireg.register")
    monkeypatch.setattr(
        register_module,
        "robreg",
        lambda *args, **kwargs: _fake_tensor(np.eye(4, dtype=np.float64)),
    )
    overlapping_corner = _make_img(shape=(12, 13, 14), affine=_geom_affine(1.0, (15.0, 15.0, 15.0)))

    with pytest.warns(RuntimeWarning, match="overlaps the template grid by only"):
        multireg(images, init_target_index=0, nmax=1, template_iterations=0, template_geometry=overlapping_corner)


def test_multireg_accepts_a_template_grid_framing_part_of_the_inputs(monkeypatch: pytest.MonkeyPatch):
    # A small grid entirely inside the inputs frames one structure of a larger
    # image, which is a legitimate request and must stay silent.
    images = [_make_img(shift=(0, 0, 0)), _make_img(shift=(2, 0, 0))]
    register_module = importlib.import_module("neuroreg.multireg.register")
    monkeypatch.setattr(
        register_module,
        "robreg",
        lambda *args, **kwargs: _fake_tensor(np.eye(4, dtype=np.float64)),
    )
    inside = _make_img(shape=(6, 6, 6), affine=_geom_affine(1.0, (7.0, 7.0, 7.0)))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = multireg(images, init_target_index=0, nmax=1, template_iterations=0, template_geometry=inside)

    assert result.template_image.shape[:3] == (6, 6, 6)


def test_multireg_template_geom_rejects_fix_target(monkeypatch: pytest.MonkeyPatch):
    images = [_make_img(shift=(0, 0, 0)), _make_img(shift=(2, 0, 0))]
    register_module = importlib.import_module("neuroreg.multireg.register")
    monkeypatch.setattr(
        register_module,
        "robreg",
        lambda *args, **kwargs: pytest.fail("multireg must reject the pair before registering"),
    )

    with pytest.raises(ValueError, match="pass only one"):
        multireg(images, init_target_index=0, template_iterations=0, template_geometry=_make_img(), fix_target=True)


def test_multireg_template_geom_ignores_disagreeing_init_lta_geometry(monkeypatch: pytest.MonkeyPatch):
    # Deliberate: with the geometry supplied externally, the destination blocks of
    # the input transforms are never read, so they neither have to agree with each
    # other nor be present at all.
    images = [_make_img(shift=(0, 0, 0)), _make_img(shift=(2, 0, 0))]
    register_module = importlib.import_module("neuroreg.multireg.register")
    monkeypatch.setattr(
        register_module,
        "robreg",
        lambda *args, **kwargs: pytest.fail("robreg should not be called when reusing init_ltas with no iterations"),
    )
    identity = np.eye(4, dtype=np.float64)
    init_ltas = [
        LTA.from_matrix(identity, "tp1.nii.gz", images[0], "a.nii.gz", _make_img(shape=(21, 21, 21)), lta_type=1),
        LTA.from_matrix(
            identity,
            "tp2.nii.gz",
            images[1],
            "b.nii.gz",
            _make_img(shape=(15, 15, 15), affine=_geom_affine(2.0, (0.0, 0.0, 0.0))),
            lta_type=1,
        ),
    ]
    geom_image = _make_img(shape=(10, 10, 10), affine=_geom_affine(1.0, (2.0, 2.0, 2.0)))

    with pytest.raises(ValueError, match="identical destination geometry"):
        multireg(images, init_target_index=0, init_ltas=init_ltas, template_iterations=0)

    result = multireg(
        images,
        init_target_index=0,
        init_ltas=init_ltas,
        template_iterations=0,
        template_geometry=geom_image,
    )

    assert result.template_image.shape[:3] == (10, 10, 10)
    assert np.asarray(result.template_image.affine) == pytest.approx(geom_image.affine)


def test_multireg_template_geom_survives_template_iterations(monkeypatch: pytest.MonkeyPatch):
    images = [_make_img(shift=(0, 0, 0)), _make_img(shift=(2, 0, 0)), _make_img(shift=(-2, 0, 0))]
    register_module = importlib.import_module("neuroreg.multireg.register")
    monkeypatch.setattr(
        register_module,
        "robreg",
        lambda *args, **kwargs: _fake_tensor(np.eye(4, dtype=np.float64)),
    )
    geom_image = _make_img(shape=(12, 13, 14), affine=_geom_affine(1.0, (2.0, 3.0, 4.0)))

    result = multireg(images, init_target_index=0, nmax=1, template_iterations=3, template_geometry=geom_image)

    assert result.template_iterations_run > 0
    assert result.template_image.shape[:3] == (12, 13, 14)
    assert np.asarray(result.template_image.affine) == pytest.approx(geom_image.affine)


def test_multireg_template_geom_preserves_native_voxel_size(monkeypatch: pytest.MonkeyPatch):
    # Sub-millimeter inputs with a sub-millimeter template geometry must keep both
    # the voxel size and the image dimensions of that geometry, with no fallback
    # to a 1 mm grid. Scaled down from the 0.8 mm / 320 cubed production case.
    native_affine = _geom_affine(0.8, (-8.0, -8.0, -8.0))
    images = [
        _make_img(shape=(20, 20, 20), affine=native_affine),
        _make_img(shape=(20, 20, 20), shift=(2, 0, 0), affine=native_affine),
    ]
    register_module = importlib.import_module("neuroreg.multireg.register")
    monkeypatch.setattr(
        register_module,
        "robreg",
        lambda *args, **kwargs: _fake_tensor(np.eye(4, dtype=np.float64)),
    )
    geom_image = _make_img(shape=(24, 24, 24), affine=_geom_affine(0.8, (-9.6, -9.6, -9.6)))

    result = multireg(images, init_target_index=0, nmax=1, template_iterations=0, template_geometry=geom_image)

    assert result.template_image.shape[:3] == (24, 24, 24)
    voxel_sizes = np.linalg.norm(np.asarray(result.template_image.affine)[:3, :3], axis=0)
    assert voxel_sizes == pytest.approx([0.8, 0.8, 0.8])


def test_multireg_template_geom_maps_movables_onto_the_template_grid(monkeypatch: pytest.MonkeyPatch):
    images = [_make_img(shift=(0, 0, 0)), _make_img(shift=(2, 0, 0)), _make_img(shift=(-2, 0, 0))]
    register_module = importlib.import_module("neuroreg.multireg.register")
    monkeypatch.setattr(
        register_module,
        "robreg",
        lambda *args, **kwargs: _fake_tensor(np.eye(4, dtype=np.float64)),
    )
    geom_image = _make_img(shape=(12, 13, 14), affine=_geom_affine(1.0, (2.0, 3.0, 4.0)))

    result = multireg(
        images,
        init_target_index=0,
        nmax=1,
        template_iterations=0,
        return_mapped=True,
        template_geometry=geom_image,
    )

    assert result.mapped_images is not None
    for mapped in result.mapped_images:
        assert mapped.shape[:3] == (12, 13, 14)
        assert np.asarray(mapped.affine) == pytest.approx(geom_image.affine)


def test_cast_image_dtype_clips_cubic_overshoot_without_rescaling():
    data = np.array([-5.0, -0.4, 0.4, 128.0, 254.6, 300.0], dtype=np.float32).reshape(1, 1, 6)
    template_image = nib.Nifti1Image(data, np.eye(4, dtype=np.float32))

    cast_image = cast_image_dtype(template_image, np.dtype(np.uint8))

    assert cast_image.get_data_dtype() == np.dtype(np.uint8)
    cast_data = np.asarray(cast_image.dataobj)
    assert cast_data.ravel().tolist() == [0, 0, 0, 128, 255, 255]


def test_multireg_keep_dtype_casts_template_to_target_dtype_without_wraparound(monkeypatch: pytest.MonkeyPatch):
    shape = (21, 21, 21)
    disk = (_make_blob(shape) > 0.5).astype(np.uint8) * np.uint8(255)
    images = [
        nib.Nifti1Image(disk, np.eye(4, dtype=np.float32)),
        nib.Nifti1Image(disk, np.eye(4, dtype=np.float32)),
    ]
    register_module = importlib.import_module("neuroreg.multireg.register")
    monkeypatch.setattr(
        register_module,
        "robreg",
        lambda *args, **kwargs: _fake_tensor(np.eye(4, dtype=np.float64)),
    )

    result = multireg(images, init_target_index=0, nmax=1, template_iterations=0, mapped_keep_dtype=True)

    assert result.template_image.get_data_dtype() == np.dtype(np.uint8)
    template_data = np.asarray(result.template_image.dataobj)
    assert template_data.min() >= 0
    assert template_data.max() <= 255
