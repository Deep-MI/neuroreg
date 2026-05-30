from __future__ import annotations

import importlib
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

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


def test_multireg_rejects_voxel_size_mismatch():
    images = [
        _make_img(affine=np.diag([1.0, 1.0, 1.0, 1.0])),
        _make_img(affine=np.diag([1.2, 1.0, 1.0, 1.0])),
    ]
    with pytest.raises(ValueError, match="identical voxel sizes"):
        multireg(images, init_target_index=0, nmax=1)


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
    captured_initial_r2r: list[np.ndarray | None] = []

    def fake_robreg(src, trg, **kwargs):
        initial_r2r = kwargs.get("initial_r2r")
        if initial_r2r is not None:
            matrix = np.asarray(initial_r2r, dtype=np.float64)
            captured_initial_r2r.append(matrix.copy())
            return _fake_tensor(matrix)
        captured_initial_r2r.append(None)
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

    assert len(captured_initial_r2r) == 5
    assert captured_initial_r2r[:2] == [None, None]
    assert captured_initial_r2r[2] == pytest.approx(np.eye(4, dtype=np.float64))
    assert captured_initial_r2r[3] == pytest.approx(
        np.array(
            [
                [1.0, 0.0, 0.0, -2.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    )
    assert captured_initial_r2r[4] == pytest.approx(
        np.array(
            [
                [1.0, 0.0, 0.0, 2.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    )
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
