import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from nibabel.arrayproxy import ArrayProxy

from neuroreg.cli.segreg import main as segreg_main
from neuroreg.image import header_map_image
from neuroreg.segreg.atlas import load_fsaverage_centroids, load_fsaverage_data
from neuroreg.segreg.centroids import build_flipped_centroid_targets
from neuroreg.segreg.points import find_affine, find_rigid, register_points_robust
from neuroreg.segreg.register import segreg
from neuroreg.transforms import LTA


def _label_volume() -> np.ndarray:
    data = np.zeros((8, 8, 8), dtype=np.int16)
    data[1, 1, 1] = 1
    data[1, 5, 2] = 2
    data[5, 2, 6] = 3
    data[6, 6, 4] = 4
    return data


def _write_seg(path: Path, *, affine: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(_label_volume(), affine=affine), path)


def _write_float_image(path: Path, *, affine: np.ndarray) -> None:
    data = np.arange(8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
    nib.save(nib.Nifti1Image(data, affine=affine), path)


def _cortex_lr_volume() -> np.ndarray:
    data = np.zeros((8, 8, 8), dtype=np.int16)
    data[1, 1, 1] = 1002
    data[6, 1, 1] = 2002
    data[2, 5, 2] = 1003
    data[5, 5, 2] = 2003
    return data


def _write_cortex_lr_seg(path: Path, *, affine: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(_cortex_lr_volume(), affine=affine), path)


def _apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (transform[:3, :3] @ points.T).T + transform[:3, 3]


def test_find_rigid_recovers_known_transform():
    mov = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ]
    )
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([4.0, -2.0, 1.5])
    dst = (rotation @ mov.T).T + translation

    rigid = find_rigid(mov, dst)
    expected = np.eye(4)
    expected[:3, :3] = rotation
    expected[:3, 3] = translation
    assert rigid == pytest.approx(expected, abs=1e-8)


def test_find_affine_recovers_known_transform():
    mov = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    affine = np.array(
        [
            [1.2, 0.1, 0.0, 3.0],
            [0.0, 0.8, -0.2, -1.0],
            [0.0, 0.3, 1.5, 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    dst = (affine[:3, :3] @ mov.T).T + affine[:3, 3]

    recovered = find_affine(mov, dst)
    assert recovered == pytest.approx(affine, abs=1e-8)


def test_register_points_robust_recovers_known_transform():
    mov = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.5],
        ]
    )
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([4.0, -2.0, 1.5])
    dst = (rotation @ mov.T).T + translation

    robust, info = register_points_robust(mov, dst)
    expected = np.eye(4)
    expected[:3, :3] = rotation
    expected[:3, 3] = translation

    assert robust == pytest.approx(expected, abs=1e-8)
    assert info.estimator == "tukey"
    assert info.iterations >= 1
    assert info.bound_scale == pytest.approx(0.5)
    assert info.weights.shape == (mov.shape[0],)
    assert info.residuals.shape == (mov.shape[0],)
    assert info.weights == pytest.approx(np.ones(mov.shape[0]), abs=1e-8)


def test_register_points_robust_downweights_outlier():
    mov = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([4.0, -2.0, 1.5])
    dst_true = (rotation @ mov.T).T + translation
    dst_outlier = dst_true.copy()
    dst_outlier[-1] += np.array([12.0, -10.0, 8.0])

    closed = find_rigid(mov, dst_outlier)
    robust, info = register_points_robust(mov, dst_outlier)

    closed_inlier_error = np.mean(np.linalg.norm(_apply_transform(mov[:-1], closed) - dst_true[:-1], axis=1))
    robust_inlier_error = np.mean(np.linalg.norm(_apply_transform(mov[:-1], robust) - dst_true[:-1], axis=1))

    assert robust_inlier_error < closed_inlier_error
    assert robust_inlier_error < 0.1
    assert info.weights[-1] < 1e-3
    assert info.weights[:-1].mean() > info.weights[-1]


def test_build_flipped_centroid_targets_mirrors_and_swaps_pairs():
    voxel_centroids = {
        1002: np.array([1.0, 2.0, 3.0]),
        2002: np.array([6.0, 2.0, 3.0]),
        1003: np.array([1.5, 4.0, 2.0]),
        2003: np.array([5.5, 4.0, 2.0]),
    }
    source, target, labels = build_flipped_centroid_targets(
        voxel_centroids,
        ((1002, 2002), (1003, 2003)),
        mid_slice=3.5,
        min_common_labels=4,
    )

    assert labels == [1002, 2002, 1003, 2003]
    assert source == pytest.approx(
        np.array(
            [
                [1.0, 2.0, 3.0],
                [6.0, 2.0, 3.0],
                [1.5, 4.0, 2.0],
                [5.5, 4.0, 2.0],
            ]
        )
    )
    assert target == pytest.approx(
        np.array(
            [
                [1.0, 2.0, 3.0],
                [6.0, 2.0, 3.0],
                [1.5, 4.0, 2.0],
                [5.5, 4.0, 2.0],
            ]
        )
    )


def test_load_fsaverage_resources():
    centroids = load_fsaverage_centroids()
    affine, header = load_fsaverage_data()

    assert 2 in centroids
    assert centroids[2].shape == (3,)
    assert affine.shape == (4, 4)
    assert header["dims"] == [256, 256, 256]


def test_segreg_registers_segmentation_images_in_ras(tmp_path: Path):
    mov_path = tmp_path / "mov_seg.nii.gz"
    ref_path = tmp_path / "ref_seg.nii.gz"

    mov_affine = np.eye(4)
    ref_affine = np.array(
        [
            [1.0, 0.0, 0.0, 5.0],
            [0.0, 1.0, 0.0, -3.0],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    _write_seg(mov_path, affine=mov_affine)
    _write_seg(ref_path, affine=ref_affine)

    result = segreg(mov_path, ref_path, dof=6)
    expected = ref_affine @ np.linalg.inv(mov_affine)

    assert result.r2r == pytest.approx(expected, abs=1e-6)
    assert result.labels == [1, 2, 3, 4]
    assert result.robust_info is None


def test_segreg_robust_mode_returns_diagnostics(tmp_path: Path):
    mov_path = tmp_path / "mov_seg.nii.gz"
    ref_path = tmp_path / "ref_seg.nii.gz"

    mov_affine = np.eye(4)
    ref_affine = np.array(
        [
            [1.0, 0.0, 0.0, 5.0],
            [0.0, 1.0, 0.0, -3.0],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    _write_seg(mov_path, affine=mov_affine)
    _write_seg(ref_path, affine=ref_affine)

    result = segreg(mov_path, ref_path, dof=6, robust=True)
    expected = ref_affine @ np.linalg.inv(mov_affine)

    assert result.r2r == pytest.approx(expected, abs=1e-6)
    assert result.robust_info is not None
    assert result.robust_info.weights.shape == (4,)
    assert result.robust_info.residuals.shape == (4,)


def test_segreg_robust_mode_improves_centroid_outlier_fit(tmp_path: Path):
    mov_path = tmp_path / "mov_seg.nii.gz"
    ref_centroids_path = tmp_path / "ref_centroids.json"
    _write_seg(mov_path, affine=np.eye(4))

    mov_points = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 5.0, 2.0],
            [5.0, 2.0, 6.0],
            [6.0, 6.0, 4.0],
        ]
    )
    labels = [1, 2, 3, 4]
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([4.0, -2.0, 1.5])
    expected = np.eye(4)
    expected[:3, :3] = rotation
    expected[:3, 3] = translation

    target_points = (rotation @ mov_points.T).T + translation
    target_points[-1] += np.array([12.0, -10.0, 8.0])
    ref_centroids = {str(label): point.tolist() for label, point in zip(labels, target_points, strict=True)}
    ref_centroids_path.write_text(json.dumps(ref_centroids) + "\n")

    closed = segreg(mov_path, ref_centroids=ref_centroids_path)
    robust = segreg(mov_path, ref_centroids=ref_centroids_path, robust=True)

    closed_inlier_error = np.mean(np.linalg.norm(_apply_transform(mov_points[:-1], closed.r2r) - _apply_transform(
        mov_points[:-1], expected), axis=1))
    robust_inlier_error = np.mean(np.linalg.norm(_apply_transform(mov_points[:-1], robust.r2r) - _apply_transform(
        mov_points[:-1], expected), axis=1))

    assert robust_inlier_error < closed_inlier_error
    assert robust_inlier_error < 0.1
    assert robust.robust_info is not None
    assert robust.robust_info.weights[-1] < 1e-3


def test_segreg_flipped_robust_mode_returns_identity_for_symmetric_labels(tmp_path: Path):
    mov_path = tmp_path / "mov_cortex_seg.nii.gz"
    _write_cortex_lr_seg(mov_path, affine=np.eye(4))

    result = segreg(mov_path, flipped=True, robust=True)

    assert result.r2r == pytest.approx(np.eye(4), abs=1e-6)
    assert result.labels == [1002, 2002, 1003, 2003]
    assert result.robust_info is not None


def test_segreg_rejects_robust_affine_registration(tmp_path: Path):
    mov_path = tmp_path / "mov_seg.nii.gz"
    ref_path = tmp_path / "ref_seg.nii.gz"
    _write_seg(mov_path, affine=np.eye(4))
    _write_seg(ref_path, affine=np.eye(4))

    with pytest.raises(ValueError, match="rigid registration only"):
        segreg(mov_path, ref_path, dof=12, robust=True)


def test_cli_writes_lta_and_mapmov(tmp_path: Path):
    mov_seg = tmp_path / "mov_seg.nii.gz"
    ref_seg = tmp_path / "ref_seg.nii.gz"
    mov_img = tmp_path / "mov_img.nii.gz"
    out_lta = tmp_path / "out.lta"
    out_map = tmp_path / "mapped.nii.gz"

    mov_affine = np.eye(4)
    ref_affine = np.array(
        [
            [1.0, 0.0, 0.0, 5.0],
            [0.0, 1.0, 0.0, -3.0],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    _write_seg(mov_seg, affine=mov_affine)
    _write_seg(ref_seg, affine=ref_affine)
    _write_float_image(mov_img, affine=mov_affine)

    segreg_main(
        [
            "--mov",
            str(mov_seg),
            "--ref",
            str(ref_seg),
            "--movimg",
            str(mov_img),
            "--lta",
            str(out_lta),
            "--mapmov",
            str(out_map),
        ]
    )

    assert out_lta.exists()
    assert out_map.exists()
    mapped_img = nib.load(str(out_map))
    assert mapped_img.shape[:3] == nib.load(str(ref_seg)).shape[:3]
    assert mapped_img.affine == pytest.approx(ref_affine)


def test_cli_writes_lta_with_robust_flag(tmp_path: Path):
    mov_seg = tmp_path / "mov_seg.nii.gz"
    ref_seg = tmp_path / "ref_seg.nii.gz"
    out_lta = tmp_path / "robust_out.lta"

    _write_seg(mov_seg, affine=np.eye(4))
    _write_seg(ref_seg, affine=np.eye(4))

    segreg_main(
        [
            "--mov",
            str(mov_seg),
            "--ref",
            str(ref_seg),
            "--robust",
            "--lta",
            str(out_lta),
        ]
    )

    assert out_lta.exists()


def test_cli_rejects_robust_affine_combination(tmp_path: Path):
    mov_seg = tmp_path / "mov_seg.nii.gz"
    ref_seg = tmp_path / "ref_seg.nii.gz"
    out_lta = tmp_path / "out.lta"

    _write_seg(mov_seg, affine=np.eye(4))
    _write_seg(ref_seg, affine=np.eye(4))

    with pytest.raises(SystemExit):
        segreg_main(
            [
                "--mov",
                str(mov_seg),
                "--ref",
                str(ref_seg),
                "--robust",
                "--affine",
                "--lta",
                str(out_lta),
            ]
        )


def test_cli_writes_header_only_mapmovhdr(tmp_path: Path):
    mov_seg = tmp_path / "mov_seg.nii.gz"
    ref_seg = tmp_path / "ref_seg.nii.gz"
    mov_img = tmp_path / "mov_img.nii.gz"
    out_maphdr = tmp_path / "mapped_hdr.nii.gz"

    mov_affine = np.eye(4)
    ref_affine = np.array(
        [
            [1.0, 0.0, 0.0, 5.0],
            [0.0, 1.0, 0.0, -3.0],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    _write_seg(mov_seg, affine=mov_affine)
    _write_seg(ref_seg, affine=ref_affine)
    _write_float_image(mov_img, affine=mov_affine)

    segreg_main(
        [
            "--mov",
            str(mov_seg),
            "--ref",
            str(ref_seg),
            "--movimg",
            str(mov_img),
            "--mapmovhdr",
            str(out_maphdr),
        ]
    )

    assert out_maphdr.exists()
    mapped_img = nib.load(str(out_maphdr))
    assert mapped_img.affine == pytest.approx(ref_affine)


def test_header_map_image_preserves_lazy_proxy(tmp_path: Path):
    image_path = tmp_path / "proxy_img.nii.gz"
    affine = np.eye(4)
    _write_float_image(image_path, affine=affine)

    loaded = nib.load(str(image_path))
    mapped = header_map_image(loaded, np.array(
        [[1.0, 0.0, 0.0, 5.0], [0.0, 1.0, 0.0, -3.0], [0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 0.0, 1.0]]))

    assert isinstance(loaded.dataobj, ArrayProxy)
    assert isinstance(mapped.dataobj, ArrayProxy)
    assert mapped.affine == pytest.approx(
        np.array([[1.0, 0.0, 0.0, 5.0], [0.0, 1.0, 0.0, -3.0], [0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 0.0, 1.0]]))


def test_cli_atlas_mode_can_export_target_centroids(tmp_path: Path):
    mov_seg = tmp_path / "mov_seg.nii.gz"
    out_json = tmp_path / "atlas_centroids.json"
    _write_seg(mov_seg, affine=np.eye(4))

    segreg_main([
        "--mov",
        str(mov_seg),
        "--atlas",
        "fsaverage",
        "--write-ref-centroids",
        str(out_json),
    ])

    assert out_json.exists()
    exported = load_fsaverage_centroids()
    with out_json.open() as f:
        saved = {int(k): np.asarray(v) for k, v in __import__("json").load(f).items()}
    assert set(saved) == set(exported)


def test_cli_ref_centroids_without_ref_geom_writes_invalid_dst_lta(tmp_path: Path):
    mov_seg = tmp_path / "mov_seg.nii.gz"
    ref_centroids = tmp_path / "ref_centroids.json"
    out_lta = tmp_path / "out_invalid_dst.lta"

    _write_seg(mov_seg, affine=np.eye(4))
    ref_centroids.write_text(
        '{"1": [1.0, 1.0, 1.0], "2": [1.0, 5.0, 2.0], "3": [5.0, 2.0, 6.0], "4": [6.0, 6.0, 4.0]}\n')

    segreg_main([
        "--mov",
        str(mov_seg),
        "--ref-centroids",
        str(ref_centroids),
        "--lta",
        str(out_lta),
    ])

    lta = LTA.read(out_lta)
    assert lta.dst["valid"] == 0
    assert lta.dst["filename"] == str(ref_centroids)
    with pytest.raises(ValueError, match="valid = 0"):
        lta.v2v()
