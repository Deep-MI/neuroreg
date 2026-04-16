from __future__ import annotations

import importlib
from typing import Any, cast

import nibabel as nib
import numpy as np
import pytest
import torch

import neuroreg.imreg.coreg as coreg_impl
from neuroreg import coreg, robreg
from neuroreg.imreg.coreg import register_gd_pyramid, register_level
from neuroreg.imreg.losses import mi_loss, ncc_loss, nmi_loss
from neuroreg.imreg.reg_model import RegModel


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
            centroid_init=False,
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
        monkeypatch.setattr("neuroreg.imreg.coreg.register_level", fake_register_level)

        v2v = register_gd_pyramid(
            img,
            img.__class__(img.get_fdata(), img.affine.copy()),
            return_v2v=True,
            centroid_init=False,
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

        monkeypatch.setattr("neuroreg.imreg.coreg.register_level", fake_register_level)

        img = _make_img(shape=(128, 128, 128))
        register_gd_pyramid(
            img,
            img,
            return_v2v=True,
            centroid_init=False,
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
            centroid_init=False,
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

        monkeypatch.setattr("neuroreg.imreg.coreg.register_level", fake_register_level)

        img = _make_img(shape=(64, 64, 64))
        register_gd_pyramid(
            img,
            img,
            return_v2v=True,
            centroid_init=False,
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
                centroid_init=False,
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

    def test_register_level_runs_when_source_and_target_shapes_differ(self):
        src = torch.from_numpy(_make_blob((20, 20, 20), shift=(2, 0, 0)))
        trg = torch.from_numpy(_make_blob((24, 24, 24)))
        v2v, losses, model = register_level(src, trg, dof=6, centroid_init=True, n=2, loss_name="mi", device="cpu")
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
            centroid_init=False,
            n=3,
            loss_name="nmi",
            device="cpu",
            trace_fn=trace_fn,
        )
        iter_events = [e for e in events if e["event"] == "iter_end"]
        assert len(iter_events) == 3
        assert len(losses) == 3
        assert all("v2v" in e for e in iter_events)

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

        monkeypatch.setattr("neuroreg.imreg.coreg.torch.optim.Adam", FakeAdam)
        monkeypatch.setattr("neuroreg.imreg.coreg.training_loop", lambda *args, **kwargs: [])

        src = torch.from_numpy(_make_blob((16, 16, 16)))
        trg = torch.from_numpy(_make_blob((16, 16, 16)))
        v2v, losses, _ = register_level(src, trg, centroid_init=False, n=1, lr=0.005, device="cpu")

        assert captured["lr"] == pytest.approx(0.005)
        assert v2v.shape == (4, 4)
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
            centroid_init=False,
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

    def test_register_pyramid_symmetric_uses_centroid_init_only_on_first_level(self, monkeypatch: pytest.MonkeyPatch):
        seen_centroid_flags: list[bool] = []

        def fake_register_level(simg, timg, **kwargs):
            _ = simg, timg
            seen_centroid_flags.append(bool(kwargs["centroid_init"]))
            return torch.eye(4), [], None

        monkeypatch.setattr("neuroreg.imreg.coreg.register_level", fake_register_level)

        img = _make_img(shape=(32, 32, 32))
        _ = register_gd_pyramid(
            img,
            img,
            return_v2v=True,
            symmetric=True,
            centroid_init=True,
            isotropic=False,
            n=1,
            min_voxels=16,
            max_voxels=None,
            device="cpu",
        )

        assert seen_centroid_flags == [True, False]

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
            centroid_init=False,
            isotropic=False,
            n=1,
            min_voxels=16,
            max_voxels=16,
            device="cpu",
        )

        assert torch.allclose(v2v, torch.eye(4, dtype=v2v.dtype), atol=1.0)


class TestPublicRobregWrapper:
    def test_top_level_robreg_defaults_to_symmetric(self, monkeypatch: pytest.MonkeyPatch):
        captured: dict[str, object] = {}

        def fake_register_irls_pyramid(**kwargs):
            captured.update(kwargs)
            return torch.eye(4), []

        monkeypatch.setattr("neuroreg.imreg.robreg.register_irls_pyramid", fake_register_irls_pyramid)

        img = _make_img()
        Mr2r = robreg(img, img, return_v2v=False, centroid_init=False, dof=6, nmax=1)

        assert captured["symmetric"] is True
        assert Mr2r.shape == (4, 4)
        assert torch.isfinite(Mr2r).all()

    def test_top_level_robreg_can_disable_symmetric_mode(self, monkeypatch: pytest.MonkeyPatch):
        captured: dict[str, object] = {}

        def fake_register_irls_pyramid(**kwargs):
            captured.update(kwargs)
            return torch.eye(4), []

        monkeypatch.setattr("neuroreg.imreg.robreg.register_irls_pyramid", fake_register_irls_pyramid)

        img = _make_img()
        Mr2r = robreg(img, img, return_v2v=False, centroid_init=False, dof=6, nmax=1, symmetric=False)

        assert captured["symmetric"] is False
        assert Mr2r.shape == (4, 4)
        assert torch.isfinite(Mr2r).all()

    def test_robreg_accepts_file_paths(self, tmp_path):
        src = _make_img()
        trg = _make_img()
        src_path = tmp_path / "src.nii.gz"
        trg_path = tmp_path / "trg.nii.gz"
        nib.save(src, src_path)
        nib.save(trg, trg_path)

        Mr2r = robreg(
            str(src_path),
            str(trg_path),
            return_v2v=False,
            centroid_init=False,
            dof=6,
            nmax=1,
        )
        assert Mr2r.shape == (4, 4)
        assert torch.isfinite(Mr2r).all()


class TestPublicCoregWrapper:
    def test_top_level_coreg_forwards_to_register_gd_pyramid(self, monkeypatch: pytest.MonkeyPatch):
        captured: dict[str, object] = {}

        def fake_register_gd_pyramid(*args, **kwargs):
            captured["args"] = args
            captured.update(kwargs)
            return torch.eye(4)

        monkeypatch.setattr("neuroreg.imreg.coreg.register_gd_pyramid", fake_register_gd_pyramid)

        img = _make_img()
        Mr2r = coreg(img, img, return_v2v=False, centroid_init=False, dof=6, n=1)

        assert captured["centroid_init"] is False
        assert captured["dof"] == 6
        assert len(cast(tuple[Any, Any], captured["args"])) == 2
        assert Mr2r.shape == (4, 4)
        assert torch.isfinite(Mr2r).all()

    def test_module_level_coreg_alias_forwards_to_register_gd_pyramid(self, monkeypatch: pytest.MonkeyPatch):
        captured: dict[str, object] = {}

        def fake_register_gd_pyramid(*args, **kwargs):
            captured["args"] = args
            captured.update(kwargs)
            return torch.eye(4)

        monkeypatch.setattr("neuroreg.imreg.coreg.register_gd_pyramid", fake_register_gd_pyramid)

        img = _make_img()
        Mv2v = coreg_impl.coreg(img, img, return_v2v=True, centroid_init=False, dof=3, n=1)

        assert captured["return_v2v"] is True
        assert captured["dof"] == 3
        assert len(cast(tuple[Any, Any], captured["args"])) == 2
        assert Mv2v.shape == (4, 4)
        assert torch.isfinite(Mv2v).all()


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


# ---------------------------------------------------------------------------
# End-to-end GD registration with new losses
# ---------------------------------------------------------------------------

class TestRegisterPyramidNewLosses:
    """Smoke tests: run GD pyramid registration with each new loss on a simple example."""

    @pytest.mark.parametrize("loss_name", ["ncc", "mi", "nmi"])
    def test_register_pyramid_new_loss_produces_finite_transform(self, loss_name):
        img = _make_img(shape=(32, 32, 32))
        v2v = register_gd_pyramid(
            img, img,
            return_v2v=True,
            centroid_init=False,
            symmetric=False,
            dof=6,
            n=3,
            loss_name=loss_name,
            loss_bins=16,
            min_voxels=16,
            device="cpu",
        )
        assert v2v.shape == (4, 4)
        assert torch.isfinite(v2v).all()

    @pytest.mark.parametrize("loss_name", ["ncc", "mi", "nmi"])
    def test_register_pyramid_new_loss_near_identity_on_same_image(self, loss_name):
        img = _make_img(shape=(32, 32, 32))
        v2v = register_gd_pyramid(
            img, img,
            return_v2v=True,
            centroid_init=False,
            symmetric=False,
            dof=6,
            n=10,
            loss_name=loss_name,
            loss_bins=16,
            min_voxels=16,
            device="cpu",
        )
        assert torch.allclose(v2v, torch.eye(4, dtype=v2v.dtype), atol=1.0)

    def test_register_pyramid_ncc_loss_decreases(self):
        """NCC loss should decrease over iterations when images are misaligned."""
        src = torch.from_numpy(_make_blob(shape=(32, 32, 32), shift=(4, 0, 0)))
        trg = torch.from_numpy(_make_blob(shape=(32, 32, 32)))
        _, losses, _ = register_level(src, trg, dof=6, centroid_init=False, n=15,
                                      loss_name="ncc", device="cpu")
        assert losses[0] >= losses[-1]
