"""Shared transform-format inference and LTA conversion helpers."""

from __future__ import annotations

from pathlib import Path

from .afni import AFNIAffine
from .antsmat import ANTsMatTransform
from .fsl import FSLMat
from .itk import ITKTransform
from .lta import LTA
from .niftyreg import NiftyRegTransform
from .regdat import RegisterDat
from .xfm import XFM

TRANSFORM_FORMATS = ("lta", "xfm", "fsl", "regdat", "itk", "antsmat", "afni", "niftyreg")


def infer_transform_format(path: str, explicit: str | None = None) -> str:
    """Infer the linear-transform format from a filename or explicit override.

    Parameters
    ----------
    path : str
        Input or output transform path.
    explicit : str, optional
        Explicit format override. When provided, it is returned unchanged.

    Returns
    -------
    str
        One of :data:`TRANSFORM_FORMATS`.

    Raises
    ------
    ValueError
        If the path suffix is unsupported or ambiguous without an explicit
        override.
    """
    if explicit is not None:
        return explicit

    lower = path.lower()
    suffix = Path(lower).suffix
    if suffix == ".lta":
        return "lta"
    if suffix == ".xfm":
        return "xfm"
    if lower.endswith("genericaffine.mat"):
        return "antsmat"
    if suffix in {".mat", ".fslmat"}:
        return "fsl"
    if suffix in {".dat", ".reg"}:
        return "regdat"
    if suffix == ".tfm" or lower.endswith(".itk.txt") or lower.endswith(".ants.txt"):
        return "itk"
    if lower.endswith(".aff12.1d"):
        return "afni"
    if lower.endswith(".niftyreg.txt"):
        return "niftyreg"
    raise ValueError(
        f"Unsupported transform format for {path!r}; expected .lta, .xfm, .mat, .fslmat, "
        ".dat, .reg, .tfm, .aff12.1D, \\*GenericAffine.mat, or .niftyreg.txt. "
        "Use --transform-format/--in-format/--out-format for ambiguous text formats such as .txt, .1D, or .mat"
    )


def read_transform_as_lta(
        path: str,
        src_img: str | None = None,
        dst_img: str | None = None,
        fmt: str | None = None,
) -> LTA:
    """Read a supported transform file and convert it to an :class:`LTA`.

    Parameters
    ----------
    path : str
        Transform file to read.
    src_img : str, optional
        Moving/source image geometry path for geometry-dependent formats.
    dst_img : str, optional
        Reference/target image geometry path for geometry-dependent formats.
    fmt : str, optional
        Explicit input-format override.

    Returns
    -------
    LTA
        The loaded transform converted to LTA representation.

    Raises
    ------
    ValueError
        If the format is unsupported, ambiguous without an override, or needs
        source/destination geometry that was not supplied.
    """
    resolved_format = infer_transform_format(path, explicit=fmt)
    if resolved_format == "lta":
        return LTA.read(path)
    if resolved_format == "xfm":
        return XFM.read(path).to_lta(src_fname=src_img, src_img=src_img, dst_fname=dst_img, dst_img=dst_img)
    if resolved_format == "itk":
        return ITKTransform.read(path).to_lta(src_fname=src_img, src_img=src_img, dst_fname=dst_img, dst_img=dst_img)
    if resolved_format == "antsmat":
        return ANTsMatTransform.read(path).to_lta(
            src_fname=src_img,
            src_img=src_img,
            dst_fname=dst_img,
            dst_img=dst_img,
        )
    if resolved_format == "afni":
        return AFNIAffine.read(path).to_lta(src_fname=src_img, src_img=src_img, dst_fname=dst_img, dst_img=dst_img)
    if resolved_format == "niftyreg":
        return NiftyRegTransform.read(path).to_lta(
            src_fname=src_img,
            src_img=src_img,
            dst_fname=dst_img,
            dst_img=dst_img,
        )
    if src_img is None or dst_img is None:
        kind = "FSL" if resolved_format == "fsl" else "register.dat"
        raise ValueError(f"{kind} conversion requires both source and destination image geometry.")
    if resolved_format == "fsl":
        return FSLMat.read(path).to_lta(src_fname=src_img, src_img=src_img, dst_fname=dst_img, dst_img=dst_img)
    return RegisterDat.read(path).to_lta(src_fname=src_img, src_img=src_img, dst_fname=dst_img, dst_img=dst_img)


def write_lta_as_transform(
        lta: LTA,
        output: str,
        *,
        output_format: str | None = None,
        out_type: str | None = None,
        subject: str | None = None,
        fscale: float | None = None,
        float2int: str = "round",
) -> None:
    """Write an :class:`LTA` as another supported transform format.

    Parameters
    ----------
    lta : LTA
        Transform to write.
    output : str
        Destination filename.
    output_format : str, optional
        Explicit output-format override.
    out_type : {"ras2ras", "vox2vox"}, optional
        Requested LTA storage type when writing ``.lta`` output.
    subject : str, optional
        Subject metadata for ``.lta`` or ``register.dat`` output.
    fscale : float, optional
        Intensity scaling metadata for ``.lta`` or ``register.dat`` output.
    float2int : {"tkregister", "round", "floor"}, default="round"
        Float-to-int footer mode when writing ``register.dat``.

    Returns
    -------
    None
        The converted transform is written to disk.

    Raises
    ------
    ValueError
        If the output format is unsupported or ambiguous without an explicit
        override.
    """
    resolved_format = infer_transform_format(output, explicit=output_format)
    if resolved_format == "lta":
        if subject is not None:
            lta.subject = subject
        if fscale is not None:
            lta.fscale = fscale
        elif lta.fscale is None:
            lta.fscale = 0.1
        lta_type = None if out_type is None else {"vox2vox": 0, "ras2ras": 1}[out_type]
        lta.write(output, lta_type=lta_type)
        return
    if resolved_format == "xfm":
        XFM.from_lta(lta).write(output)
        return
    if resolved_format == "fsl":
        FSLMat.from_lta(lta).write(output)
        return
    if resolved_format == "itk":
        ITKTransform.from_lta(lta).write(output)
        return
    if resolved_format == "antsmat":
        ANTsMatTransform.from_lta(lta).write(output)
        return
    if resolved_format == "afni":
        AFNIAffine.from_lta(lta).write(output)
        return
    if resolved_format == "niftyreg":
        NiftyRegTransform.from_lta(lta).write(output)
        return
    RegisterDat.from_lta(lta, subject=subject, intensity=fscale, float2int=float2int).write(output)
