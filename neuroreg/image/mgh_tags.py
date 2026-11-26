"""Read and write the tag section FreeSurfer appends to MGH/MGZ volumes.

An MGH file is a 284-byte header, the voxel data, five big-endian float32 scan
parameters (TR, flip angle, TE, TI, FoV), and then an optional sequence of
tags. nibabel reads and writes everything up to the scan parameters and
ignores the rest, so any volume it writes carries no tags. FreeSurfer stores
two pieces of metadata there that tools rely on: the path of the talairach
transform (``TAG_MGH_XFORM``), read by ``recon-all`` stages and ``mri_info``,
and an embedded colour lookup table (``TAG_OLD_COLORTABLE``), which freeview
uses to name and colour a segmentation without an external LUT.

Each tag is a big-endian int32 id followed, for most ids, by an int64 payload
length and the payload. A few ids carry no length; the only one that occurs in
an ordinary volume is the colour table, which is therefore decoded to find its
end. Rewriting is done at the byte level: the header, voxels and scan
parameters are copied verbatim, every tag not being replaced is kept in its
original order, and only the requested tags change. The format follows
FreeSurfer's ``utils/fstagsio.cpp``, ``utils/colortab.cpp`` and
``MRITAGread``/``MRITAGwrite`` in ``utils/mriio.cpp``.
"""

from __future__ import annotations

import gzip
import io
import logging
import os
import platform
import shlex
import struct
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from nibabel.filebasedimages import ImageFileError
from nibabel.freesurfer.mghformat import MGHHeader

logger = logging.getLogger(__name__)

#: Embedded colour lookup table. Written without a length field.
TAG_OLD_COLORTABLE = 1
#: Command line of a program that wrote the volume.
TAG_CMDLINE = 3
#: Legacy transform-path tag, with an int32 length that counts the NUL.
TAG_OLD_MGH_XFORM = 30
#: Path of the talairach transform associated with the volume.
TAG_MGH_XFORM = 31
#: Siemens AutoAlign matrix, copied verbatim from the DICOM protocol.
TAG_AUTO_ALIGN = 33
#: Phase-encode direction.
TAG_PEDIR = 41
#: Per-frame metadata.
TAG_MRI_FRAME = 42
#: Field strength of the acquisition.
TAG_FIELDSTRENGTH = 43
#: RAS-to-voxel matrix of the original acquisition grid.
TAG_ORIG_RAS2VOX = 44

# Tags describing the voxel values or how they were acquired. They stay valid
# wherever the values go, whatever coordinate frame the output lives in.
_VALUE_TAGS = frozenset({TAG_OLD_COLORTABLE, TAG_PEDIR, TAG_MRI_FRAME, TAG_FIELDSTRENGTH})
# Readable names for the tags dropped when an output leaves the source frame.
_FRAME_TAG_NAMES = {
    TAG_MGH_XFORM: "talairach xform path",
    TAG_OLD_MGH_XFORM: "legacy xform path",
    TAG_AUTO_ALIGN: "AutoAlign matrix",
}
# FreeSurfer writes a matrix tag as text in a fixed, NUL-padded buffer.
_MATRIX_STRLEN = 4 * 4 * 100

#: NIfTI header-extension code registered for FreeSurfer's tags.
NIFTI_ECODE_FREESURFER = 14
# In the NIfTI extension every tag carries an int64 length, and the data ends
# with this tag so that trailing zero bytes are never mistaken for padding.
_TAG_END_NIIHDREXTENSION = -1
# Big-endian marker, intent 0 (an ordinary volume) as a big-endian uint16, and
# MGH format version 1: the four bytes FreeSurfer writes before the tags.
_NIFTI_EXTENSION_HEADER = b">\x00\x00\x01"
# Tags whose matrix is text in MGH but 16 big-endian float32 in the extension.
_MATRIX_TAGS = frozenset({TAG_AUTO_ALIGN, TAG_ORIG_RAS2VOX})
# NIfTI-only tags: DOF, FreeSurfer's copy of the geometry, and the scan
# parameters. The image header already holds what they describe.
_NIFTI_ONLY_TAGS = frozenset({7, 8, 9, 45})
# MGH tags whose NIfTI layout differs and is not written here: FreeSurfer folds
# the first two into the scan-parameter tag, and stores frames in fewer fields.
_MGH_ONLY_TAGS = {TAG_PEDIR: "phase-encode direction", TAG_FIELDSTRENGTH: "field strength", TAG_MRI_FRAME: "frames"}
_NIFTI_SUFFIXES = (".nii", ".nii.gz")
# A RAS-to-RAS map this close to identity leaves the coordinate frame alone.
_IDENTITY_ATOL = 1e-6

# Ids that FreeSurfer also writes without a length field. They only occur in
# warp-field volumes, whose payloads this module does not decode, so a file
# holding one cannot be stepped through safely.
_UNSIZED_WARP_TAGS = {10: "TAG_GCAMORPH_GEOM", 11: "TAG_GCAMORPH_TYPE", 12: "TAG_GCAMORPH_LABELS"}

_SCAN_PARAMETERS_SIZE = 20
_GZIP_MAGIC = b"\x1f\x8b"
_MGH_SUFFIXES = (".mgz", ".mgh")
# FreeSurfer writes .mgz through zlib's default level, so match it.
_GZIP_LEVEL = 6
_CTAB_VERSION_2 = -2


@dataclass(frozen=True)
class ColorTableEntry:
    """One label of a colour lookup table.

    Parameters
    ----------
    name : str
        Structure name.
    rgb : tuple of int
        Red, green and blue components, each in ``[0, 255]``.
    transparency : int, default=0
        The fourth colour column of a FreeSurfer LUT, conventionally headed
        ``A`` but read by FreeSurfer as transparency, so ``0`` is opaque. It is
        stored unchanged.
    """

    name: str
    rgb: tuple[int, int, int]
    transparency: int = 0


@dataclass(frozen=True)
class ColorTable:
    """A colour lookup table mapping label values to names and colours.

    Parameters
    ----------
    entries : dict of int to ColorTableEntry
        Entries keyed by label value. Values need not be contiguous.
    fname : str, default=""
        Name recorded with the table, conventionally the LUT it was read from.
    """

    entries: dict[int, ColorTableEntry]
    fname: str = ""


@dataclass
class MGHTags:
    """The tags of an MGH volume, in file order.

    Parameters
    ----------
    tags : list of tuple of (int, bytes)
        Tag id and raw payload, excluding the id and any length field.
    trailer : bytes, default=b""
        Bytes after the last tag that FreeSurfer does not read as one, such as
        a zero id it treats as the end. Kept so a rewrite loses nothing.
    """

    tags: list[tuple[int, bytes]] = field(default_factory=list)
    trailer: bytes = b""

    @property
    def xform(self) -> str | None:
        """Return the recorded transform path, or ``None`` if there is none."""
        for tag, payload in self.tags:
            if tag in (TAG_MGH_XFORM, TAG_OLD_MGH_XFORM):
                return _decode_c_string(payload)
        return None

    @property
    def color_table(self) -> ColorTable | None:
        """Return the embedded colour table, or ``None`` if there is none."""
        for tag, payload in self.tags:
            if tag == TAG_OLD_COLORTABLE:
                return decode_color_table(payload)[0]
        return None

    @property
    def command_lines(self) -> list[str]:
        """Return the recorded command lines, oldest first."""
        return [_decode_c_string(payload) for tag, payload in self.tags if tag == TAG_CMDLINE]

    def derive(self, *, r2r: Any = None, keep_color_table: bool = True) -> MGHTags:
        """Return the tags an output derived from this volume's values carries.

        Tags describing the values (colour table, phase-encode direction,
        field strength, frame metadata) and the command-line history always
        carry over. Tags tied to the coordinate frame carry over only when the
        output stays in it, meaning ``r2r`` is absent or the identity. When it
        does not, the original RAS-to-voxel matrix is re-expressed in the new
        frame, while the talairach path, the AutoAlign matrix and any unknown
        tag are dropped: none of them can be updated reliably, and FreeSurfer
        would act on a stale value. Anything after a zero tag is not carried.

        Parameters
        ----------
        r2r : array-like of shape (4, 4) or None, optional
            RAS-to-RAS map from this volume's frame to the output's.
        keep_color_table : bool, default=True
            Whether the output still holds the same labels. Pass ``False`` when
            the values were relabelled, for example by binarizing, since the
            table would then name the wrong structures.

        Returns
        -------
        MGHTags
            Tags for the output, in the original order.
        """
        matrix = None if r2r is None else np.asarray(r2r, dtype=np.float64)
        moved = matrix is not None and not np.allclose(matrix, np.eye(4), atol=_IDENTITY_ATOL)
        kept: list[tuple[int, bytes]] = []
        dropped: list[str] = []
        for tag, payload in self.tags:
            if tag == TAG_OLD_COLORTABLE and not keep_color_table:
                dropped.append("colour table")
                continue
            if moved and tag == TAG_ORIG_RAS2VOX:
                try:
                    # Mapping p_new = M p_old means p_old = M^-1 p_new.
                    payload = _encode_matrix(_decode_matrix(payload) @ np.linalg.inv(matrix))
                except (ValueError, np.linalg.LinAlgError):
                    dropped.append("unreadable original RAS-to-voxel matrix")
                    continue
            elif moved and tag not in _VALUE_TAGS and tag != TAG_CMDLINE:
                dropped.append(_FRAME_TAG_NAMES.get(tag, f"tag {tag}"))
                continue
            kept.append((tag, payload))
        if dropped:
            logger.info("Not carrying over: %s.", ", ".join(dropped))
        return MGHTags(kept)

    def rebase_xform(self, source_dir: str | Path, out_dir: str | Path) -> MGHTags:
        """Return the tags with a relative transform path moved to a new directory.

        A relative path is relative to the directory of the volume holding it,
        so writing the volume elsewhere would make it point somewhere else. It
        is re-expressed relative to ``out_dir``, or made absolute when the two
        share no relative path. Absolute paths are kept as they are.

        Parameters
        ----------
        source_dir : str or Path
            Directory of the volume the tags were read from.
        out_dir : str or Path
            Directory of the volume they will be written to.

        Returns
        -------
        MGHTags
            Tags with any relative transform path rebased.
        """
        if os.path.abspath(source_dir) == os.path.abspath(out_dir):
            # Same directory: the path already resolves correctly, and leaving
            # it untouched keeps an in-place rewrite byte for byte.
            return MGHTags(list(self.tags), self.trailer)
        rebased: list[tuple[int, bytes]] = []
        for tag, payload in self.tags:
            if tag in (TAG_MGH_XFORM, TAG_OLD_MGH_XFORM):
                path = _decode_c_string(payload)
                if path and not os.path.isabs(path):
                    path = relative_xform_path(os.path.join(source_dir, path), out_dir)
                    payload = path.encode() + b"\0"
            rebased.append((tag, payload))
        return MGHTags(rebased, self.trailer)

    def with_history(self, entry: str) -> MGHTags:
        """Return the tags with ``entry`` appended as the newest command line."""
        return MGHTags([*self.tags, (TAG_CMDLINE, entry.encode() + b"\0")], self.trailer)

    def conflicts(self, *, xform: str | None = None, color_table: ColorTable | None = None) -> list[str]:
        """Describe each requested tag that would overwrite a different value.

        Setting a tag the volume does not have yet, or one it already holds
        with the same value, is not a conflict. Colour tables are compared by
        their entries, since the recorded file name is only where a table came
        from: the same labels and colours loaded from another copy of a LUT are
        the same table.

        Parameters
        ----------
        xform : str or None, optional
            Transform path that would be recorded.
        color_table : ColorTable or None, optional
            Colour table that would be embedded.

        Returns
        -------
        list of str
            One description per conflict, empty when nothing would be lost.
        """
        found = []
        if xform is not None and self.xform is not None and self.xform != xform:
            found.append(f"it records the talairach xform {self.xform!r}, not {xform!r}")
        existing = self.color_table if color_table is not None else None
        if existing is not None and existing.entries != color_table.entries:
            found.append(
                f"it embeds a different colour table ({len(existing.entries)} entries, from "
                f"{existing.fname or 'an unnamed LUT'!r})"
            )
        return found


# ── colour tables ─────────────────────────────────────────────────────────────


def read_color_lut(path: str | Path) -> ColorTable:
    """Read a FreeSurfer-style ASCII colour lookup table.

    Each entry line is ``index name R G B A``, optionally followed by further
    columns, which are ignored. Lines whose first field is not an integer,
    such as comments and blank lines, are skipped, as FreeSurfer does.

    Parameters
    ----------
    path : str or Path
        LUT file, for example ``FreeSurferColorLUT.txt``.

    Returns
    -------
    ColorTable
        The table, with ``fname`` set to ``path`` as given, matching what
        FreeSurfer records when it embeds a LUT.

    Raises
    ------
    ValueError
        If an entry line is malformed, repeats a label value, or the file
        contains no entries.
    """
    entries: dict[int, ColorTableEntry] = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        try:
            index = int(fields[0])
        except ValueError:
            continue
        if len(fields) < 6:
            raise ValueError(f"{path}:{number}: expected 'index name R G B A', got: {line.strip()!r}")
        try:
            red, green, blue, transparency = (int(value) for value in fields[2:6])
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: colour components must be integers: {line.strip()!r}") from exc
        if index < 0:
            raise ValueError(f"{path}:{number}: label value must not be negative, got {index}")
        if index in entries:
            raise ValueError(f"{path}:{number}: label value {index} is defined twice")
        entries[index] = ColorTableEntry(fields[1], (red, green, blue), transparency)
    if not entries:
        raise ValueError(f"{path} contains no colour table entries")
    return ColorTable(dict(sorted(entries.items())), str(path))


def encode_color_table(table: ColorTable) -> bytes:
    """Serialize a colour table in FreeSurfer's binary version 2 layout.

    Parameters
    ----------
    table : ColorTable
        Table to encode.

    Returns
    -------
    bytes
        Payload of a ``TAG_OLD_COLORTABLE`` tag.

    Raises
    ------
    ValueError
        If the table has no entries.
    """
    if not table.entries:
        raise ValueError("a colour table needs at least one entry")
    fname = table.fname.encode() + b"\0"
    # nentries is the size of FreeSurfer's lookup array, one past the largest
    # label value, and count is how many slots of it are filled.
    out = [struct.pack(">iii", _CTAB_VERSION_2, max(table.entries) + 1, len(fname)), fname]
    out.append(struct.pack(">i", len(table.entries)))
    for index in sorted(table.entries):
        entry = table.entries[index]
        name = entry.name.encode() + b"\0"
        out.append(struct.pack(">ii", index, len(name)))
        out.append(name)
        out.append(struct.pack(">4i", *entry.rgb, entry.transparency))
    return b"".join(out)


def decode_color_table(buf: bytes, pos: int = 0) -> tuple[ColorTable, int]:
    """Decode a binary colour table in either of FreeSurfer's layouts.

    Parameters
    ----------
    buf : bytes
        Buffer holding the table.
    pos : int, default=0
        Offset at which the table starts.

    Returns
    -------
    ColorTable
        The decoded table.
    int
        Offset just past its end, which is how the tag's length is found,
        since this tag carries no length field.

    Raises
    ------
    ValueError
        If the table is truncated, has an unknown version, or is malformed.
    """
    reader = _Reader(buf, pos)
    first = reader.int32()
    entries: dict[int, ColorTableEntry] = {}
    if first > 0:
        # Version 1 has no version marker: the first int is the entry count,
        # and label values are implicit in the order.
        fname = _decode_c_string(reader.take(reader.length()))
        for index in range(first):
            name = _decode_c_string(reader.take(reader.length()))
            entries[index] = ColorTableEntry(name, (reader.int32(), reader.int32(), reader.int32()), reader.int32())
        return ColorTable(entries, fname), reader.pos
    if first != _CTAB_VERSION_2:
        raise ValueError(f"unknown colour table version {-first}")
    nentries = reader.int32()
    fname = _decode_c_string(reader.take(reader.length()))
    for _ in range(reader.length()):
        index = reader.int32()
        if not 0 <= index < nentries:
            raise ValueError(f"colour table label {index} is outside [0, {nentries})")
        if index in entries:
            raise ValueError(f"colour table defines label {index} twice")
        name = _decode_c_string(reader.take(reader.length()))
        entries[index] = ColorTableEntry(name, (reader.int32(), reader.int32(), reader.int32()), reader.int32())
    return ColorTable(entries, fname), reader.pos


# ── MGH tag sections ──────────────────────────────────────────────────────────


def read_mgh_tags(path: str | Path) -> MGHTags:
    """Read the tags of an MGH or MGZ volume.

    Parameters
    ----------
    path : str or Path
        Volume to read. Compression is detected from the content.

    Returns
    -------
    MGHTags
        Its tags in file order; empty if the file has none.

    Raises
    ------
    ValueError
        If the file is not a readable MGH volume, is truncated, or holds a tag
        whose extent cannot be determined.
    """
    return _split_mgh(_read_mgh_bytes(path), str(path))[1]


def write_mgh_tags(
        source: str | Path,
        out: str | Path,
        *,
        xform: str | None = None,
        color_table: ColorTable | None = None,
) -> None:
    """Copy an MGH volume, setting its transform path and/or colour table.

    The header, voxels and scan parameters are copied byte for byte and every
    other tag is kept in its original order, so only the requested tags
    change. A replaced tag keeps its position; a new transform tag goes first,
    where FreeSurfer writes it, and a new colour table goes last. ``out`` may
    be ``source``: the file is replaced only once the new one is complete.

    The one other change is to a kept relative transform path when ``out`` is
    in another directory: it is rebased so it still names the same file, since
    a relative path is resolved from the directory of the volume holding it.

    Parameters
    ----------
    source : str or Path
        Input MGH or MGZ volume.
    out : str or Path
        Output path ending in ``.mgz`` (compressed) or ``.mgh``.
    xform : str or None, optional
        Transform path to record verbatim, replacing any recorded one.
        ``None`` keeps the existing tag.
    color_table : ColorTable or None, optional
        Colour table to embed, replacing any embedded one. ``None`` keeps the
        existing tag.

    Returns
    -------
    None
        The volume is written for its side effect.

    Raises
    ------
    ValueError
        If ``out`` is not an MGH path, ``source`` is not a readable MGH
        volume, or ``xform`` is empty.
    """
    out = Path(out)
    if not out.name.lower().endswith(_MGH_SUFFIXES):
        raise ValueError(f"tags can only be stored in MGH volumes (.mgz or .mgh), not {out.name!r}")
    if xform is not None and not xform:
        raise ValueError("the transform path to record must not be empty")
    body, tags = _split_mgh(_read_mgh_bytes(source), str(source))
    tags = tags.rebase_xform(Path(source).parent, out.parent)
    if xform is not None:
        _replace_tag(
            tags.tags,
            (TAG_MGH_XFORM, xform.encode() + b"\0"),
            replaces=(TAG_MGH_XFORM, TAG_OLD_MGH_XFORM),
            prepend=True,
        )
    if color_table is not None:
        _replace_tag(
            tags.tags,
            (TAG_OLD_COLORTABLE, encode_color_table(color_table)),
            replaces=(TAG_OLD_COLORTABLE,),
            prepend=False,
        )
    _write_mgh(out, body, tags)


def write_mgh_image(image: Any, out: str | Path, tags: MGHTags) -> None:
    """Write an ``MGHImage`` followed by ``tags``.

    nibabel writes the header, voxels and scan parameters, which is everything
    up to where the tags begin, so its output is used unchanged and the tags
    are appended to it.

    Parameters
    ----------
    image : nibabel.MGHImage
        Image to write.
    out : str or Path
        Output path ending in ``.mgz`` (compressed) or ``.mgh``.
    tags : MGHTags
        Tags to append.

    Returns
    -------
    None
        The volume is written for its side effect.

    Raises
    ------
    ValueError
        If ``out`` is not an MGH path.
    """
    out = Path(out)
    if not out.name.lower().endswith(_MGH_SUFFIXES):
        raise ValueError(f"tags can only be stored in MGH volumes (.mgz or .mgh), not {out.name!r}")
    _write_mgh(out, image.to_bytes(), tags)


def inherit_tags(
        source: str | Path,
        out: str | Path,
        *,
        r2r: Any = None,
        keep_color_table: bool = True,
) -> MGHTags:
    """Return the tags an output inherits from the volume its values came from.

    Reads the tags of ``source``, applies :meth:`MGHTags.derive`, and rebases a
    relative transform path from the directory of ``source`` to that of
    ``out``. MGH volumes and NIfTI volumes with FreeSurfer's extension have
    tags; other formats have none. A source whose tags cannot be read
    contributes none, with a warning, rather than failing the command that is
    writing the output.

    Parameters
    ----------
    source : str or Path
        Volume the output's voxel values were derived from.
    out : str or Path
        Path the output will be written to.
    r2r : array-like of shape (4, 4) or None, optional
        RAS-to-RAS map from the frame of ``source`` to that of the output.
    keep_color_table : bool, default=True
        Whether the output still holds the source's labels.

    Returns
    -------
    MGHTags
        Tags for the output.
    """
    try:
        tags = read_image_tags(source)
    except (OSError, ValueError, ImageFileError) as exc:
        logger.warning("Not carrying over the tags of %s: %s", source, exc)
        return MGHTags()
    derived = tags.derive(r2r=r2r, keep_color_table=keep_color_table)
    return derived.rebase_xform(Path(source).parent, Path(out).parent)


def encode_nifti_extension(tags: MGHTags) -> bytes | None:
    """Serialize tags as FreeSurfer's NIfTI header extension (code 14).

    The layout follows FreeSurfer 8's ``niiWrite``: a four-byte header, then
    each tag as an int32 id, an int64 length and the payload, all big-endian,
    then an end tag. ``TAG_RAS_XFORM``, FreeSurfer's own copy of the geometry,
    is not written: FreeSurfer reads the geometry from the sform/qform when it
    is absent, as surfa relies on, and a copy can only go stale. Tags with an
    MGH-only layout are not stored, which is logged.

    Parameters
    ----------
    tags : MGHTags
        Tags in MGH layout.

    Returns
    -------
    bytes or None
        The extension content, or ``None`` when there is nothing to store.
    """
    encoded: list[bytes] = []
    skipped: list[str] = []
    for tag, payload in tags.tags:
        if tag in _MGH_ONLY_TAGS:
            skipped.append(_MGH_ONLY_TAGS[tag])
            continue
        if tag in _MATRIX_TAGS:
            try:
                payload = struct.pack(">16f", *_decode_matrix(payload).ravel())
            except ValueError:
                skipped.append(f"unreadable tag {tag}")
                continue
        elif tag == TAG_OLD_MGH_XFORM:
            # The extension gives every tag a length, so the legacy framing
            # has no meaning there; FreeSurfer writes the path as tag 31.
            tag = TAG_MGH_XFORM
        encoded.append(struct.pack(">iq", tag, len(payload)) + payload)
    if skipped:
        logger.info("Not storing in the NIfTI extension: %s.", ", ".join(skipped))
    if not encoded:
        return None
    end = struct.pack(">iq", _TAG_END_NIIHDREXTENSION, 1) + b"*"
    return _NIFTI_EXTENSION_HEADER + b"".join(encoded) + end


def decode_nifti_extension(content: bytes) -> MGHTags:
    """Parse FreeSurfer's NIfTI header extension into tags in MGH layout.

    NIfTI-only tags (DOF, the geometry copy, scan parameters) are skipped,
    since the image header carries what they describe, and so is the frame
    metadata, whose layout differs. nibabel strips trailing zero bytes from
    extension content, which can shorten the last tag of an extension written
    without an end tag; such a tag is padded back to its declared length.

    Parameters
    ----------
    content : bytes
        Extension content, without the NIfTI size and code fields.

    Returns
    -------
    MGHTags
        The tags.

    Raises
    ------
    ValueError
        If ``content`` is not a FreeSurfer extension or is malformed.
    """
    if content[:1] != _NIFTI_EXTENSION_HEADER[:1]:
        raise ValueError("NIfTI extension 14 does not start with FreeSurfer's big-endian marker")
    reader = _Reader(content, len(_NIFTI_EXTENSION_HEADER))
    tags: list[tuple[int, bytes]] = []
    while len(content) - reader.pos >= 12:
        tag = reader.int32()
        length = reader.int64()
        if tag in (0, _TAG_END_NIIHDREXTENSION):
            break
        if length < 0:
            raise ValueError(f"tag {tag} has negative length {length}")
        remaining = len(content) - reader.pos
        if length > remaining:
            payload = content[reader.pos :] + bytes(length - remaining)
            reader.pos = len(content)
        else:
            payload = reader.take(length)
        if tag in _NIFTI_ONLY_TAGS or tag == TAG_MRI_FRAME:
            continue
        if tag in _MATRIX_TAGS:
            if len(payload) != 64:
                raise ValueError(f"matrix tag {tag} holds {len(payload)} bytes, not 16 float32 values")
            payload = _encode_matrix(np.array(struct.unpack(">16f", payload), dtype=np.float64).reshape(4, 4))
        tags.append((tag, payload))
    return MGHTags(tags)


def read_nifti_tags(path: str | Path) -> MGHTags:
    """Read the FreeSurfer tags of a NIfTI volume, in MGH layout.

    Parameters
    ----------
    path : str or Path
        NIfTI volume. Only the header is read.

    Returns
    -------
    MGHTags
        Its tags; empty if it has no FreeSurfer extension.

    Raises
    ------
    ValueError
        If the FreeSurfer extension is malformed.
    """
    import nibabel as nib

    for extension in getattr(nib.load(str(path)).header, "extensions", []):
        if extension.get_code() == NIFTI_ECODE_FREESURFER:
            return decode_nifti_extension(_extension_bytes(extension))
    return MGHTags()


def read_image_tags(path: str | Path) -> MGHTags:
    """Read the FreeSurfer tags of an MGH or NIfTI volume.

    Parameters
    ----------
    path : str or Path
        Volume to read. Formats that cannot hold tags have none.

    Returns
    -------
    MGHTags
        Its tags, in MGH layout.

    Raises
    ------
    ValueError
        If the volume's tags are malformed.
    """
    name = str(path).lower()
    if name.endswith(_MGH_SUFFIXES):
        return read_mgh_tags(path)
    if name.endswith(_NIFTI_SUFFIXES):
        return read_nifti_tags(path)
    return MGHTags()


def save_nifti_with_tags(image: Any, out: str | Path, tags: MGHTags | None) -> None:
    """Save a NIfTI image with ``tags`` as its FreeSurfer extension.

    Any FreeSurfer extension the image already carries is replaced, never
    duplicated; other extensions are kept. The extension is attached only for
    the save, so the caller's image and every other header field are left as
    they were.

    Parameters
    ----------
    image : nibabel.Nifti1Image
        Image to save.
    out : str or Path
        Output path ending in ``.nii`` or ``.nii.gz``.
    tags : MGHTags or None
        Tags to store. ``None`` or no tags removes a FreeSurfer extension.

    Returns
    -------
    None
        The volume is written for its side effect.
    """
    import nibabel as nib
    from nibabel.nifti1 import Nifti1Extension, Nifti1Extensions

    content = None if tags is None else encode_nifti_extension(tags)
    header = image.header
    original = header.extensions
    kept = [extension for extension in original if extension.get_code() != NIFTI_ECODE_FREESURFER]
    if content is not None:
        kept.append(Nifti1Extension(NIFTI_ECODE_FREESURFER, content))
    header.extensions = Nifti1Extensions(kept)
    try:
        nib.save(image, str(out))
    finally:
        header.extensions = original


def write_image_tags(
        source: str | Path,
        out: str | Path,
        *,
        xform: str | None = None,
        color_table: ColorTable | None = None,
) -> None:
    """Copy an MGH or NIfTI volume, setting its transform path and/or colour table.

    MGH volumes are rewritten at the byte level by :func:`write_mgh_tags`.
    NIfTI volumes are re-saved through nibabel with only the FreeSurfer
    extension changed, which keeps their header and voxel values. ``source``
    and ``out`` must share a format, since this copies rather than converts.

    Parameters
    ----------
    source : str or Path
        Input volume.
    out : str or Path
        Output path, in the same format family as ``source``.
    xform : str or None, optional
        Transform path to record verbatim. ``None`` keeps the existing one.
    color_table : ColorTable or None, optional
        Colour table to embed. ``None`` keeps the existing one.

    Returns
    -------
    None
        The volume is written for its side effect.

    Raises
    ------
    ValueError
        If the formats differ or cannot hold tags, or ``xform`` is empty.
    """
    source_name, out = str(source).lower(), Path(out)
    if source_name.endswith(_MGH_SUFFIXES) and out.name.lower().endswith(_MGH_SUFFIXES):
        write_mgh_tags(source, out, xform=xform, color_table=color_table)
        return
    if not (source_name.endswith(_NIFTI_SUFFIXES) and out.name.lower().endswith(_NIFTI_SUFFIXES)):
        raise ValueError("tags are copied within MGH or within NIfTI; convert between them with vol2vol")
    if xform is not None and not xform:
        raise ValueError("the transform path to record must not be empty")
    import nibabel as nib

    image = nib.load(str(source))
    tags = read_nifti_tags(source).rebase_xform(Path(source).parent, out.parent)
    if xform is not None:
        _replace_tag(tags.tags, (TAG_MGH_XFORM, xform.encode() + b"\0"), replaces=(TAG_MGH_XFORM,), prepend=True)
    if color_table is not None:
        _replace_tag(
            tags.tags,
            (TAG_OLD_COLORTABLE, encode_color_table(color_table)),
            replaces=(TAG_OLD_COLORTABLE,),
            prepend=False,
        )
    # Written next to the output and moved into place, so OUT may be SOURCE:
    # the image still reads its voxels from SOURCE while the new file is built.
    suffix = ".nii.gz" if out.name.lower().endswith(".nii.gz") else ".nii"
    _atomic_save(out, suffix, lambda tmp: save_nifti_with_tags(image, tmp, tags))


def relative_xform_path(target: str | Path, out_dir: str | Path) -> str:
    """Express a transform path relative to the directory of the volume naming it.

    A relative path keeps a subject directory valid when it is moved as a whole,
    but only when the volume and the transform live in it together. When they
    share no directory below the filesystem root, the relative form would only
    climb to the root and back down, breaking as soon as either side moves,
    so the absolute path is returned instead. The same holds for paths on
    different drives, which have no relative form at all.

    Parameters
    ----------
    target : str or Path
        The transform, as an absolute path or one relative to the current
        directory.
    out_dir : str or Path
        Directory of the volume that will record it.

    Returns
    -------
    str
        The path to record.
    """
    target = os.path.abspath(target)
    out_dir = os.path.abspath(out_dir)
    try:
        common = os.path.commonpath([target, out_dir])
    except ValueError:
        return target
    if os.path.dirname(common) == common:
        # The only shared directory is the root itself.
        return target
    return os.path.relpath(target, out_dir)


def history_entry(program: str, arguments: Sequence[str]) -> str:
    """Build a command-line tag recording how a volume was written.

    The fields follow the layout FreeSurfer writes, so ``mri_info`` and other
    readers show it alongside FreeSurfer's own entries. The user and machine
    names FreeSurfer also records are left out, since volumes are often shared
    beyond the place they were made.

    Parameters
    ----------
    program : str
        Name of the command, such as ``"vol2vol"``.
    arguments : sequence of str
        Its arguments, without the program name.

    Returns
    -------
    str
        The entry, for :meth:`MGHTags.with_history`.
    """
    from .._version import __version__

    fields = (
        ("ProgramName", program),
        ("ProgramArguments", shlex.join([program, *arguments])),
        ("ProgramVersion", f"neuroreg {__version__}"),
        ("TimeStamp", time.strftime("%Y/%m/%d-%H:%M:%S-GMT", time.gmtime())),
        ("Platform", platform.system()),
        ("PlatformVersion", platform.release()),
    )
    return "  ".join(f"{name}: {value}" for name, value in fields)


# ── internals ─────────────────────────────────────────────────────────────────


class _Reader:
    """Bounds-checked big-endian reads from a byte buffer."""

    def __init__(self, buf: bytes, pos: int) -> None:
        self.buf = buf
        self.pos = pos

    def take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.buf):
            raise ValueError(f"truncated: needed {n} bytes at offset {self.pos}, {len(self.buf) - self.pos} left")
        chunk = self.buf[self.pos : self.pos + n]
        self.pos += n
        return chunk

    def int32(self) -> int:
        return struct.unpack(">i", self.take(4))[0]

    def int64(self) -> int:
        return struct.unpack(">q", self.take(8))[0]

    def length(self) -> int:
        value = self.int32()
        if value < 0:
            raise ValueError(f"negative length {value} at offset {self.pos - 4}")
        return value


def _decode_c_string(payload: bytes) -> str:
    """Decode a NUL-terminated string, dropping the terminator and any padding."""
    return payload.split(b"\0", 1)[0].decode(errors="replace")


def _read_mgh_bytes(path: str | Path) -> bytes:
    """Return the uncompressed bytes of an MGH or MGZ file."""
    raw = Path(path).read_bytes()
    return gzip.decompress(raw) if raw[:2] == _GZIP_MAGIC else raw


def _split_mgh(raw: bytes, name: str) -> tuple[bytes, MGHTags]:
    """Split an MGH file into everything before the tags, and the tags.

    The first part ends after the scan parameters. A file that stops right
    after its voxels has none, so zeros are supplied, which is what FreeSurfer
    and nibabel assume for a missing value; tags can only follow them.
    """
    try:
        header = MGHHeader.from_fileobj(io.BytesIO(raw))
        footer = header.get_footer_offset()
    except Exception as exc:
        raise ValueError(f"{name} is not a readable MGH volume: {exc}") from exc
    if len(raw) < footer:
        raise ValueError(f"{name} is truncated: the voxel data ends at byte {footer}, the file at {len(raw)}")
    scan_end = footer + _SCAN_PARAMETERS_SIZE
    if len(raw) == footer:
        return raw + bytes(_SCAN_PARAMETERS_SIZE), MGHTags()
    if len(raw) < scan_end:
        raise ValueError(f"{name} is truncated inside its scan parameters")

    tags = MGHTags()
    reader = _Reader(raw, scan_end)
    try:
        while len(raw) - reader.pos >= 4:
            start = reader.pos
            tag = reader.int32()
            if tag == 0:
                # FreeSurfer stops reading here; keep whatever follows.
                reader.pos = start
                break
            if tag in _UNSIZED_WARP_TAGS:
                raise ValueError(
                    f"it holds {_UNSIZED_WARP_TAGS[tag]} ({tag}), a warp-field tag without a length field, "
                    "so the tags after it cannot be located"
                )
            if tag == TAG_OLD_COLORTABLE:
                end = decode_color_table(raw, reader.pos)[1]
                payload = reader.take(end - reader.pos)
            elif tag == TAG_OLD_MGH_XFORM:
                payload = reader.take(reader.length())
            else:
                length = reader.int64()
                if length < 0:
                    raise ValueError(f"tag {tag} has negative length {length}")
                payload = reader.take(length)
            tags.tags.append((tag, payload))
    except ValueError as exc:
        raise ValueError(f"cannot read the tags of {name}: {exc}") from exc
    tags.trailer = raw[reader.pos :]
    return raw[:scan_end], tags


def _write_mgh(out: Path, body: bytes, tags: MGHTags) -> None:
    """Write ``body``, which ends after the scan parameters, then ``tags``."""
    raw = body + b"".join(_encode_tag(tag, payload) for tag, payload in tags.tags) + tags.trailer
    if out.name.lower().endswith(".mgz"):
        raw = gzip.compress(raw, compresslevel=_GZIP_LEVEL)
    _atomic_write(out, raw)


def _decode_matrix(payload: bytes) -> np.ndarray:
    """Parse a FreeSurfer matrix tag: a label followed by 16 numbers, row major."""
    fields = _decode_c_string(payload).split()
    if len(fields) != 17:
        raise ValueError(f"expected a label and 16 matrix entries, got {len(fields)} fields")
    return np.array([float(value) for value in fields[1:]], dtype=np.float64).reshape(4, 4)


def _encode_matrix(matrix: np.ndarray) -> bytes:
    """Format a matrix tag exactly as FreeSurfer's ``FStagsIO::write_matrix`` does."""
    text = "Matrix " + " ".join(f"{value:10f}" for value in np.asarray(matrix, dtype=np.float64).ravel())
    return text.encode().ljust(_MATRIX_STRLEN, b"\0")


def _encode_tag(tag: int, payload: bytes) -> bytes:
    """Frame a tag the way FreeSurfer writes it into an MGH file."""
    if tag == TAG_OLD_COLORTABLE:
        return struct.pack(">i", tag) + payload
    if tag == TAG_OLD_MGH_XFORM:
        return struct.pack(">ii", tag, len(payload)) + payload
    return struct.pack(">iq", tag, len(payload)) + payload


def _replace_tag(
        tags: list[tuple[int, bytes]],
        new: tuple[int, bytes],
        *,
        replaces: tuple[int, ...],
        prepend: bool,
) -> None:
    """Put ``new`` where the first tag in ``replaces`` was, dropping the rest.

    FreeSurfer applies tags in order and a later one of the same kind wins, so
    leaving a duplicate behind would let a stale value override the new one.
    """
    positions = [i for i, (tag, _) in enumerate(tags) if tag in replaces]
    for i in reversed(positions):
        del tags[i]
    if positions:
        tags.insert(positions[0], new)
    elif prepend:
        tags.insert(0, new)
    else:
        tags.append(new)


def _atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` without leaving a partial file behind."""

    def write(tmp: Path) -> None:
        tmp.write_bytes(data)

    _atomic_save(path, ".tmp", write)


def _atomic_save(path: Path, suffix: str, writer: Callable[[Path], None]) -> None:
    """Have ``writer`` fill a temporary file next to ``path``, then move it there.

    The temporary file keeps ``suffix`` so that writers choosing a format from
    the extension, such as nibabel, write the right one. It is created private
    (mode 0600), so it is given the mode the destination would otherwise have:
    the replaced file's, or the umask default for a new one. Without that,
    tagging a volume in a shared subject directory would make it unreadable to
    everyone else.
    """
    if path.exists():
        mode = path.stat().st_mode & 0o7777
    else:
        umask = os.umask(0)
        os.umask(umask)
        mode = 0o666 & ~umask
    handle, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=suffix)
    os.close(handle)
    try:
        writer(Path(tmp))
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _extension_bytes(extension: Any) -> bytes:
    """Return a NIfTI extension's raw content across nibabel versions."""
    content = getattr(extension, "content", None)
    if isinstance(content, bytes):
        return content
    raw = extension.get_content()
    return raw if isinstance(raw, bytes) else bytes(raw)
