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
import os
import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from nibabel.freesurfer.mghformat import MGHHeader

#: Embedded colour lookup table. Written without a length field.
TAG_OLD_COLORTABLE = 1
#: Command line of a program that wrote the volume.
TAG_CMDLINE = 3
#: Legacy transform-path tag, with an int32 length that counts the NUL.
TAG_OLD_MGH_XFORM = 30
#: Path of the talairach transform associated with the volume.
TAG_MGH_XFORM = 31

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
    raw = body + b"".join(_encode_tag(tag, payload) for tag, payload in tags.tags) + tags.trailer
    if out.name.lower().endswith(".mgz"):
        raw = gzip.compress(raw, compresslevel=_GZIP_LEVEL)
    _atomic_write(out, raw)


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
    """Write ``data`` to ``path`` without leaving a partial file behind.

    The temporary file is created private (mode 0600), so it is given the mode
    the destination would otherwise have: the replaced file's, or the umask
    default for a new one. Without that, tagging a volume in a shared subject
    directory would make it unreadable to everyone else.
    """
    if path.exists():
        mode = path.stat().st_mode & 0o7777
    else:
        umask = os.umask(0)
        os.umask(umask)
        mode = 0o666 & ~umask
    handle, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
