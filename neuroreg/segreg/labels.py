"""Label presets and left/right cortical pairing tables for segreg.

The helpers in this module provide stable label subsets for centroid-based
registration modes, especially the left/right pairing used by upright/self-flip
registration.
"""

from __future__ import annotations

from typing import Literal

LabelSetName = Literal["all_shared", "target_centroids", "cortex_lr_pairs"]

CORTEX_LH_LABELS: tuple[int, ...] = (
    1002,
    1003,
    1005,
    1006,
    1007,
    1008,
    1009,
    1010,
    1011,
    1012,
    1013,
    1014,
    1015,
    1016,
    1017,
    1018,
    1019,
    1020,
    1021,
    1022,
    1023,
    1024,
    1025,
    1026,
    1027,
    1028,
    1029,
    1030,
    1031,
    1034,
    1035,
)

CORTEX_RH_LABELS: tuple[int, ...] = (
    2002,
    2003,
    2005,
    2006,
    2007,
    2008,
    2009,
    2010,
    2011,
    2012,
    2013,
    2014,
    2015,
    2016,
    2017,
    2018,
    2019,
    2020,
    2021,
    2022,
    2023,
    2024,
    2025,
    2026,
    2027,
    2028,
    2029,
    2030,
    2031,
    2034,
    2035,
)

CORTEX_LR_PAIRS: tuple[tuple[int, int], ...] = tuple(zip(CORTEX_LH_LABELS, CORTEX_RH_LABELS, strict=True))
CORTEX_LR_LABELS: tuple[int, ...] = CORTEX_LH_LABELS + CORTEX_RH_LABELS


def get_cortex_lr_pairs() -> tuple[tuple[int, int], ...]:
    """Return cortical left/right label pairs used for upright registration.

    Returns
    -------
    tuple[tuple[int, int], ...]
        Paired FastSurfer cortical labels ordered as ``(left, right)``.
    """
    return CORTEX_LR_PAIRS


def get_cortex_lr_labels() -> list[int]:
    """Return the flattened cortical label list used by upright mode.

    Returns
    -------
    list[int]
        Left-hemisphere labels followed by the matching right-hemisphere labels.
    """
    return list(CORTEX_LR_LABELS)
