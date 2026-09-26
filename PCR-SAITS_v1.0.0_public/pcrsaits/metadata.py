from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple


VALID_METADATA_MODES = {"explicit", "paper_legacy_inference"}
VALID_FEATURE_GROUPS = {"pollutant", "sensor", "meteorological"}

# Canonical names/tokens intentionally chosen so the already-audited legacy
# group_id() implementation returns the desired domain tags without changing
# Phase 3/4 numerical core code.
_GROUP_TO_CORE_NAME = {
    "pollutant": "CO(GT)",
    "sensor": "PT08.__PCR_PUBLIC_SENSOR__",
    "meteorological": "T",
}


def validate_feature_names(feature_names: Sequence[str]) -> List[str]:
    names = list(feature_names)
    if not names:
        raise ValueError("feature_names must contain at least one feature.")
    if not all(isinstance(name, str) and name.strip() for name in names):
        raise ValueError("Every feature name must be a non-empty string.")
    return names


def resolve_core_feature_names(
    feature_names: Sequence[str],
    *,
    metadata_mode: str = "explicit",
    feature_groups: Optional[Sequence[str]] = None,
) -> Tuple[List[str], Optional[List[str]]]:
    """Resolve public metadata to names consumed by the audited PCR core.

    `explicit` is the public default. The caller supplies one group per feature;
    those groups are encoded through canonical legacy-recognized names.

    `paper_legacy_inference` preserves paper-suite behavior exactly by passing
    the user's feature names through unchanged and letting the audited legacy
    inference logic determine groups.
    """
    names = validate_feature_names(feature_names)

    if metadata_mode not in VALID_METADATA_MODES:
        raise ValueError(
            "metadata_mode must be one of "
            f"{sorted(VALID_METADATA_MODES)}, got {metadata_mode!r}."
        )

    if metadata_mode == "paper_legacy_inference":
        if feature_groups is not None:
            raise ValueError(
                "feature_groups must be omitted when "
                "metadata_mode='paper_legacy_inference'."
            )
        return names, None

    if feature_groups is None:
        raise ValueError(
            "feature_groups is required when metadata_mode='explicit'. "
            "Provide one of: pollutant, sensor, meteorological for each feature."
        )

    groups = [str(group) for group in feature_groups]
    if len(groups) != len(names):
        raise ValueError(
            "feature_groups must have the same length as feature_names: "
            f"{len(groups)} != {len(names)}."
        )

    invalid = sorted(set(groups) - VALID_FEATURE_GROUPS)
    if invalid:
        raise ValueError(
            f"Unknown feature_groups {invalid}; allowed groups are "
            f"{sorted(VALID_FEATURE_GROUPS)}."
        )

    core_names = [_GROUP_TO_CORE_NAME[group] for group in groups]
    return core_names, groups
