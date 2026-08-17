"""Outcome-independent EXP-206 verdict logic."""

from __future__ import annotations

from dataclasses import asdict, dataclass


NULL = 0.50
ECAPA_MATERIAL_POINT = 0.60
WAVLM_MATERIAL_POINT = 0.55
DIRECTIONS = ("primary_mic1_to_mic2", "reverse_mic2_to_mic1")
READOUTS = ("ecapa", "wavlm")


@dataclass(frozen=True)
class Verdict:
    event_signal_present: bool
    material_event_signal: bool
    ecapa_bidirectional: bool
    wavlm_bidirectional: bool
    verdict: str
    manuscript_permission: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def decide(cells: dict[str, dict[str, dict[str, float]]]) -> Verdict:
    """Apply the complete, non-rescuing four-cell truth table."""
    observed: dict[tuple[str, str], tuple[float, float]] = {}
    for direction in DIRECTIONS:
        if set(cells.get(direction, {})) != set(READOUTS):
            raise ValueError(f"invalid readout keys for {direction}")
        for readout in READOUTS:
            record = cells[direction][readout]
            if set(record) != {"point", "lcb"}:
                raise ValueError(f"invalid cell schema: {direction}/{readout}")
            point = float(record["point"])
            lcb = float(record["lcb"])
            if not (0.0 <= lcb <= point <= 1.0):
                raise ValueError(f"invalid cell values: {direction}/{readout}")
            observed[(direction, readout)] = (point, lcb)

    ecapa_bidirectional = all(
        observed[(direction, "ecapa")][1] > NULL for direction in DIRECTIONS
    )
    wavlm_bidirectional = all(
        observed[(direction, "wavlm")][1] > NULL for direction in DIRECTIONS
    )
    present = ecapa_bidirectional and wavlm_bidirectional
    material = present and all(
        observed[(direction, "ecapa")][0] >= ECAPA_MATERIAL_POINT
        and observed[(direction, "wavlm")][0] >= WAVLM_MATERIAL_POINT
        for direction in DIRECTIONS
    )

    if material:
        label = "MATERIAL_BIDIRECTIONAL_CROSS_GENERATOR_CROSS_TEXT"
        permission = "ABSTRACT_AND_CONCLUSION_UPGRADE_PERMITTED"
    elif present:
        label = "POSITIVE_BELOW_MATERIAL_BAR"
        permission = "LIMITED_SUPPORT_ONLY"
    elif ecapa_bidirectional or wavlm_bidirectional:
        label = "ENCODER_DEPENDENT_NOT_CONFIRMED"
        permission = "NO_HEADLINE_UPGRADE"
    else:
        label = "NOT_CONFIRMED"
        permission = "NO_HEADLINE_UPGRADE"
    return Verdict(
        event_signal_present=present,
        material_event_signal=material,
        ecapa_bidirectional=ecapa_bidirectional,
        wavlm_bidirectional=wavlm_bidirectional,
        verdict=label,
        manuscript_permission=permission,
    )
