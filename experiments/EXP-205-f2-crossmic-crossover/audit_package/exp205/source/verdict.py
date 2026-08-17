"""Outcome-independent verdict logic for the draft EXP-205 protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass


NULL = 0.50
OPERATIONAL_POINT = 0.80
OPERATIONAL_LCB = 0.70


@dataclass(frozen=True)
class PrimaryVerdict:
    existence: bool
    operational: bool
    representation_replication: bool
    verdict: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReverseVerdict:
    ecapa_positive: bool
    wavlm_positive: bool
    verdict: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def primary_verdict(
    *, ecapa_point: float, ecapa_lcb: float, wavlm_lcb: float
) -> PrimaryVerdict:
    """Return the complete non-rescuing primary truth-table state."""
    values = (ecapa_point, ecapa_lcb, wavlm_lcb)
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"probability outside [0,1]: {values}")
    if ecapa_lcb > ecapa_point:
        raise ValueError("ECAPA lower bound exceeds its point estimate")

    existence = ecapa_lcb > NULL
    operational = ecapa_point >= OPERATIONAL_POINT and ecapa_lcb > OPERATIONAL_LCB
    replication = wavlm_lcb > NULL

    if operational and not existence:
        raise AssertionError("operational state must imply ECAPA existence")
    mapping = {
        (True, True, True): "OPERATIONAL_CROSSMIC_CONFIRMATION",
        (True, True, False): "PRIMARY_ONLY",
        (True, False, True): "REPLICATED_WEAK_POSITIVE",
        (True, False, False): "WEAK_PRIMARY_ONLY",
        (False, False, True): "SECONDARY_ONLY_NOT_CONFIRMED",
        (False, False, False): "NOT_CONFIRMED",
    }
    state = (existence, operational, replication)
    if state not in mapping:
        raise AssertionError(f"unhandled primary state: {state}")
    return PrimaryVerdict(*state, verdict=mapping[state])


def reverse_verdict(*, ecapa_lcb: float, wavlm_lcb: float) -> ReverseVerdict:
    """Return reverse-direction status; it never alters the primary state."""
    values = (ecapa_lcb, wavlm_lcb)
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"probability outside [0,1]: {values}")
    ecapa_positive = ecapa_lcb > NULL
    wavlm_positive = wavlm_lcb > NULL
    mapping = {
        (True, True): "BIDIRECTIONAL_REPLICATION",
        (True, False): "REVERSE_REPRESENTATION_DEPENDENT",
        (False, True): "REVERSE_REPRESENTATION_DEPENDENT",
        (False, False): "REVERSE_NOT_CONFIRMED",
    }
    state = (ecapa_positive, wavlm_positive)
    return ReverseVerdict(*state, verdict=mapping[state])


def headline_permission(primary: PrimaryVerdict, reverse: ReverseVerdict) -> str:
    """Encode the only state that permits a bidirectional headline."""
    if (
        primary.verdict == "OPERATIONAL_CROSSMIC_CONFIRMATION"
        and reverse.verdict == "BIDIRECTIONAL_REPLICATION"
    ):
        return "BIDIRECTIONAL_HEADLINE_PERMITTED"
    if primary.verdict == "OPERATIONAL_CROSSMIC_CONFIRMATION":
        return "PRIMARY_DIRECTIONAL_HEADLINE_ONLY"
    return "NO_OPERATIONAL_HEADLINE"

