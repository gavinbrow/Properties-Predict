"""Conservative consensus over multiple engine results.

Rules (per scope philosophy — prefer silence over confident guessing):
1. If a curated lookup hit exists, short-circuit with `reference` tier.
2. Otherwise, gather OK results (drop OUT_OF_DOMAIN / TIMEOUT / ENGINE_ERROR).
3. Zero usable engines -> `unsupported`.
4. One usable engine -> `medium` (or `low` if status=LOW_CONFIDENCE / wide sigma).
5. Multiple usable engines:
   - Compute spread (max - min).
   - If spread <= tight threshold -> uncertainty-weighted mean, tier `high`.
   - If spread <= wide threshold -> uncertainty-weighted mean, tier `medium`,
     and surface a range.
   - Else -> no single value; return `low` with a range.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from server.schemas.engines import EngineResult, EngineStatus
from server.schemas.predict import ConfidenceTier, Method, PropertyPrediction

# (tight, wide) spread thresholds per property, in the property's unit.
SPREAD_THRESHOLDS: dict[str, tuple[float, float]] = {
    "mp": (15.0, 40.0),     # degC
    "bp": (10.0, 30.0),     # degC
    "density": (0.03, 0.10),  # g/mL
}


_USABLE = {EngineStatus.OK, EngineStatus.LOW_CONFIDENCE}


@dataclass(frozen=True, slots=True)
class LookupSignal:
    """Subset of LookupHit the consensus layer needs, kept as a plain struct
    so app.lookup doesn't have to be imported here (avoids a cycle)."""
    value: float
    unit: str
    source: str


def _weight(r: EngineResult) -> float:
    if r.uncertainty is not None and r.uncertainty > 0 and isfinite(r.uncertainty):
        return 1.0 / (r.uncertainty * r.uncertainty)
    return 1.0


def _weighted_mean(results: list[EngineResult]) -> float:
    num = 0.0
    den = 0.0
    for r in results:
        if r.value is None:
            continue
        w = _weight(r)
        num += w * r.value
        den += w
    return num / den if den else float("nan")


def _combined_sigma(results: list[EngineResult]) -> float | None:
    den = sum(_weight(r) for r in results if r.value is not None)
    if den <= 0:
        return None
    return (1.0 / den) ** 0.5


def combine(
    *,
    prop: str,
    engine_results: list[EngineResult],
    lookup: LookupSignal | None,
) -> PropertyPrediction:
    """Apply the rule set above and return a frontend-ready prediction."""
    if lookup is not None:
        return PropertyPrediction(
            property=prop,
            value=lookup.value,
            unit=lookup.unit,
            confidence=ConfidenceTier.REFERENCE,
            method=Method.LOOKUP,
            selected_engine="lookup",
            source=lookup.source,
            engine_details=tuple(engine_results),
        )

    usable = [
        r for r in engine_results if r.status in _USABLE and r.value is not None
    ]

    if not usable:
        warnings = tuple(w for r in engine_results for w in r.warnings) or (
            "no engine produced a trustworthy value",
        )
        return PropertyPrediction(
            property=prop,
            value=None,
            unit=None,
            confidence=ConfidenceTier.UNSUPPORTED,
            method=Method.NONE,
            warnings=warnings,
            engine_details=tuple(engine_results),
        )

    if len(usable) == 1:
        only = usable[0]
        tier = ConfidenceTier.LOW if only.status == EngineStatus.LOW_CONFIDENCE else ConfidenceTier.MEDIUM
        method = Method(only.engine) if only.engine in {m.value for m in Method} else Method.CONSENSUS
        return PropertyPrediction(
            property=prop,
            value=only.value,
            unit=only.unit,
            confidence=tier,
            method=method,
            selected_engine=only.engine,
            uncertainty=only.uncertainty,
            warnings=only.warnings,
            engine_details=tuple(engine_results),
        )

    values = [r.value for r in usable if r.value is not None]
    lo, hi = min(values), max(values)
    spread = hi - lo
    tight, wide = SPREAD_THRESHOLDS.get(prop, (float("inf"), float("inf")))

    any_low = any(r.status == EngineStatus.LOW_CONFIDENCE for r in usable)
    unit = usable[0].unit

    if spread <= tight:
        return PropertyPrediction(
            property=prop,
            value=_weighted_mean(usable),
            unit=unit,
            confidence=ConfidenceTier.MEDIUM if any_low else ConfidenceTier.HIGH,
            method=Method.CONSENSUS,
            uncertainty=_combined_sigma(usable),
            engine_details=tuple(engine_results),
        )
    if spread <= wide:
        return PropertyPrediction(
            property=prop,
            value=_weighted_mean(usable),
            unit=unit,
            confidence=ConfidenceTier.MEDIUM,
            method=Method.CONSENSUS,
            range=(lo, hi),
            uncertainty=_combined_sigma(usable),
            warnings=(f"engines disagree ({lo:.3g}..{hi:.3g} {unit})",),
            engine_details=tuple(engine_results),
        )
    return PropertyPrediction(
        property=prop,
        value=None,
        unit=unit,
        confidence=ConfidenceTier.LOW,
        method=Method.CONSENSUS,
        range=(lo, hi),
        warnings=(
            f"engine spread {spread:.3g} {unit} exceeds wide threshold {wide}",
        ),
        engine_details=tuple(engine_results),
    )
