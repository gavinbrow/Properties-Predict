"""Per-request orchestration.

Steps:
1. Normalize the SMILES via RDKit.
2. Run domain filters; if rejected, return an `unsupported` response.
3. For each requested property:
   a. Check the lookup DB; if hit, short-circuit.
   b. Otherwise call the engines routed to that property, in parallel-safe
      sequence (they all talk to disk/subprocesses, so serial is fine here).
   c. Feed engine results + optional lookup into the consensus combiner.
4. Log a one-line decision summary per property for observability.
"""
from __future__ import annotations

import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Iterable
from dataclasses import dataclass

from server.core.constants import (
    PROPERTY_CONDITIONS,
    PROPERTY_UNITS,
    SUPPORTED_PROPERTIES,
)
from server.engines.base import PredictionEngine
from server.engines.gnn import GnnEngine
from server.engines.knn import KnnEngine
from server.engines.thermo import ThermoDensityEngine
from server.lookup.service import LookupHit
from server.lookup.service import lookup as lookup_one
from server.schemas.chemistry import DomainRejection, NormalizedMolecule
from server.schemas.engines import EngineResult, EngineStatus
from server.schemas.predict import (
    CompoundInfo,
    ConfidenceTier,
    Method,
    PredictionResponse,
    PropertyPrediction,
)
from server.services.consensus import LookupSignal, combine
from server.services.domain import check_mol
from server.services.normalize import NormalizationError, normalize, smiles_to_mol

logger = logging.getLogger("properties_predict.orchestrator")


# Property -> ordered list of engine names to consult.
# kNN runs first so the orchestrator can short-circuit on perfect matches
# before paying for the heavier GNN. Legacy adapters remain in the codebase for
# historical audits, but they are not registered or routed in the active app.
PROPERTY_ROUTING: dict[str, tuple[str, ...]] = {
    "mp": ("knn", "gnn"),
    "bp": ("knn", "gnn"),
    "density": ("thermo",),
}


@dataclass(frozen=True, slots=True)
class EngineRegistry:
    by_name: dict[str, PredictionEngine]

    @classmethod
    def default(cls) -> EngineRegistry:
        return cls(
            by_name={
                "knn": KnnEngine(),
                "gnn": GnnEngine(),
                "thermo": ThermoDensityEngine(),
            }
        )

    def get(self, name: str) -> PredictionEngine | None:
        return self.by_name.get(name)


def _unsupported_response(
    smiles: str, rejections: Iterable[DomainRejection]
) -> PredictionResponse:
    return PredictionResponse(
        compound=CompoundInfo(input_smiles=smiles),
        supported=False,
        rejections=tuple(rejections),
        predictions=(),
    )


def _parse_error_response(smiles: str, msg: str) -> PredictionResponse:
    return PredictionResponse(
        compound=CompoundInfo(input_smiles=smiles),
        supported=False,
        rejections=(DomainRejection(code="parse_error", message=msg),),
        predictions=(),
    )


def _lookup_signal(inchikey: str, prop: str) -> tuple[LookupHit | None, LookupSignal | None]:
    hit = lookup_one(inchikey, prop)
    if hit is None:
        return None, None
    return hit, LookupSignal(value=hit.value, unit=hit.unit, source=hit.source)


def _resolve_engine_names(
    registry: EngineRegistry,
    prop: str,
    selected: tuple[str, ...] | None,
) -> tuple[str, ...]:
    routed = PROPERTY_ROUTING.get(prop, ())
    if selected is None:
        return tuple(
            name
            for name in routed
            if (engine := registry.get(name)) is not None and engine.supports(prop)
        )
    allowed = set(selected)
    return tuple(
        name
        for name in routed
        if name in allowed
        and (engine := registry.get(name)) is not None
        and engine.supports(prop)
    )


def _run_engine(
    registry: EngineRegistry,
    molecule: NormalizedMolecule,
    prop: str,
    name: str,
) -> EngineResult:
    engine = registry.get(name)
    if engine is None or not engine.supports(prop):
        raise ValueError(f"engine {name!r} does not support property {prop!r}")
    return engine.predict(molecule, prop)


def _run_engine_tasks(
    registry: EngineRegistry,
    molecule: NormalizedMolecule,
    engine_tasks: list[tuple[str, str]],
) -> dict[str, list[EngineResult]]:
    results: dict[str, list[EngineResult]] = {}
    if not engine_tasks:
        return results

    max_workers = min(len(engine_tasks), max(2, min(os.cpu_count() or 4, 8)))
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="predict") as executor:
        future_map = {
            executor.submit(_run_engine, registry, molecule, prop, engine_name): (prop, engine_name)
            for prop, engine_name in engine_tasks
        }
        for future in as_completed(future_map):
            prop, engine_name = future_map[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - adapters should wrap routine failures
                engine = registry.get(engine_name)
                version = getattr(engine, "version", "unknown") if engine else "unknown"
                result = EngineResult(
                    engine=engine_name,
                    engine_version=version,
                    property=prop,
                    status=EngineStatus.ENGINE_ERROR,
                    warnings=(f"{engine_name} raised unexpectedly: {exc}",),
                )
            results.setdefault(prop, []).append(result)
    return results


def _is_usable_result(result: EngineResult) -> bool:
    return (
        result.status in {EngineStatus.OK, EngineStatus.LOW_CONFIDENCE}
        and result.value is not None
    )


def _order_results(
    prop: str,
    results: list[EngineResult],
    engine_names: tuple[str, ...],
) -> list[EngineResult]:
    order = {name: idx for idx, name in enumerate(engine_names)}
    return sorted(
        results,
        key=lambda item: (order.get(item.engine, len(order)), item.engine),
    )


def _predict_property(
    prop: str,
    *,
    norm: NormalizedMolecule,
    engine_results: list[EngineResult],
) -> PropertyPrediction:
    prediction = combine(prop=prop, engine_results=engine_results, lookup=None)
    _log_decision(norm.inchikey, prop, prediction)
    return _annotate_condition(prediction)


def _run_engines(
    registry: EngineRegistry,
    molecule: NormalizedMolecule,
    prop: str,
    *,
    selected_engines: tuple[str, ...] | None = None,
) -> list[EngineResult]:
    names = _resolve_engine_names(registry, prop, selected_engines)
    if not names:
        return []
    out: list[EngineResult] = []
    for name in names:
        engine = registry.get(name)
        if engine is None or not engine.supports(prop):
            continue
        out.append(engine.predict(molecule, prop))
    return out


def _maybe_run_knn(
    registry: EngineRegistry,
    norm: NormalizedMolecule,
    prop: str,
    selected_engines: tuple[str, ...] | None,
) -> EngineResult | None:
    """Run the kNN engine sequentially (off the parallel pool) so the
    orchestrator can inspect ``shortcircuit_active`` before deciding whether
    to dispatch the GNN. The result is returned regardless of whether the
    short-circuit fired so the consensus combiner can still see the kNN's
    vote in the engine pool.

    Returns ``None`` if the kNN engine is not registered, isn't selected by
    the caller, or doesn't support the property.
    """
    if selected_engines is not None and "knn" not in selected_engines:
        return None
    knn = registry.get("knn")
    if knn is None or not knn.supports(prop):
        return None
    return knn.predict(norm, prop)


def _log_decision(
    inchikey: str, prop: str, prediction: PropertyPrediction
) -> None:
    val = "none" if prediction.value is None else f"{prediction.value:.4g}"
    logger.info(
        "decide inchikey=%s prop=%s method=%s tier=%s value=%s engines=%d",
        inchikey,
        prop,
        prediction.method.value,
        prediction.confidence.value,
        val,
        len(prediction.engine_details),
    )


def predict_for_smiles(
    smiles: str,
    properties: tuple[str, ...] | None = None,
    *,
    registry: EngineRegistry | None = None,
    engines: tuple[str, ...] | None = None,
) -> PredictionResponse:
    """Produce a full prediction response for one SMILES."""
    registry = registry or EngineRegistry.default()
    requested = _resolve_properties(properties)

    try:
        norm = normalize(smiles)
    except NormalizationError as e:
        return _parse_error_response(smiles, str(e))

    mol = smiles_to_mol(smiles)
    domain = check_mol(mol)
    if not domain.in_domain:
        return _unsupported_response(smiles, domain.rejections)

    selected_engines = tuple(dict.fromkeys(engines)) if engines is not None else None

    predictions_by_prop: dict[str, PropertyPrediction] = {}
    pending_props: list[str] = []
    engine_names_by_prop: dict[str, tuple[str, ...]] = {}
    engine_tasks: list[tuple[str, str]] = []

    knn_results_by_prop: dict[str, EngineResult] = {}

    for prop in requested:
        lookup_hit, _ = _lookup_signal(norm.inchikey, prop)
        if lookup_hit is not None:
            prediction = PropertyPrediction(
                property=prop,
                value=lookup_hit.value,
                unit=lookup_hit.unit,
                confidence=ConfidenceTier.REFERENCE,
                method=Method.LOOKUP,
                selected_engine="lookup",
                source=lookup_hit.source,
            )
            _log_decision(norm.inchikey, prop, prediction)
            predictions_by_prop[prop] = _annotate_condition(prediction)
            continue

        # Run kNN sequentially first (cheap, ~1 ms after warm-up). Exact
        # matches short-circuit the GNN. Non-exact kNN results are held back
        # unless every heavier routed engine fails to produce a usable value;
        # that gives prod a low-confidence fallback when the GNN is outside
        # its applicability domain.
        knn_result = _maybe_run_knn(registry, norm, prop, selected_engines)
        knn_shortcircuited = (
            knn_result is not None
            and knn_result.value is not None
            and bool((knn_result.raw or {}).get("shortcircuit_active"))
        )
        if knn_result is not None:
            knn_results_by_prop[prop] = knn_result

        pending_props.append(prop)
        engine_names = _resolve_engine_names(registry, prop, selected_engines)
        engine_names_by_prop[prop] = engine_names
        # kNN already ran sequentially; drop GNN when short-circuit is active.
        skip = {"knn"}
        if knn_shortcircuited:
            skip.add("gnn")
        engine_tasks.extend((prop, name) for name in engine_names if name not in skip)

    results_by_prop = _run_engine_tasks(registry, norm, engine_tasks)
    for prop, knn_result in knn_results_by_prop.items():
        prop_results = results_by_prop.setdefault(prop, [])
        knn_shortcircuited = bool((knn_result.raw or {}).get("shortcircuit_active"))
        has_usable_non_knn = any(_is_usable_result(result) for result in prop_results)
        if knn_shortcircuited or not has_usable_non_knn:
            prop_results.append(knn_result)

    for prop in pending_props:
        engine_results = _order_results(
            prop,
            results_by_prop.get(prop, []),
            engine_names_by_prop.get(prop, ()),
        )
        predictions_by_prop[prop] = _predict_property(
            prop,
            norm=norm,
            engine_results=engine_results,
        )

    return PredictionResponse(
        compound=CompoundInfo(
            input_smiles=smiles,
            canonical_smiles=norm.canonical_smiles,
            inchikey=norm.inchikey,
            formula=norm.formula,
            mw=norm.descriptors.mw,
        ),
        supported=True,
        rejections=(),
        predictions=tuple(predictions_by_prop[prop] for prop in requested),
    )


def _annotate_condition(p: PropertyPrediction) -> PropertyPrediction:
    """Attach the unit if missing; conditions are handled via scope doc."""
    if p.unit is None and p.value is not None:
        return p.model_copy(update={"unit": PROPERTY_UNITS.get(p.property)})
    return p


def _resolve_properties(properties: tuple[str, ...] | None) -> tuple[str, ...]:
    if properties is None:
        # preserve a stable order
        return tuple(p for p in ("mp", "bp", "density") if p in SUPPORTED_PROPERTIES)
    filtered = tuple(p for p in properties if p in SUPPORTED_PROPERTIES)
    return filtered


def property_conditions() -> dict[str, str]:
    return dict(PROPERTY_CONDITIONS)
