"""Common schemas shared across engine adapters.

`EngineResult` is the single shape every adapter returns. It is deliberately
conservative: `value` may be `None` when the engine produced no trustworthy
output, in which case `status` carries the reason. The consensus layer reads
`status`, `value`, `uncertainty`, and `in_domain` to decide what to do.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class EngineStatus(str, Enum):
    OK = "ok"
    OUT_OF_DOMAIN = "out_of_domain"
    LOW_CONFIDENCE = "low_confidence"
    UNSUPPORTED_PROPERTY = "unsupported_property"
    PARSE_ERROR = "parse_error"
    TIMEOUT = "timeout"
    ENGINE_ERROR = "engine_error"
    NOT_CONFIGURED = "not_configured"


class EngineResult(BaseModel):
    """Standard per-engine result for one (SMILES, property) request.

    Fields with defaults are optional so adapters can omit what they do not
    produce. `engine` + `engine_version` are always filled by the adapter.
    """

    model_config = ConfigDict(frozen=True)

    engine: str = Field(description="Engine identifier, e.g. 'opera', 'chemprop'.")
    engine_version: str = Field(description="Adapter/model version tag for reproducibility.")
    property: str = Field(description="Property key (mp, bp, density).")
    status: EngineStatus = EngineStatus.OK

    value: float | None = Field(default=None, description="Predicted value in the property's canonical unit, or None.")
    unit: str | None = Field(default=None, description="Canonical unit for `value`.")
    uncertainty: float | None = Field(
        default=None,
        description="Engine-reported uncertainty half-width in the property's unit, if known.",
    )

    in_domain: bool | None = Field(
        default=None, description="True/False if the engine reports applicability-domain info."
    )
    ad_score: float | None = Field(default=None, description="Engine-native AD index [0,1], if provided.")
    confidence_score: float | None = Field(
        default=None, description="Engine-native confidence index [0,1], if provided."
    )

    runtime_ms: int | None = Field(default=None, description="Wall-clock adapter runtime.")
    warnings: tuple[str, ...] = ()
    raw: dict | None = Field(default=None, description="Raw engine output for audit/debug.")


def error_result(
    *,
    engine: str,
    engine_version: str,
    prop: str,
    status: EngineStatus,
    message: str,
    runtime_ms: int | None = None,
) -> EngineResult:
    """Shortcut for non-OK results."""
    return EngineResult(
        engine=engine,
        engine_version=engine_version,
        property=prop,
        status=status,
        warnings=(message,),
        runtime_ms=runtime_ms,
    )
