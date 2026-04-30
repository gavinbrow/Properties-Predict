"""API-facing request/response schemas.

The public `PredictionResponse` is the shape the frontend cards render. It
carries the confidence tier, the selected `method`, the engine-level details
used to produce the answer, and any domain rejections.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from server.schemas.chemistry import DomainRejection
from server.schemas.engines import EngineResult


class ConfidenceTier(str, Enum):
    REFERENCE = "reference"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNSUPPORTED = "unsupported"


class Method(str, Enum):
    LOOKUP = "lookup"
    KNN = "knn"
    OPERA = "opera"
    CHEMPROP = "chemprop"
    GNN = "gnn"
    THERMO = "thermo"
    CONSENSUS = "consensus"
    NONE = "none"


class PropertyPrediction(BaseModel):
    model_config = ConfigDict(frozen=True)

    property: str
    value: float | None = None
    unit: str | None = None
    confidence: ConfidenceTier
    method: Method
    selected_engine: str | None = None
    range: tuple[float, float] | None = Field(
        default=None,
        description="Reported when engines disagree and a single value cannot be chosen.",
    )
    uncertainty: float | None = None
    source: str | None = Field(
        default=None, description="For lookup hits, the source dataset name."
    )
    warnings: tuple[str, ...] = ()
    engine_details: tuple[EngineResult, ...] = ()


class CompoundInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_smiles: str
    canonical_smiles: str | None = None
    inchikey: str | None = None
    formula: str | None = None
    mw: float | None = None


class PredictionResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    compound: CompoundInfo
    supported: bool
    rejections: tuple[DomainRejection, ...] = ()
    predictions: tuple[PropertyPrediction, ...] = ()


class PredictionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    smiles: str = Field(description="Input SMILES string.")
    properties: tuple[str, ...] | None = Field(
        default=None,
        description="Subset of supported properties; defaults to all if omitted.",
    )
    engines: tuple[str, ...] | None = Field(
        default=None,
        description="Optional subset of predictive engines to run; defaults to routed engines if omitted.",
    )


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    items: tuple[PredictionRequest, ...]


class BatchPredictionResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    results: tuple[PredictionResponse, ...]
