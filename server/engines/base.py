"""Engine adapter contract.

Every external predictor (OPERA, Chemprop, etc.) implements `PredictionEngine`.
Adapters must not raise on routine failures — they wrap errors into an
`EngineResult` with an appropriate `EngineStatus` so the consensus layer can
treat a crashing engine uniformly with a low-confidence one.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from server.schemas.chemistry import NormalizedMolecule
from server.schemas.engines import EngineResult


@runtime_checkable
class PredictionEngine(Protocol):
    """Contract all prediction engines satisfy."""

    name: str
    version: str

    def supports(self, prop: str) -> bool:
        """Return True if this engine can produce predictions for `prop`."""
        ...

    def predict(self, molecule: NormalizedMolecule, prop: str) -> EngineResult:
        """Predict a single property for one normalized molecule.

        Must never raise for routine conditions (timeouts, parse failures,
        out-of-domain). Pack those into `EngineResult.status` instead.
        """
        ...

    def predict_batch(
        self, molecules: list[NormalizedMolecule], prop: str
    ) -> list[EngineResult]:
        """Optional batch path. Default impl calls `predict` per molecule."""
        ...


class BaseEngine:
    """Minimal base class — provides a default batch loop over `predict`."""

    name: str = "base"
    version: str = "0.0.0"

    def supports(self, prop: str) -> bool:  # pragma: no cover - overridden
        raise NotImplementedError

    def predict(self, molecule: NormalizedMolecule, prop: str) -> EngineResult:  # pragma: no cover
        raise NotImplementedError

    def predict_batch(
        self, molecules: list[NormalizedMolecule], prop: str
    ) -> list[EngineResult]:
        return [self.predict(m, prop) for m in molecules]
