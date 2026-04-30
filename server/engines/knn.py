"""kNN engine adapter.

Holds, per property, a Morgan-fingerprint matrix of the curated training set
plus the corresponding measured values. At inference time it computes Tanimoto
similarity from the query fingerprint against every stored fingerprint with a
single dense matmul, takes the top-k neighbors, and returns a similarity-
weighted average of their values.

The orchestrator uses this engine in two roles:

1. **Short-circuit**: if the top neighbor is sufficiently similar
   (max_sim >= ``shortcircuit_similarity``, default 0.95) the orchestrator
   accepts the kNN value as-is and skips the heavy engines for that property.
   The query is effectively memorized — running OPERA/Chemprop/GNN would burn
   latency without improving accuracy.

2. **Voting member**: when no short-circuit applies, the kNN result is still
   emitted as one of the engine results so the consensus combiner can use it
   alongside the parametric models. Lower similarities yield lower confidence
   scores and a wider uncertainty estimate, which the combiner respects.

Artifacts (built by ``GNN/kNN/build.py`` and installed by
``GNN/kNN/install.py``) live under ``settings.knn_models_root/{prop}/``:
- ``metadata.json`` — property, fingerprint params, k, threshold, version
- ``data.npz`` — fingerprints (uint8 [N, n_bits]), values (float32 [N]),
  inchikeys, smiles, splits
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import numpy as np

from server.core.config import get_settings
from server.core.constants import PROPERTY_UNITS
from server.engines.base import BaseEngine
from server.schemas.chemistry import NormalizedMolecule
from server.schemas.engines import EngineResult, EngineStatus, error_result

KNN_ROLES: dict[str, str] = {"mp": "secondary", "bp": "secondary"}
SUPPORTED_PROPS = frozenset(KNN_ROLES)
DEFAULT_K = 5
DEFAULT_SHORTCIRCUIT_SIM = 0.95


@dataclass(slots=True)
class _PropertyIndex:
    prop: str
    metadata: dict
    fingerprints: np.ndarray  # (N, n_bits) uint8
    popcounts: np.ndarray  # (N,) int32
    values: np.ndarray  # (N,) float32
    inchikeys: np.ndarray  # (N,) str
    smiles: np.ndarray  # (N,) str
    n_bits: int
    radius: int
    k: int
    shortcircuit_similarity: float
    version: str = field(default="unversioned")

    def lookup_top_k(self, query_bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (similarities, indices) for the top-k neighbors of a query
        fingerprint, sorted descending by similarity.
        """
        intersect = self.fingerprints @ query_bits.astype(np.float32)
        query_pop = float(query_bits.sum())
        denom = self.popcounts + query_pop - intersect
        denom = np.where(denom <= 0.0, 1.0, denom)
        sims = intersect / denom
        if self.k >= len(sims):
            order = np.argsort(-sims)
        else:
            top_unsorted = np.argpartition(-sims, self.k)[: self.k]
            order = top_unsorted[np.argsort(-sims[top_unsorted])]
        return sims[order], order


def _load_property(root: Path, prop: str) -> _PropertyIndex | None:
    """Load and validate a property's kNN artifact bundle. Return None if missing."""
    prop_dir = root / prop
    meta_path = prop_dir / "metadata.json"
    if not meta_path.exists():
        return None
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    data_artifact = metadata.get("data_artifact", "data.npz")
    data_path = prop_dir / data_artifact
    if not data_path.exists():
        raise FileNotFoundError(f"missing kNN data artifact for {prop}: {data_path}")
    bundle = np.load(data_path, allow_pickle=False)
    fingerprints = bundle["fingerprints"].astype(np.uint8)
    if fingerprints.ndim != 2:
        raise ValueError(f"kNN fingerprints for {prop} have unexpected shape {fingerprints.shape}")
    fp_meta = metadata.get("fingerprint", {})
    n_bits = int(fp_meta.get("n_bits", fingerprints.shape[1]))
    if n_bits != fingerprints.shape[1]:
        raise ValueError(
            f"kNN n_bits mismatch for {prop}: metadata={n_bits}, data={fingerprints.shape[1]}"
        )
    radius = int(fp_meta.get("radius", 2))
    values = bundle["values"].astype(np.float32)
    inchikeys = bundle["inchikeys"].astype(str)
    smiles = bundle["smiles"].astype(str)
    if not (len(values) == len(fingerprints) == len(inchikeys) == len(smiles)):
        raise ValueError(f"kNN array length mismatch for {prop}")
    popcounts = fingerprints.sum(axis=1).astype(np.float32)
    return _PropertyIndex(
        prop=prop,
        metadata=metadata,
        fingerprints=fingerprints.astype(np.float32, copy=False),
        popcounts=popcounts,
        values=values,
        inchikeys=inchikeys,
        smiles=smiles,
        n_bits=n_bits,
        radius=radius,
        k=int(metadata.get("k", DEFAULT_K)),
        shortcircuit_similarity=float(
            metadata.get("shortcircuit_similarity", DEFAULT_SHORTCIRCUIT_SIM)
        ),
        version=str(metadata.get("version", "unversioned")),
    )


class KnnEngine(BaseEngine):
    """Curated similarity-weighted lookup over the GNN training set."""

    name = "knn"

    def __init__(self, models_root: Path | None = None) -> None:
        settings = get_settings()
        self._root = Path(models_root) if models_root else Path(settings.knn_models_root)
        self._indices: dict[str, _PropertyIndex] = {}
        self._lock = Lock()
        self.version = "v1"

    def supports(self, prop: str) -> bool:
        return prop in SUPPORTED_PROPS

    def role(self, prop: str) -> str | None:
        return KNN_ROLES.get(prop)

    def _inspect_property(self, prop: str) -> dict:
        prop_dir = self._root / prop
        meta_path = prop_dir / "metadata.json"
        issues: list[str] = []
        metadata: dict | None = None
        if not prop_dir.exists():
            issues.append(f"knn model directory missing at {prop_dir}")
        elif not meta_path.exists():
            issues.append(f"metadata.json missing for {prop} at {prop_dir}")
        else:
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                issues.append(f"invalid metadata.json for {prop} at {meta_path}: {e.msg}")
            else:
                data_artifact = metadata.get("data_artifact", "data.npz")
                if not (prop_dir / data_artifact).exists():
                    issues.append(
                        f"missing kNN data artifact for {prop}: {prop_dir / data_artifact}"
                    )
        return {
            "ready": not issues,
            "root": str(prop_dir),
            "model_version": metadata.get("version") if metadata else None,
            "issues": tuple(issues),
        }

    def status(self) -> dict:
        properties = {prop: self._inspect_property(prop) for prop in sorted(SUPPORTED_PROPS)}
        ready_for = [prop for prop, info in properties.items() if info["ready"]]
        return {
            "ready": bool(ready_for),
            "models_root": str(self._root),
            "ready_for": ready_for,
            "issues": (),
            "properties": properties,
        }

    def artifact_digest(self) -> str | None:
        h = hashlib.sha256()
        found = False
        for prop in sorted(SUPPORTED_PROPS):
            meta_path = self._root / prop / "metadata.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            version = meta.get("version")
            if version:
                h.update(f"{prop}:{version}".encode())
                found = True
        return h.hexdigest()[:12] if found else None

    def _index(self, prop: str) -> _PropertyIndex | None:
        with self._lock:
            cached = self._indices.get(prop)
            if cached is not None:
                return cached
            try:
                index = _load_property(self._root, prop)
            except (FileNotFoundError, ValueError, KeyError):
                return None
            if index is None:
                return None
            self._indices[prop] = index
            return index

    def warm_up(self) -> None:
        """Eagerly load every supported property's index."""
        for prop in sorted(SUPPORTED_PROPS):
            try:
                self._index(prop)
            except Exception:  # pragma: no cover - warmup is best-effort
                continue

    @staticmethod
    def _query_fingerprint(smiles: str, n_bits: int, radius: int) -> np.ndarray | None:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        bits = np.zeros((n_bits,), dtype=np.uint8)
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), bits)
        return bits

    def predict(self, molecule: NormalizedMolecule, prop: str) -> EngineResult:
        return self.predict_batch([molecule], prop)[0]

    def predict_batch(
        self, molecules: list[NormalizedMolecule], prop: str
    ) -> list[EngineResult]:
        if not self.supports(prop):
            return [
                error_result(
                    engine=self.name,
                    engine_version=self.version,
                    prop=prop,
                    status=EngineStatus.UNSUPPORTED_PROPERTY,
                    message=f"kNN adapter does not cover {prop!r}",
                )
                for _ in molecules
            ]
        inspect = self._inspect_property(prop)
        if inspect["issues"]:
            message = "; ".join(inspect["issues"])
            return [
                error_result(
                    engine=self.name,
                    engine_version=self.version,
                    prop=prop,
                    status=EngineStatus.NOT_CONFIGURED,
                    message=message,
                )
                for _ in molecules
            ]
        index = self._index(prop)
        if index is None:
            return [
                error_result(
                    engine=self.name,
                    engine_version=self.version,
                    prop=prop,
                    status=EngineStatus.NOT_CONFIGURED,
                    message=f"kNN index for {prop} could not be loaded from {self._root / prop}",
                )
                for _ in molecules
            ]

        results: list[EngineResult] = []
        for molecule in molecules:
            t0 = time.monotonic()
            bits = self._query_fingerprint(molecule.canonical_smiles, index.n_bits, index.radius)
            if bits is None:
                runtime_ms = int((time.monotonic() - t0) * 1000)
                results.append(
                    error_result(
                        engine=self.name,
                        engine_version=self._version_tag(index),
                        prop=prop,
                        status=EngineStatus.PARSE_ERROR,
                        message="kNN could not featurize SMILES",
                        runtime_ms=runtime_ms,
                    )
                )
                continue
            sims, idxs = index.lookup_top_k(bits)
            runtime_ms = int((time.monotonic() - t0) * 1000)
            results.append(self._build_result(index, prop, sims, idxs, runtime_ms))
        return results

    def _version_tag(self, index: _PropertyIndex) -> str:
        return f"{self.version}::{index.version}"

    def _build_result(
        self,
        index: _PropertyIndex,
        prop: str,
        sims: np.ndarray,
        idxs: np.ndarray,
        runtime_ms: int,
    ) -> EngineResult:
        max_sim = float(sims[0]) if len(sims) else 0.0
        neighbor_values = index.values[idxs]

        # When the top neighbor is a near-exact match the prediction must be
        # *that* neighbor's measured value, not a smoothed average — the
        # orchestrator treats this as a lookup-quality answer. Without this
        # guard, dissimilar runners-up drag e.g. benzene MP from 5.5 -> 97 C.
        if max_sim >= index.shortcircuit_similarity and len(sims):
            value = float(neighbor_values[0])
            uncertainty = 0.0
            weights = np.zeros_like(sims, dtype=np.float32)
            weights[0] = 1.0
        else:
            # Sharp similarity weighting (sim ** SIM_POWER) so an in-domain
            # near-neighbor dominates over distant runners-up. Linear weighting
            # is too soft: with a sim=1.0 + four sim=0.3 neighbors, the top
            # gets only ~46% of the weight, polluting the answer.
            SIM_POWER = 4
            weights = np.clip(sims, 0.0, None).astype(np.float32) ** SIM_POWER
            if weights.sum() <= 0:
                weights = np.ones_like(weights) / max(len(weights), 1)
            else:
                weights = weights / weights.sum()
            value = float((weights * neighbor_values).sum())
            if len(neighbor_values) > 1:
                uncertainty = float(np.sqrt(((neighbor_values - value) ** 2 * weights).sum()))
            else:
                uncertainty = 0.0

        warnings: list[str] = []
        status = EngineStatus.OK
        if max_sim < 0.4:
            status = EngineStatus.OUT_OF_DOMAIN
            warnings.append(f"max-Tanimoto {max_sim:.2f} below kNN domain threshold 0.40")
        elif max_sim < index.shortcircuit_similarity:
            # Not a short-circuit candidate but still a valid voting result.
            warnings.append(
                f"kNN similarity {max_sim:.2f} below short-circuit threshold "
                f"{index.shortcircuit_similarity:.2f}; emitting as voting result"
            )

        emitted_value = None if status == EngineStatus.OUT_OF_DOMAIN else value
        confidence_score = float(np.clip(max_sim, 0.0, 1.0))

        return EngineResult(
            engine=self.name,
            engine_version=self._version_tag(index),
            property=prop,
            status=status,
            value=emitted_value,
            unit=PROPERTY_UNITS[prop],
            uncertainty=uncertainty if emitted_value is not None else None,
            in_domain=max_sim >= 0.4,
            ad_score=float(max_sim),
            confidence_score=confidence_score,
            runtime_ms=runtime_ms,
            warnings=tuple(warnings),
            raw={
                "max_tanimoto": max_sim,
                "neighbor_similarities": [float(s) for s in sims.tolist()],
                "neighbor_values": [float(v) for v in neighbor_values.tolist()],
                "neighbor_weights": [float(w) for w in weights.tolist()],
                "neighbor_inchikeys": [str(k) for k in index.inchikeys[idxs].tolist()],
                "neighbor_smiles": [str(s) for s in index.smiles[idxs].tolist()],
                "shortcircuit_similarity": index.shortcircuit_similarity,
                "shortcircuit_active": bool(max_sim >= index.shortcircuit_similarity),
                "k": int(len(sims)),
            },
        )

    def shortcircuit_threshold(self, prop: str) -> float | None:
        """Return the configured short-circuit similarity for ``prop`` if known."""
        index = self._index(prop)
        return None if index is None else index.shortcircuit_similarity
