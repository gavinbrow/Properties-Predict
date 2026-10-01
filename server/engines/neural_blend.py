"""Serve the frozen Gen 14 recipe as one prediction engine."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
from threading import Lock

import numpy as np

from server.engines.base import BaseEngine
from server.schemas.engines import EngineResult, EngineStatus, error_result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def blend_predictions(recipes: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Equal seed means inside each family, then the locked family weights."""
    if set(recipes) != set(weights) or not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("incomplete or invalid locked blend")
    if any(weight <= 0 for weight in weights.values()):
        raise ValueError("blend weights must be positive")
    means = {name: np.asarray(values).mean(axis=0) for name, values in recipes.items()}
    if any(not np.isfinite(value).all() for value in means.values()):
        raise ValueError("nonfinite neural prediction")
    return sum(weights[name] * value for name, value in means.items())


class NeuralBlendEngine(BaseEngine):
    name = "gnn"

    def __init__(self, models_root: Path) -> None:
        self._root = Path(models_root)
        self._manifest = json.loads((self._root / "release.json").read_text())
        self.version = self._manifest["version"]
        self._members: dict[str, dict] = {}
        self._lock = Lock()
        self._features = None
        self._fingerprints = None
        self._verified = False
        self._load_errors: dict[str, str] = {}

    def supports(self, prop: str) -> bool:
        return prop in {"mp", "bp"}

    def role(self, prop: str) -> str | None:
        return "secondary" if self.supports(prop) else None

    def artifact_digest(self) -> str:
        return sha256_file(self._root / "release.json")[:12]

    def dataset_manifest_digest(self) -> str:
        return self._manifest["split_manifest_sha256"][:12]

    def status(self) -> dict:
        runtime_issues = [
            f"{name} is not installed"
            for name in ("torch", "torch_geometric", "chemprop")
            if importlib.util.find_spec(name) is None
        ]
        properties = {}
        for prop, config in self._manifest["targets"].items():
            issues = list(runtime_issues)
            if prop in self._load_errors:
                issues.append(self._load_errors[prop])
            paths = [entry["path"] for recipe in config["recipes"].values() for entry in recipe]
            paths += [self._manifest["features_path"], self._manifest["fingerprints_path"]]
            issues += [
                f"missing release artifact: {rel}"
                for rel in paths
                if not (self._root / rel).is_file()
            ]
            properties[prop] = {
                "ready": not issues,
                "issues": issues,
                "model_version": self.version,
                "model_label": "Gen 14 locked neural blend",
                "members": len(paths) - 2,
                "recipes": config["weights"],
                "uncertainty_mode": "uncalibrated",
            }
        ready_for = [prop for prop, info in properties.items() if info["ready"]]
        return {
            "ready": len(ready_for) == 2,
            "ready_for": ready_for,
            "issues": [issue for info in properties.values() for issue in info["issues"]],
            "properties": properties,
            "models_root": str(self._root),
        }

    def _verify(self) -> None:
        if self._verified:
            return
        for rel, expected in self._manifest["artifacts"].items():
            if sha256_file(self._root / rel) != expected:
                raise ValueError(f"release artifact checksum mismatch: {rel}")
        from .neural import model, molecular_features

        for module in (model, molecular_features):
            name = Path(module.__file__).name
            if sha256_file(Path(module.__file__)) != self._manifest["source_sha256"][name]:
                raise ValueError(f"release inference source mismatch: {name}")
        if self._manifest["feature_version"] != molecular_features.FEATURE_VERSION:
            raise ValueError("release feature version mismatch")
        with np.load(self._root / self._manifest["features_path"], allow_pickle=False) as data:
            self._features = (data["inchikey"], data["global_features"])
        with np.load(self._root / self._manifest["fingerprints_path"], allow_pickle=False) as data:
            self._fingerprints = {key: data[key] for key in data.files}
        self._verified = True

    def _load(self, prop: str) -> dict:
        with self._lock:
            if prop in self._members:
                return self._members[prop]
            try:
                self._verify()
                import torch

                torch.set_num_threads(max(1, int(os.getenv("GNN_CPU_THREADS", "4"))))
                from chemprop.models import load_model

                from .neural.model import ModelConfig, MultiTaskMVEModel
                from .neural.molecular_features import ATOM_FEATURE_DIM, BOND_FEATURE_DIM

                recipes = {}
                for name, entries in self._manifest["targets"][prop]["recipes"].items():
                    members = []
                    for entry in entries:
                        path = self._root / entry["path"]
                        if name.startswith("gine"):
                            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                            if checkpoint["focus_target"] != prop:
                                raise ValueError("checkpoint target mismatch")
                            model = MultiTaskMVEModel(
                                ATOM_FEATURE_DIM,
                                BOND_FEATURE_DIM,
                                ModelConfig(**checkpoint["model_config"]),
                            )
                            model.load_state_dict(checkpoint["model_state"], strict=True)
                            scaler = checkpoint["target_scaler"]
                        else:
                            model, scaler = load_model(path), None
                        model.eval()
                        model.requires_grad_(False)
                        members.append((model, scaler))
                    recipes[name] = members
                self._members[prop] = recipes
                self._load_errors.pop(prop, None)
                return recipes
            except Exception as exc:
                self._load_errors[prop] = str(exc)
                raise

    def _graph(self, molecule):
        import torch
        from rdkit import Chem
        from torch_geometric.data import Data

        from .neural import molecular_features as features

        mol = Chem.MolFromSmiles(molecule.canonical_smiles)
        if mol is None:
            raise ValueError("could not featurize SMILES")
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        keys, values = self._features
        index = np.searchsorted(keys, molecule.inchikey)
        if index < len(keys) and keys[index] == molecule.inchikey:
            global_values = values[index]
        else:
            global_values = np.asarray(
                features.global_feature_vector(
                    molecule.canonical_smiles,
                    descriptor_overrides=molecule.descriptors.model_dump(),
                ),
                dtype=np.float32,
            )
        edges, attrs = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges.extend(((i, j), (j, i)))
            attrs.extend((features.bond_features(bond), features.bond_features(bond)))
        graph = Data(
            x=torch.tensor(
                [features.atom_features(atom)[0] for atom in mol.GetAtoms()], dtype=torch.float32
            ),
            edge_index=torch.tensor(edges, dtype=torch.long).reshape(-1, 2).t().contiguous(),
            edge_attr=torch.tensor(attrs, dtype=torch.float32).reshape(
                -1, features.BOND_FEATURE_DIM
            ),
            global_features=torch.tensor(global_values[None, :], dtype=torch.float32),
        )
        descriptor_indices = [
            features.GLOBAL_FEATURE_NAMES.index(name) for name in self._manifest["descriptor_names"]
        ]
        return graph, global_values[descriptor_indices]

    def _similarity(self, smiles: str, prop: str) -> float:
        from .neural.molecular_features import fingerprint_vector

        query = np.packbits(np.asarray(fingerprint_vector(smiles), dtype=np.uint8))
        table = self._fingerprints[prop]
        counts = self._fingerprints[prop + "_popcounts"]
        popcount = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
        maximum = 0.0
        for start in range(0, len(table), 8192):
            intersection = popcount[np.bitwise_and(table[start : start + 8192], query)].sum(axis=1)
            union = counts[start : start + 8192] + popcount[query].sum() - intersection
            sims = np.divide(intersection, union, out=np.zeros(len(union)), where=union > 0)
            maximum = max(maximum, float(sims.max()))
        return maximum

    def predict(self, molecule, prop: str) -> EngineResult:
        return self.predict_batch([molecule], prop)[0]

    def predict_batch(self, molecules, prop: str) -> list[EngineResult]:
        started = time.monotonic()
        if not molecules:
            return []
        if not self.supports(prop):
            return [
                error_result(
                    engine=self.name,
                    engine_version=self.version,
                    prop=prop,
                    status=EngineStatus.UNSUPPORTED_PROPERTY,
                    message=f"neural blend does not support {prop}",
                )
                for _ in molecules
            ]
        try:
            import torch
            from chemprop import data
            from torch_geometric.data import Batch

            recipes = self._load(prop)
            graphs, descriptors = zip(*(self._graph(mol) for mol in molecules), strict=True)
            batch = Batch.from_data_list(list(graphs))
            cp_batches = {}
            for mode in ("plain", "desc"):
                points = [
                    data.MoleculeDatapoint.from_smi(
                        mol.canonical_smiles, x_d=desc if mode == "desc" else None
                    )
                    for mol, desc in zip(molecules, descriptors, strict=True)
                ]
                dataset = data.MoleculeDataset(points)
                cp_batches[mode] = next(
                    iter(
                        data.build_dataloader(
                            dataset, batch_size=len(points), shuffle=False, num_workers=0
                        )
                    )
                )
            predictions = {}
            head = ("mp", "bp").index(prop)
            with torch.inference_mode():
                for name, members in recipes.items():
                    values = []
                    for model, scaler in members:
                        if scaler is not None:
                            means, _ = model(batch)
                            values.append(
                                (
                                    means[:, head] * scaler["stds"][head] + scaler["means"][head]
                                ).numpy()
                            )
                        else:
                            bmg, V_d, X_d, *_ = cp_batches[
                                "desc" if name.endswith("desc") else "plain"
                            ]
                            values.append(model(bmg, V_d, X_d).numpy().reshape(-1))
                    predictions[name] = np.stack(values)
            weights = self._manifest["targets"][prop]["weights"]
            blended = blend_predictions(predictions, weights)
            results = []
            for index, (mol, value) in enumerate(zip(molecules, blended, strict=True)):
                similarity = self._similarity(mol.canonical_smiles, prop)
                in_domain = similarity >= self._manifest["ood_threshold"]
                if not np.isfinite(value) or value < -273.15:
                    raise ValueError("nonphysical neural prediction")
                warning = (
                    "Gen 14 blend uncertainty has not been calibrated; treat this estimate as low confidence."
                    if in_domain
                    else "molecule is outside the neural blend training domain"
                )
                results.append(
                    EngineResult(
                        engine=self.name,
                        engine_version=f"{self.version}::{self.artifact_digest()}",
                        property=prop,
                        status=EngineStatus.LOW_CONFIDENCE
                        if in_domain
                        else EngineStatus.OUT_OF_DOMAIN,
                        value=float(value) if in_domain else None,
                        unit="degC",
                        uncertainty=None,
                        in_domain=in_domain,
                        ad_score=similarity,
                        warnings=(warning,),
                        runtime_ms=int((time.monotonic() - started) * 1000),
                        raw={
                            "feature_version": self._manifest["feature_version"],
                            "release": "v14-final",
                            "uncertainty_mode": "uncalibrated",
                            "blend_weights": weights,
                            "recipe_predictions": {
                                name: float(values[:, index].mean())
                                for name, values in predictions.items()
                            },
                            "member_predictions": {
                                name: values[:, index].tolist()
                                for name, values in predictions.items()
                            },
                            "selection_lock_sha256": self._manifest["selection_lock_sha256"],
                        },
                    )
                )
            return results
        except Exception as exc:
            status = (
                EngineStatus.NOT_CONFIGURED
                if isinstance(exc, (FileNotFoundError, ImportError))
                else EngineStatus.ENGINE_ERROR
            )
            return [
                error_result(
                    engine=self.name,
                    engine_version=self.version,
                    prop=prop,
                    status=status,
                    message=f"neural blend failed: {exc}",
                )
                for _ in molecules
            ]
