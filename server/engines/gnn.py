"""Custom GNN adapter supporting archived models and the active Gen 7 model."""
from __future__ import annotations

import hashlib
import json
import math
import pickle
from threading import Lock
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from server.core.config import get_settings
from server.core.constants import PROPERTY_UNITS
from server.engines.base import BaseEngine
from server.schemas.chemistry import NormalizedMolecule
from server.schemas.engines import EngineResult, EngineStatus, error_result

GNN_ROLES: dict[str, str] = {"mp": "secondary", "bp": "secondary"}
SUPPORTED_PROPS = frozenset(GNN_ROLES)

FEATURE_VERSION_V1 = "phase6_graphs_v1"
FEATURE_VERSION_V2 = "phase10_graphs_v2"
FEATURE_VERSION_V3 = "phase11_graphs_v3"
FEATURE_VERSION_V4 = "phase12_graphs_v4"
FEATURE_VERSION_V5 = "phase13_graphs_v5"
FEATURE_VERSION_V5_1 = "phase14_graphs_v5_1"
FEATURE_VERSION_V6 = "phase15_graphs_v6"
FEATURE_VERSION_V7 = "phase15_graphs_v7"
SUPPORTED_FEATURE_VERSIONS = frozenset(
    {
        FEATURE_VERSION_V1,
        FEATURE_VERSION_V2,
        FEATURE_VERSION_V3,
        FEATURE_VERSION_V4,
        FEATURE_VERSION_V5,
        FEATURE_VERSION_V5_1,
        FEATURE_VERSION_V6,
        FEATURE_VERSION_V7,
    }
)
FINGERPRINT_DIM = 2048
_TOPOLOGICAL_DIAMETER_HEAVY_ATOM_CAP = 100
_SMARTS_CACHE: dict[str, object | None] = {}
_MORGAN_GENERATOR = None

GLOBAL_BASE_FEATURE_NAMES_V2 = (
    "mw",
    "heavy_atom_count",
    "num_rotatable_bonds",
    "tpsa",
    "logp",
    "num_h_acceptors",
    "num_h_donors",
    "num_aromatic_rings",
    "fraction_csp3",
    "num_atom_stereo_centers",
    "num_unspecified_atom_stereo_centers",
)

GLOBAL_EXTRA_FEATURE_NAMES_V3 = (
    "ring_count",
    "hetero_count",
    "num_aliphatic_rings",
    "num_saturated_rings",
    "num_bridgehead_atoms",
    "num_spiro_atoms",
    "num_heterocycles",
    "mol_mr",
    "labute_asa",
    "bertz_ct",
)

GLOBAL_EXTRA_FEATURE_NAMES_V6 = GLOBAL_EXTRA_FEATURE_NAMES_V3 + (
    "ring_system_count",
    "largest_ring_system_size",
    "fused_ring_count",
    "murcko_scaffold_atoms",
    "murcko_scaffold_fraction",
    "topological_diameter",
    "carbonyl_count",
    "hydroxyl_count",
    "carboxylic_acid_count",
    "aromatic_atom_fraction",
)

GLOBAL_THREED_FEATURE_NAMES = (
    "conformer_success",
    "radius_of_gyration",
    "asphericity",
    "eccentricity",
    "inertial_shape_factor",
    "npr1",
    "npr2",
    "pbf",
    "pmi1",
    "pmi2",
    "pmi3",
    "spherocity_index",
)

GLOBAL_FEATURE_NAMES_V2 = GLOBAL_BASE_FEATURE_NAMES_V2 + GLOBAL_THREED_FEATURE_NAMES
GLOBAL_FEATURE_NAMES_V3 = (
    GLOBAL_BASE_FEATURE_NAMES_V2 + GLOBAL_EXTRA_FEATURE_NAMES_V3 + GLOBAL_THREED_FEATURE_NAMES
)
GLOBAL_FEATURE_NAMES_V6 = (
    GLOBAL_BASE_FEATURE_NAMES_V2 + GLOBAL_EXTRA_FEATURE_NAMES_V6 + GLOBAL_THREED_FEATURE_NAMES
)


def _one_hot(idx: int, size: int) -> list[float]:
    v = [0.0] * size
    v[idx] = 1.0
    return v


def _feature_spec(version: str) -> dict:
    from rdkit.Chem.rdchem import BondStereo, BondType, ChiralType, HybridizationType

    atom_vocab = ("B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I", "H", "other")
    degree_vocab = (0, 1, 2, 3, 4, 5, "other")
    formal_charge_vocab = (-2, -1, 0, 1, 2, "other")
    hybridization_vocab = (
        HybridizationType.SP,
        HybridizationType.SP2,
        HybridizationType.SP3,
        HybridizationType.SP3D,
        HybridizationType.SP3D2,
        "other",
    )
    hydrogen_count_vocab = (0, 1, 2, 3, 4, "other")
    chirality_vocab = (
        ChiralType.CHI_UNSPECIFIED,
        ChiralType.CHI_TETRAHEDRAL_CW,
        ChiralType.CHI_TETRAHEDRAL_CCW,
        "other",
    )
    bond_type_vocab = (
        BondType.SINGLE,
        BondType.DOUBLE,
        BondType.TRIPLE,
        BondType.AROMATIC,
        "other",
    )
    bond_stereo_vocab = (
        BondStereo.STEREONONE,
        BondStereo.STEREOANY,
        BondStereo.STEREOZ,
        BondStereo.STEREOE,
        BondStereo.STEREOCIS,
        BondStereo.STEREOTRANS,
        "other",
    )
    spec = {
        "version": version,
        "atom": {v: i for i, v in enumerate(atom_vocab)},
        "degree": {v: i for i, v in enumerate(degree_vocab)},
        "formal_charge": {v: i for i, v in enumerate(formal_charge_vocab)},
        "hybridization": {v: i for i, v in enumerate(hybridization_vocab)},
        "hydrogen": {v: i for i, v in enumerate(hydrogen_count_vocab)},
        "chirality": {v: i for i, v in enumerate(chirality_vocab)},
        "bond_type": {v: i for i, v in enumerate(bond_type_vocab)},
        "bond_stereo": {v: i for i, v in enumerate(bond_stereo_vocab)},
        "_sizes": {
            "atom": len(atom_vocab),
            "degree": len(degree_vocab),
            "formal_charge": len(formal_charge_vocab),
            "hybridization": len(hybridization_vocab),
            "hydrogen": len(hydrogen_count_vocab),
            "chirality": len(chirality_vocab),
            "bond_type": len(bond_type_vocab),
            "bond_stereo": len(bond_stereo_vocab),
        },
        "use_cip": version in {FEATURE_VERSION_V2, FEATURE_VERSION_V3, FEATURE_VERSION_V4, FEATURE_VERSION_V5, FEATURE_VERSION_V5_1, FEATURE_VERSION_V6, FEATURE_VERSION_V7},
    }
    if version in {FEATURE_VERSION_V2, FEATURE_VERSION_V3, FEATURE_VERSION_V4, FEATURE_VERSION_V5, FEATURE_VERSION_V5_1, FEATURE_VERSION_V6, FEATURE_VERSION_V7}:
        cip_vocab = ("R", "S", "other")
        spec["cip"] = {v: i for i, v in enumerate(cip_vocab)}
        spec["_sizes"]["cip"] = len(cip_vocab)
    return spec


def _atom_feature_row(atom, spec: dict) -> list[float]:
    sizes = spec["_sizes"]
    atom_idx = spec["atom"].get(atom.GetSymbol(), spec["atom"]["other"])
    degree_idx = spec["degree"].get(atom.GetDegree(), spec["degree"]["other"])
    fc_idx = spec["formal_charge"].get(atom.GetFormalCharge(), spec["formal_charge"]["other"])
    hyb_idx = spec["hybridization"].get(atom.GetHybridization(), spec["hybridization"]["other"])
    h_count = atom.GetTotalNumHs(includeNeighbors=True)
    h_idx = spec["hydrogen"].get(h_count, spec["hydrogen"]["other"])
    chir_idx = spec["chirality"].get(atom.GetChiralTag(), spec["chirality"]["other"])
    row = (
        _one_hot(atom_idx, sizes["atom"])
        + _one_hot(degree_idx, sizes["degree"])
        + _one_hot(fc_idx, sizes["formal_charge"])
        + _one_hot(hyb_idx, sizes["hybridization"])
        + [float(atom.GetIsAromatic())]
        + _one_hot(h_idx, sizes["hydrogen"])
        + _one_hot(chir_idx, sizes["chirality"])
    )
    if spec["use_cip"]:
        cip = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else "other"
        row += _one_hot(spec["cip"].get(cip, spec["cip"]["other"]), sizes["cip"])
        row += [float(atom.HasProp("_ChiralityPossible"))]
    return row


def _bond_feature_row(bond, spec: dict) -> list[float]:
    sizes = spec["_sizes"]
    bt_idx = spec["bond_type"].get(bond.GetBondType(), spec["bond_type"]["other"])
    stereo_idx = spec["bond_stereo"].get(bond.GetStereo(), spec["bond_stereo"]["other"])
    return [
        *_one_hot(bt_idx, sizes["bond_type"]),
        float(bond.IsInRing()),
        float(bond.GetIsConjugated()),
        *_one_hot(stereo_idx, sizes["bond_stereo"]),
    ]


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or out in (float("inf"), float("-inf")):
        return default
    return out


def _compiled_smarts(smarts: str):
    from rdkit import Chem

    if smarts not in _SMARTS_CACHE:
        _SMARTS_CACHE[smarts] = Chem.MolFromSmarts(smarts)
    return _SMARTS_CACHE[smarts]


def _pattern_count(mol, smarts: str) -> float:
    pattern = _compiled_smarts(smarts)
    if pattern is None:
        return 0.0
    return float(len(mol.GetSubstructMatches(pattern)))


def _ring_system_descriptor_values(mol) -> dict[str, float]:
    atom_rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
    if not atom_rings:
        return {
            "ring_system_count": 0.0,
            "largest_ring_system_size": 0.0,
            "fused_ring_count": 0.0,
        }

    remaining = set(range(len(atom_rings)))
    system_count = 0
    largest_system_atoms = 0
    fused_ring_count = 0
    while remaining:
        current = remaining.pop()
        stack = [current]
        component = {current}
        while stack:
            idx = stack.pop()
            ring_atoms = atom_rings[idx]
            neighbors = {
                other
                for other in tuple(remaining)
                if len(ring_atoms & atom_rings[other]) >= 2
            }
            for neighbor in neighbors:
                remaining.remove(neighbor)
                component.add(neighbor)
                stack.append(neighbor)
        system_count += 1
        component_atoms = set().union(*(atom_rings[index] for index in component))
        largest_system_atoms = max(largest_system_atoms, len(component_atoms))
        if len(component) > 1:
            fused_ring_count += len(component)

    return {
        "ring_system_count": float(system_count),
        "largest_ring_system_size": float(largest_system_atoms),
        "fused_ring_count": float(fused_ring_count),
    }


def _murcko_descriptor_values(mol) -> dict[str, float]:
    from rdkit.Chem.Scaffolds import MurckoScaffold

    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        scaffold = None
    scaffold_atoms = float(scaffold.GetNumHeavyAtoms()) if scaffold is not None else 0.0
    heavy_atoms = float(max(mol.GetNumHeavyAtoms(), 1))
    return {
        "murcko_scaffold_atoms": scaffold_atoms,
        "murcko_scaffold_fraction": scaffold_atoms / heavy_atoms,
    }


def _topological_diameter(mol) -> float:
    from rdkit import Chem

    if mol.GetNumHeavyAtoms() > _TOPOLOGICAL_DIAMETER_HEAVY_ATOM_CAP:
        return 0.0
    try:
        distances = Chem.GetDistanceMatrix(mol)
    except Exception:
        return 0.0
    if distances.size == 0:
        return 0.0
    return _safe_float(float(np.max(distances)))


def _shape_descriptor_values(mol) -> dict[str, float]:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors3D, rdMolDescriptors

    mol_3d = Chem.AddHs(Chem.Mol(mol))
    heavy_atoms = mol.GetNumHeavyAtoms()
    rotatable_bonds = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
    ring_count = int(mol.GetRingInfo().NumRings())
    num_confs = 2
    optimize_geometry = True
    max_iters = 75
    if heavy_atoms >= 55 or rotatable_bonds >= 18 or ring_count >= 7:
        return {name: 0.0 for name in GLOBAL_THREED_FEATURE_NAMES}
    elif heavy_atoms >= 45 or rotatable_bonds >= 10 or ring_count >= 6:
        num_confs = 1
        max_iters = 40

    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    params.numThreads = 1
    try:
        conf_ids = list(AllChem.EmbedMultipleConfs(mol_3d, numConfs=num_confs, params=params))
    except Exception:
        return {name: 0.0 for name in GLOBAL_THREED_FEATURE_NAMES}

    if not conf_ids:
        return {name: 0.0 for name in GLOBAL_THREED_FEATURE_NAMES}

    best_conf_id = int(conf_ids[0])
    best_energy = float("inf")
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol_3d):
            results = AllChem.MMFFOptimizeMoleculeConfs(mol_3d, numThreads=1, maxIters=max_iters)
        else:
            results = AllChem.UFFOptimizeMoleculeConfs(mol_3d, numThreads=1, maxIters=max_iters)
    except Exception:
        results = []

    for conf_id, result in zip(conf_ids, results):
        try:
            not_converged, energy = result
            energy_value = float(energy)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(energy_value):
            continue
        scored_energy = energy_value + (1e3 if int(not_converged) else 0.0)
        if scored_energy < best_energy:
            best_energy = scored_energy
            best_conf_id = int(conf_id)

    return {
        "conformer_success": 1.0,
        "radius_of_gyration": _safe_float(Descriptors3D.RadiusOfGyration(mol_3d, confId=best_conf_id)),
        "asphericity": _safe_float(Descriptors3D.Asphericity(mol_3d, confId=best_conf_id)),
        "eccentricity": _safe_float(Descriptors3D.Eccentricity(mol_3d, confId=best_conf_id)),
        "inertial_shape_factor": _safe_float(Descriptors3D.InertialShapeFactor(mol_3d, confId=best_conf_id)),
        "npr1": _safe_float(Descriptors3D.NPR1(mol_3d, confId=best_conf_id)),
        "npr2": _safe_float(Descriptors3D.NPR2(mol_3d, confId=best_conf_id)),
        "pbf": _safe_float(Descriptors3D.PBF(mol_3d, confId=best_conf_id)),
        "pmi1": _safe_float(Descriptors3D.PMI1(mol_3d, confId=best_conf_id)),
        "pmi2": _safe_float(Descriptors3D.PMI2(mol_3d, confId=best_conf_id)),
        "pmi3": _safe_float(Descriptors3D.PMI3(mol_3d, confId=best_conf_id)),
        "spherocity_index": _safe_float(Descriptors3D.SpherocityIndex(mol_3d, confId=best_conf_id)),
    }


def _global_feature_vector(molecule: NormalizedMolecule, mol, feature_version: str) -> list[float]:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    hetero_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6))
    heavy_atoms = float(max(mol.GetNumHeavyAtoms(), 1))
    aromatic_atom_fraction = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic()) / heavy_atoms
    values = {
        "mw": float(molecule.descriptors.mw),
        "heavy_atom_count": float(molecule.descriptors.heavy_atom_count),
        "num_rotatable_bonds": float(molecule.descriptors.num_rotatable_bonds),
        "tpsa": float(molecule.descriptors.tpsa),
        "logp": float(molecule.descriptors.logp),
        "num_h_acceptors": float(molecule.descriptors.num_h_acceptors),
        "num_h_donors": float(molecule.descriptors.num_h_donors),
        "num_aromatic_rings": float(molecule.descriptors.num_aromatic_rings),
        "fraction_csp3": _safe_float(Descriptors.FractionCSP3(mol)),
        "num_atom_stereo_centers": _safe_float(rdMolDescriptors.CalcNumAtomStereoCenters(mol)),
        "num_unspecified_atom_stereo_centers": _safe_float(rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)),
        "ring_count": float(mol.GetRingInfo().NumRings()),
        "hetero_count": float(hetero_count),
        "num_aliphatic_rings": _safe_float(rdMolDescriptors.CalcNumAliphaticRings(mol)),
        "num_saturated_rings": _safe_float(rdMolDescriptors.CalcNumSaturatedRings(mol)),
        "num_bridgehead_atoms": _safe_float(rdMolDescriptors.CalcNumBridgeheadAtoms(mol)),
        "num_spiro_atoms": _safe_float(rdMolDescriptors.CalcNumSpiroAtoms(mol)),
        "num_heterocycles": _safe_float(rdMolDescriptors.CalcNumHeterocycles(mol)),
        "mol_mr": _safe_float(Descriptors.MolMR(mol)),
        "labute_asa": _safe_float(rdMolDescriptors.CalcLabuteASA(mol)),
        "bertz_ct": _safe_float(Descriptors.BertzCT(mol)),
        "topological_diameter": _topological_diameter(mol),
        "carbonyl_count": _pattern_count(mol, "[CX3]=[OX1]"),
        "hydroxyl_count": _pattern_count(mol, "[OX2H]"),
        "carboxylic_acid_count": _pattern_count(mol, "[CX3](=O)[OX2H1]"),
        "aromatic_atom_fraction": _safe_float(aromatic_atom_fraction),
    }
    values.update(_ring_system_descriptor_values(mol))
    values.update(_murcko_descriptor_values(mol))
    shape = _shape_descriptor_values(mol)
    values = {**values, **shape}
    if feature_version in {FEATURE_VERSION_V6, FEATURE_VERSION_V7}:
        feature_names = GLOBAL_FEATURE_NAMES_V6
    elif feature_version in {FEATURE_VERSION_V3, FEATURE_VERSION_V4, FEATURE_VERSION_V5, FEATURE_VERSION_V5_1}:
        feature_names = GLOBAL_FEATURE_NAMES_V3
    else:
        feature_names = GLOBAL_FEATURE_NAMES_V2
    return [_safe_float(values[name]) for name in feature_names]


def _fingerprint_vector(smiles: str) -> list[float]:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    global _MORGAN_GENERATOR
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse canonical SMILES: {smiles}")
    arr = np.zeros((FINGERPRINT_DIM,), dtype=np.uint8)
    if _MORGAN_GENERATOR is None:
        _MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FINGERPRINT_DIM)
    DataStructs.ConvertToNumpyArray(_MORGAN_GENERATOR.GetFingerprint(mol), arr)
    return arr.astype(np.float32, copy=False).tolist()


def _ring_and_hetero_from_smiles(smiles: str) -> tuple[float, float]:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0, 0.0
    return (
        float(mol.GetRingInfo().NumRings()),
        float(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6))),
    )


def _expected_error_features(
    molecule: NormalizedMolecule,
    prediction: dict,
    max_sim: float | None,
    feature_names: list[str] | tuple[str, ...],
) -> np.ndarray:
    ring_count, hetero_count = _ring_and_hetero_from_smiles(molecule.canonical_smiles)
    sigma_total = float(prediction["sigma_total"])
    sigma_epistemic = float(prediction.get("sigma_epistemic", 0.0))
    predicted_value = float(prediction["value"])
    feature_values = {
        "sigma_temp": sigma_total,
        "sigma_aleatoric_temp": float(prediction.get("sigma_aleatoric", 0.0)),
        "sigma_epistemic_temp": sigma_epistemic,
        "epistemic_fraction": sigma_epistemic / max(sigma_total, 1e-6),
        "predicted_value": predicted_value,
        "abs_predicted_value": abs(predicted_value),
        "pred_ge_100": float(predicted_value >= 100.0),
        "pred_ge_200": float(predicted_value >= 200.0),
        "pred_ge_300": float(predicted_value >= 300.0),
        "pred_ge_400": float(predicted_value >= 400.0),
        "max_tanimoto": float(max_sim or 0.0),
        "mw": float(molecule.descriptors.mw),
        "heavy_atom_count": float(molecule.descriptors.heavy_atom_count),
        "num_rotatable_bonds": float(molecule.descriptors.num_rotatable_bonds),
        "tpsa": float(molecule.descriptors.tpsa),
        "logp": float(molecule.descriptors.logp),
        "ring_count": ring_count,
        "hetero_count": hetero_count,
    }
    return np.asarray([float(feature_values.get(name, 0.0)) for name in feature_names], dtype=np.float32).reshape(1, -1)


def _interval_floor_features(molecule: NormalizedMolecule, prediction: dict) -> dict[str, float]:
    return {
        "predicted_value": float(prediction["value"]),
        "mw": float(molecule.descriptors.mw),
        "heavy_atom_count": float(molecule.descriptors.heavy_atom_count),
    }


def _interval_floor_from_rules(
    molecule: NormalizedMolecule,
    prediction: dict,
    rules: dict | None,
    level: float,
) -> float:
    if not rules:
        return 0.0
    values = _interval_floor_features(molecule, prediction)
    level_key = str(level)
    floor = 0.0
    for rule in rules.get("features", []):
        half_width = (rule.get("half_width") or {}).get(level_key)
        if half_width is None:
            continue
        feature_value = values.get(str(rule.get("feature")), 0.0)
        if float(rule["lo"]) <= feature_value < float(rule["hi"]):
            floor = max(floor, float(half_width))
    for rule in rules.get("pairs", []):
        half_width = (rule.get("half_width") or {}).get(level_key)
        if half_width is None:
            continue
        features = [str(name) for name in rule.get("features", [])]
        bounds = rule.get("bounds", [])
        if len(features) != 2 or len(bounds) != 2:
            continue
        matched = True
        for feature, bound in zip(features, bounds):
            feature_value = values.get(feature, 0.0)
            if not (float(bound["lo"]) <= feature_value < float(bound["hi"])):
                matched = False
                break
        if matched:
            floor = max(floor, float(half_width))
    return floor


def _molecule_to_graph(molecule: NormalizedMolecule, spec: dict):
    import torch
    from rdkit import Chem
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(molecule.canonical_smiles)
    if mol is None:
        return None
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    atom_rows = [_atom_feature_row(a, spec) for a in mol.GetAtoms()]
    edge_pairs: list[tuple[int, int]] = []
    edge_rows: list[list[float]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row = _bond_feature_row(bond, spec)
        edge_pairs.extend([(i, j), (j, i)])
        edge_rows.extend([row, row])

    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_rows, dtype=torch.float32)
    else:
        bond_dim = sum(spec["_sizes"][k] for k in ("bond_type", "bond_stereo")) + 2
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, bond_dim), dtype=torch.float32)

    payload = {
        "x": torch.tensor(atom_rows, dtype=torch.float32),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "num_nodes": len(atom_rows),
    }
    if spec["version"] in {FEATURE_VERSION_V2, FEATURE_VERSION_V3, FEATURE_VERSION_V4, FEATURE_VERSION_V5, FEATURE_VERSION_V5_1, FEATURE_VERSION_V6, FEATURE_VERSION_V7}:
        payload["global_features"] = torch.tensor(
            [_global_feature_vector(molecule, mol, spec["version"])],
            dtype=torch.float32,
        )
    if spec["version"] in {FEATURE_VERSION_V4, FEATURE_VERSION_V5, FEATURE_VERSION_V5_1, FEATURE_VERSION_V6, FEATURE_VERSION_V7}:
        payload["fingerprint_features"] = torch.tensor(
            [_fingerprint_vector(molecule.canonical_smiles)],
            dtype=torch.float32,
        )
    return Data(**payload)


def _build_model(atom_dim: int, bond_dim: int, cfg: dict):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch_geometric.nn import (
        AttentionalAggregation,
        BatchNorm,
        GINEConv,
        global_add_pool,
        global_mean_pool,
    )
    from torch_geometric.nn.models import AttentiveFP

    hidden = int(cfg["hidden_dim"])
    num_layers = int(cfg["num_layers"])
    dropout = float(cfg["dropout"])
    min_lv = float(cfg.get("min_logvar", -8.0))
    max_lv = float(cfg.get("max_logvar", 6.0))
    architecture = str(cfg.get("architecture", "gine"))
    pooling = str(cfg.get("pooling", "sum"))
    use_global_features = bool(cfg.get("use_global_features", False) and int(cfg.get("global_feature_dim", 0)) > 0)
    global_feature_dim = int(cfg.get("global_feature_dim", 0))
    descriptor_hidden_dim = int(cfg.get("descriptor_hidden_dim", max(64, hidden // 2)))
    use_fingerprint_features = bool(
        cfg.get("use_fingerprint_features", False) and int(cfg.get("fingerprint_feature_dim", 0)) > 0
    )
    fingerprint_feature_dim = int(cfg.get("fingerprint_feature_dim", 0))
    fingerprint_hidden_dim = int(cfg.get("fingerprint_hidden_dim", max(128, hidden)))
    fusion_hidden_dim = int(cfg.get("fusion_hidden_dim", hidden))
    shared_trunk_hidden_dim = int(cfg.get("shared_trunk_hidden_dim", 0))
    shared_trunk_layers = int(cfg.get("shared_trunk_layers", 1))
    legacy_head_hidden_dim = int(cfg.get("head_hidden_dim", hidden))
    target_expert_hidden_dim = int(cfg.get("target_expert_hidden_dim", cfg.get("head_hidden_dim", hidden)))
    target_expert_layers = int(cfg.get("target_expert_layers", 1))
    descriptor_residual_mode = str(cfg.get("descriptor_residual_mode", "none")).lower()
    descriptor_residual_hidden_dim = int(cfg.get("descriptor_residual_hidden_dim", descriptor_hidden_dim or hidden))
    attentivefp_timesteps = int(cfg.get("attentivefp_timesteps", 2))
    legacy_head_layout = bool(cfg.get("legacy_head_layout", False))
    modern_v4 = (not legacy_head_layout) and (
        use_fingerprint_features or shared_trunk_hidden_dim > 0 or descriptor_residual_mode != "none"
    )

    def mlp(in_dim: int, h: int, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, out_dim),
        )

    class ResidualMLPBlock(nn.Module):
        def __init__(self, dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            return F.gelu(x + self.block(x))

    class TargetExpert(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, layers: int) -> None:
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.blocks = nn.ModuleList(ResidualMLPBlock(hidden_dim, hidden_dim) for _ in range(max(layers - 1, 0)))

        def forward(self, x):
            h = self.proj(x)
            for block in self.blocks:
                h = block(h)
            return h

    class GINEBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.node_encoder = nn.Linear(atom_dim, hidden)
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            self.pooling = pooling
            for _ in range(num_layers):
                self.convs.append(GINEConv(mlp(hidden, hidden, hidden), train_eps=True, edge_dim=bond_dim))
                self.norms.append(BatchNorm(hidden))
            if self.pooling == "attention":
                self.attention_pool = AttentionalAggregation(
                    gate_nn=nn.Sequential(
                        nn.Linear(hidden, hidden),
                        nn.GELU(),
                        nn.Linear(hidden, 1),
                    )
                )
            elif self.pooling in {"sum", "mean"}:
                self.attention_pool = None
            else:
                raise ValueError(f"Unsupported pooling mode: {self.pooling}")

        def forward(self, data):
            h = self.node_encoder(data.x)
            for conv, norm in zip(self.convs, self.norms, strict=True):
                residual = h
                h = conv(h, data.edge_index, data.edge_attr)
                h = norm(h)
                h = F.gelu(h)
                h = F.dropout(h, p=dropout, training=self.training)
                h = h + residual
            batch = getattr(data, "batch", None)
            if batch is None:
                batch = torch.zeros(data.num_nodes, dtype=torch.long, device=h.device)
            if self.pooling == "sum":
                return global_add_pool(h, batch)
            if self.pooling == "mean":
                return global_mean_pool(h, batch)
            assert self.attention_pool is not None
            return self.attention_pool(h, batch)

    class AttentiveFPBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = AttentiveFP(
                in_channels=atom_dim,
                hidden_channels=hidden,
                out_channels=hidden,
                edge_dim=bond_dim,
                num_layers=num_layers,
                num_timesteps=attentivefp_timesteps,
                dropout=dropout,
            )

        def forward(self, data):
            batch = getattr(data, "batch", None)
            if batch is None:
                batch = torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)
            return self.model(data.x, data.edge_index, data.edge_attr, batch)

    if not modern_v4:
        class MultiTaskMVEModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.use_global_features = use_global_features
                self.backbone = GINEBackbone() if architecture == "gine" else AttentiveFPBackbone()
                if use_global_features:
                    self.global_encoder = nn.Sequential(
                        nn.Linear(global_feature_dim, descriptor_hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(descriptor_hidden_dim, descriptor_hidden_dim),
                        nn.GELU(),
                    )
                    self.register_buffer("global_mean", torch.zeros(global_feature_dim), persistent=True)
                    self.register_buffer("global_std", torch.ones(global_feature_dim), persistent=True)
                    self.fusion = nn.Sequential(
                        nn.Linear(hidden + descriptor_hidden_dim, fusion_hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    )
                else:
                    self.global_encoder = None
                    if legacy_head_layout:
                        # Older graph-only checkpoints fed the pooled graph embedding directly to the heads.
                        self.fusion = nn.Identity()
                    else:
                        self.fusion = nn.Sequential(
                            nn.Linear(hidden, fusion_hidden_dim),
                            nn.GELU(),
                            nn.Dropout(dropout),
                        )
                if legacy_head_layout:
                    self.heads = nn.ModuleDict(
                        {
                            target: nn.Sequential(
                                nn.Linear(fusion_hidden_dim, legacy_head_hidden_dim),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(legacy_head_hidden_dim, 2),
                            )
                            for target in ("mp", "bp")
                        }
                    )
                    self.target_experts = None
                    self.output_heads = None
                else:
                    self.target_experts = nn.ModuleDict(
                        {
                            target: TargetExpert(fusion_hidden_dim, target_expert_hidden_dim, target_expert_layers)
                            for target in ("mp", "bp")
                        }
                    )
                    self.output_heads = nn.ModuleDict(
                        {target: nn.Linear(target_expert_hidden_dim, 2) for target in ("mp", "bp")}
                    )
                    self.heads = None

            def _fused(self, data):
                graph_embeddings = self.backbone(data)
                if not self.use_global_features:
                    return self.fusion(graph_embeddings)
                raw_global = data.global_features
                if raw_global.dim() == 1:
                    raw_global = raw_global.view(1, -1)
                normalized = (raw_global - self.global_mean) / self.global_std
                global_embeddings = self.global_encoder(normalized)
                return self.fusion(torch.cat([graph_embeddings, global_embeddings], dim=1))

            def forward(self, data):
                g = self._fused(data)
                means, logvars = [], []
                for target in ("mp", "bp"):
                    if self.heads is not None:
                        out = self.heads[target](g)
                    else:
                        assert self.output_heads is not None and self.target_experts is not None
                        out = self.output_heads[target](self.target_experts[target](g))
                    means.append(out[:, 0])
                    logvars.append(out[:, 1].clamp(min_lv, max_lv))
                return torch.stack(means, dim=1), torch.stack(logvars, dim=1)

        return MultiTaskMVEModel()

    class MultiTaskMVEModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.use_global_features = use_global_features
            self.use_fingerprint_features = use_fingerprint_features
            self.descriptor_residual_mode = descriptor_residual_mode
            self.backbone = GINEBackbone() if architecture == "gine" else AttentiveFPBackbone()
            self.graph_projector = nn.Sequential(
                nn.Linear(hidden, fusion_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            if use_global_features:
                self.global_encoder = nn.Sequential(
                    nn.Linear(global_feature_dim, descriptor_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(descriptor_hidden_dim, descriptor_hidden_dim),
                    nn.GELU(),
                )
                self.global_projector = nn.Sequential(
                    nn.Linear(descriptor_hidden_dim, fusion_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                self.register_buffer("global_mean", torch.zeros(global_feature_dim), persistent=True)
                self.register_buffer("global_std", torch.ones(global_feature_dim), persistent=True)
            else:
                self.global_encoder = None
                self.global_projector = None
                self.register_buffer("global_mean", torch.zeros(0), persistent=True)
                self.register_buffer("global_std", torch.ones(0), persistent=True)
            if use_fingerprint_features:
                self.fingerprint_encoder = nn.Sequential(
                    nn.Linear(fingerprint_feature_dim, fingerprint_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(fingerprint_hidden_dim, fingerprint_hidden_dim),
                    nn.GELU(),
                )
                self.fingerprint_projector = nn.Sequential(
                    nn.Linear(fingerprint_hidden_dim, fusion_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            else:
                self.fingerprint_encoder = None
                self.fingerprint_projector = None
            branch_count = 1 + int(use_global_features) + int(use_fingerprint_features)
            shared_hidden = shared_trunk_hidden_dim if shared_trunk_hidden_dim > 0 else fusion_hidden_dim
            self.shared_input = nn.Sequential(
                nn.Linear(fusion_hidden_dim * branch_count, shared_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.shared_blocks = nn.ModuleList(
                ResidualMLPBlock(shared_hidden, shared_hidden) for _ in range(max(shared_trunk_layers - 1, 0))
            )
            self.target_gates = nn.ModuleDict(
                {
                    target: nn.Sequential(
                        nn.Linear(fusion_hidden_dim * branch_count, fusion_hidden_dim),
                        nn.GELU(),
                        nn.Linear(fusion_hidden_dim, branch_count),
                    )
                    for target in ("mp", "bp")
                }
            )
            self.target_experts = nn.ModuleDict(
                {
                    target: TargetExpert(shared_hidden + fusion_hidden_dim, target_expert_hidden_dim, target_expert_layers)
                    for target in ("mp", "bp")
                }
            )
            self.output_heads = nn.ModuleDict(
                {target: nn.Linear(target_expert_hidden_dim, 2) for target in ("mp", "bp")}
            )
            residual_targets = (
                ("mp", "bp")
                if self.descriptor_residual_mode == "all"
                else (("bp",) if self.descriptor_residual_mode == "bp" else ())
            )
            if self.use_global_features and residual_targets:
                self.descriptor_residual_heads = nn.ModuleDict(
                    {
                        target: nn.Sequential(
                            nn.Linear(descriptor_hidden_dim, descriptor_residual_hidden_dim),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(descriptor_residual_hidden_dim, 1),
                        )
                        for target in residual_targets
                    }
                )
            else:
                self.descriptor_residual_heads = nn.ModuleDict()

        def _encode_branches(self, data):
            branches = [self.graph_projector(self.backbone(data))]
            global_embedding = None
            if self.use_global_features:
                raw_global = data.global_features
                if raw_global.dim() == 1:
                    raw_global = raw_global.view(1, -1)
                normalized = (raw_global - self.global_mean) / self.global_std
                global_embedding = self.global_encoder(normalized)
                branches.append(self.global_projector(global_embedding))
            if self.use_fingerprint_features:
                raw_fp = data.fingerprint_features
                if raw_fp.dim() == 1:
                    raw_fp = raw_fp.view(1, -1)
                branches.append(self.fingerprint_projector(self.fingerprint_encoder(raw_fp)))
            flat = torch.cat(branches, dim=1)
            shared = self.shared_input(flat)
            for block in self.shared_blocks:
                shared = block(shared)
            return branches, flat, shared, global_embedding

        def forward(self, data):
            branches, flat, shared, global_embedding = self._encode_branches(data)
            branch_stack = torch.stack(branches, dim=1)
            means, logvars = [], []
            for target in ("mp", "bp"):
                weights = torch.softmax(self.target_gates[target](flat), dim=1).unsqueeze(-1)
                target_mix = torch.sum(branch_stack * weights, dim=1)
                expert_input = torch.cat([shared, target_mix], dim=1)
                out = self.output_heads[target](self.target_experts[target](expert_input))
                if target in self.descriptor_residual_heads and global_embedding is not None:
                    out = out.clone()
                    out[:, 0] = out[:, 0] + self.descriptor_residual_heads[target](global_embedding).squeeze(-1)
                means.append(out[:, 0])
                logvars.append(out[:, 1].clamp(min_lv, max_lv))
            return torch.stack(means, dim=1), torch.stack(logvars, dim=1)

    return MultiTaskMVEModel()


@dataclass(frozen=True)
class _TargetScaler:
    mean: float
    std: float

    def unscale(self, z: float) -> float:
        return z * self.std + self.mean

    def unscale_sigma(self, sigma_z: float) -> float:
        return sigma_z * self.std


class _Model:
    __slots__ = (
        "_members",
        "_train_fps_float",
        "_train_popcounts",
        "_feature_spec",
        "atom_dim",
        "bond_dim",
        "confidence_mode",
        "feature_version",
        "head_index",
        "low_conf_sigma",
        "metadata",
        "expected_error_threshold_c",
        "interval80_status_threshold_c",
        "ok_error_threshold_c",
        "ok_probability_threshold",
        "ood_threshold",
        "prop",
        "root",
        "scaler",
        "temperature",
        "expected_error_model",
    )

    def __init__(self, prop: str, root: Path) -> None:
        self.prop = prop
        self.root = root
        self.metadata: dict = {}
        self.feature_version = FEATURE_VERSION_V1
        self.head_index = 0
        self.scaler = _TargetScaler(0.0, 1.0)
        self.temperature = 1.0
        self.ood_threshold = 0.3
        self.confidence_mode = "sigma_threshold"
        self.low_conf_sigma: float | None = None
        self.expected_error_threshold_c: float | None = None
        self.interval80_status_threshold_c: float | None = None
        self.ok_probability_threshold = 0.70
        self.ok_error_threshold_c: float | None = None
        self.atom_dim = 0
        self.bond_dim = 0
        self._feature_spec: dict | None = None
        self._members: list = []
        self._train_fps_float = None
        self._train_popcounts = None
        self.expected_error_model: dict | None = None

    def ready(self) -> bool:
        return self.root.exists() and (self.root / "metadata.json").exists()

    def load(self) -> None:
        import torch

        meta_path = self.root / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        feature_version = meta.get("feature_version")
        if feature_version not in SUPPORTED_FEATURE_VERSIONS:
            raise ValueError(f"unsupported gnn feature_version for {self.prop}: {feature_version!r}")

        checkpoints = meta.get("checkpoints") or []
        if not isinstance(checkpoints, list) or not checkpoints:
            raise ValueError(f"metadata.checkpoints is empty for {self.prop}")
        resolved_ckpts: list[Path] = []
        for rel in checkpoints:
            p = Path(rel)
            if not p.is_absolute():
                p = (meta_path.parent / p).resolve()
            if not p.exists():
                raise FileNotFoundError(f"missing GNN checkpoint for {self.prop}: {p}")
            resolved_ckpts.append(p)

        self.metadata = meta
        self.feature_version = feature_version
        self.head_index = int(meta.get("head_index", 0))
        self.atom_dim = int(meta["atom_feature_dim"])
        self.bond_dim = int(meta["bond_feature_dim"])
        scaler = meta["target_scaler"]
        self.scaler = _TargetScaler(float(scaler["mean"]), float(scaler["std"]))
        calibration = meta.get("calibration") or {}
        self.temperature = float(calibration.get("temperature", 1.0) or 1.0)
        self.ood_threshold = float(calibration.get("ood_threshold", 0.3))
        self.confidence_mode = str(meta.get("confidence_mode", calibration.get("confidence_mode", "sigma_threshold")))
        self.low_conf_sigma = meta.get("low_confidence_sigma_degc")
        if self.low_conf_sigma is not None:
            self.low_conf_sigma = float(self.low_conf_sigma)
        expected_error_threshold = meta.get("expected_error_threshold_c", calibration.get("expected_error_threshold_c"))
        self.expected_error_threshold_c = (
            float(expected_error_threshold) if expected_error_threshold is not None else None
        )
        interval80_threshold = meta.get("interval80_status_threshold_c", calibration.get("interval80_status_threshold_c"))
        self.interval80_status_threshold_c = (
            float(interval80_threshold) if interval80_threshold is not None else None
        )
        self.ok_probability_threshold = float(
            meta.get("ok_probability_threshold", calibration.get("ok_probability_threshold", 0.70)) or 0.70
        )
        ok_error_threshold = meta.get("ok_error_threshold_c", calibration.get("ok_error_threshold_c"))
        self.ok_error_threshold_c = float(ok_error_threshold) if ok_error_threshold is not None else None
        self._feature_spec = _feature_spec(self.feature_version)
        error_model_rel = meta.get("expected_error_model_artifact") or calibration.get("expected_error_model_artifact")
        if error_model_rel:
            error_model_path = Path(error_model_rel)
            if not error_model_path.is_absolute():
                error_model_path = (meta_path.parent / error_model_path).resolve()
            if error_model_path.exists():
                self.expected_error_model = pickle.loads(error_model_path.read_bytes())
                if self.interval80_status_threshold_c is None:
                    interval80_threshold = self.expected_error_model.get("interval80_status_threshold_c")
                    if interval80_threshold is not None:
                        self.interval80_status_threshold_c = float(interval80_threshold)

        legacy_head_layout = bool(
            meta.get(
                "legacy_head_layout",
                "target_expert_hidden_dim" not in meta
                and "target_expert_layers" not in meta
                and "head_hidden_dim" in meta,
            )
        )

        model_cfg = {
            "architecture": meta.get("architecture", "gine"),
            "hidden_dim": int(meta["hidden_dim"]),
            "num_layers": int(meta["num_layers"]),
            "dropout": float(meta.get("dropout", 0.0)),
            "attentivefp_timesteps": int(meta.get("attentivefp_timesteps", 2)),
            "pooling": str(meta.get("pooling", "sum")),
            "use_global_features": bool(meta.get("use_global_features", False)),
            "global_feature_dim": int(meta.get("global_feature_dim", 0)),
            "descriptor_hidden_dim": int(meta.get("descriptor_hidden_dim", 0)),
            "use_fingerprint_features": bool(meta.get("use_fingerprint_features", False)),
            "fingerprint_feature_dim": int(meta.get("fingerprint_feature_dim", 0)),
            "fingerprint_hidden_dim": int(meta.get("fingerprint_hidden_dim", 0)),
            "fusion_hidden_dim": int(meta.get("fusion_hidden_dim", meta["hidden_dim"])),
            "shared_trunk_hidden_dim": int(meta.get("shared_trunk_hidden_dim", 0)),
            "shared_trunk_layers": int(meta.get("shared_trunk_layers", 1)),
            "target_expert_hidden_dim": int(
                meta.get("target_expert_hidden_dim", meta.get("head_hidden_dim", meta["hidden_dim"]))
            ),
            "target_expert_layers": int(meta.get("target_expert_layers", 1)),
            "descriptor_residual_mode": str(meta.get("descriptor_residual_mode", "none")),
            "descriptor_residual_hidden_dim": int(meta.get("descriptor_residual_hidden_dim", meta.get("descriptor_hidden_dim", 0))),
            "legacy_head_layout": legacy_head_layout,
            "max_logvar": float(meta.get("max_logvar", 6.0)),
            "min_logvar": float(meta.get("min_logvar", -8.0)),
        }

        members = []
        for ckpt_path in resolved_ckpts:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model_state = state["model_state"] if isinstance(state, dict) and "model_state" in state else state
            model = _build_model(self.atom_dim, self.bond_dim, model_cfg)
            model.load_state_dict(model_state)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            members.append(model)
        self._members = members

        fp_rel = meta.get("training_fingerprints")
        if fp_rel:
            fp_path = Path(fp_rel)
            if not fp_path.is_absolute():
                fp_path = (meta_path.parent / fp_path).resolve()
            bundle = np.load(fp_path)
            key = self.prop if self.prop in bundle.files else bundle.files[0]
            arr = bundle[key].astype(np.uint8)
            if arr.ndim != 2:
                raise ValueError(f"training fingerprints for {self.prop} have unexpected shape {arr.shape}")
            self._train_fps_float = arr.astype(np.float32)
            self._train_popcounts = arr.sum(axis=1).astype(np.int32)

    def compute_max_tanimoto(self, smiles_list: list[str]) -> list[float | None]:
        if self._train_fps_float is None:
            return [None] * len(smiles_list)
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator

        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=self._train_fps_float.shape[1])
        out: list[float | None] = []
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                out.append(None)
                continue
            bits = np.zeros((self._train_fps_float.shape[1],), dtype=np.uint8)
            from rdkit import DataStructs

            DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), bits)
            test_fp = bits.astype(np.float32)
            intersect = self._train_fps_float @ test_fp
            test_pop = float(bits.sum())
            denom = self._train_popcounts + test_pop - intersect
            denom = np.where(denom <= 0.0, 1.0, denom)
            sims = intersect / denom
            out.append(float(np.max(sims)))
        return out

    def predict_batch(self, molecules: list[NormalizedMolecule]) -> list[dict | None]:
        if not self._members or self._feature_spec is None:
            raise RuntimeError("GNN model is not loaded")
        import torch
        from torch_geometric.data import Batch

        graphs = []
        positions: list[int] = []
        for idx, molecule in enumerate(molecules):
            graph = _molecule_to_graph(molecule, self._feature_spec)
            if graph is None:
                continue
            graphs.append(graph)
            positions.append(idx)

        results: list[dict | None] = [None] * len(molecules)
        if not graphs:
            return results

        batch = Batch.from_data_list(graphs)
        mean_stack: list = []
        logvar_stack: list = []
        with torch.no_grad():
            for model in self._members:
                mean, logvar = model(batch)
                mean_stack.append(mean)
                logvar_stack.append(logvar)
        means = torch.stack(mean_stack, dim=0)
        logvars = torch.stack(logvar_stack, dim=0)

        ensemble_mean_z = means.mean(dim=0)
        aleatoric_z = torch.exp(logvars).mean(dim=0)
        epistemic_z = means.var(dim=0, unbiased=False)
        total_var_z = (aleatoric_z + epistemic_z).clamp_min(1e-8)

        head = self.head_index
        for local_idx, pos in enumerate(positions):
            z = float(ensemble_mean_z[local_idx, head].item())
            sig_total_z = float(total_var_z[local_idx, head].sqrt().item())
            sig_aleatoric_z = float(aleatoric_z[local_idx, head].sqrt().item())
            sig_epistemic_z = float(epistemic_z[local_idx, head].sqrt().item())
            value = self.scaler.unscale(z)
            sigma_total = self.scaler.unscale_sigma(sig_total_z) * self.temperature
            sigma_aleatoric = self.scaler.unscale_sigma(sig_aleatoric_z) * self.temperature
            sigma_epistemic = self.scaler.unscale_sigma(sig_epistemic_z) * self.temperature
            results[pos] = {
                "value": value,
                "sigma_total": sigma_total,
                "sigma_aleatoric": sigma_aleatoric,
                "sigma_epistemic": sigma_epistemic,
            }
        return results


class GnnEngine(BaseEngine):
    """Custom GNN ensemble adapter."""

    name = "gnn"

    def __init__(self, models_root: Path | None = None) -> None:
        settings = get_settings()
        configured = Path(models_root) if models_root else Path(settings.gnn_models_root)
        self._root = configured
        self._models: dict[str, _Model] = {}
        self._model_lock = Lock()
        self.version = self._discover_version()

    def _discover_version(self) -> str:
        # User-facing version label. The framework versions (torch/pyg) and
        # the per-checkpoint hash are still appended into EngineResult's
        # `engine_version` for audit purposes, but the UI-facing string here
        # is the model generation tag.
        try:
            import torch  # noqa: F401  (verifies the runtime is wired up)
        except Exception:
            return "not-installed"
        return "v7.0"

    def supports(self, prop: str) -> bool:
        return prop in SUPPORTED_PROPS

    def role(self, prop: str) -> str | None:
        return GNN_ROLES.get(prop)

    def _inspect_property(self, prop: str) -> dict:
        root = self._root / prop
        issues: list[str] = []
        metadata: dict | None = None
        meta_path = root / "metadata.json"
        if not root.exists():
            issues.append(f"gnn model directory missing at {root}")
        elif not meta_path.exists():
            issues.append(f"metadata.json missing for {prop} at {root}")
        else:
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                issues.append(f"invalid metadata.json for {prop} at {meta_path}: {e.msg}")
            else:
                if metadata.get("feature_version") not in SUPPORTED_FEATURE_VERSIONS:
                    issues.append(
                        f"unsupported gnn feature_version for {prop}: {metadata.get('feature_version')!r}"
                    )
                checkpoints = metadata.get("checkpoints", [])
                if not isinstance(checkpoints, list) or not checkpoints:
                    issues.append(f"metadata.checkpoints is empty for {prop} at {meta_path}")
                else:
                    missing: list[str] = []
                    for rel in checkpoints:
                        p = Path(rel)
                        if not p.is_absolute():
                            p = (meta_path.parent / p).resolve()
                        if not p.exists():
                            missing.append(str(p))
                    if missing:
                        preview = ", ".join(missing[:3])
                        if len(missing) > 3:
                            preview = f"{preview}, ..."
                        issues.append(f"missing GNN checkpoints for {prop}: {preview}")
        return {
            "ready": not issues,
            "root": str(root),
            "model_version": metadata.get("version") if metadata else None,
            "issues": tuple(issues),
        }

    def status(self) -> dict:
        properties = {prop: self._inspect_property(prop) for prop in sorted(SUPPORTED_PROPS)}
        ready_for = [prop for prop, info in properties.items() if info["ready"]]
        issues: list[str] = []
        if self.version == "not-installed":
            issues.append("torch/torch_geometric not installed in the active environment")
        return {
            "ready": self.version != "not-installed" and bool(ready_for),
            "models_root": str(self._root),
            "ready_for": ready_for,
            "issues": tuple(issues),
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
            for entry in meta.get("checkpoint_sha256", []) or []:
                sha = entry.get("sha256") if isinstance(entry, dict) else None
                if sha:
                    h.update(f"{prop}:{entry.get('seed')}:{sha}".encode())
                    found = True
        return h.hexdigest()[:12] if found else None

    def _get_model(self, prop: str) -> _Model | None:
        with self._model_lock:
            model = self._models.get(prop)
            if model is not None:
                return model
            root = self._root / prop
            model = _Model(prop=prop, root=root)
            if not model.ready():
                return None
            try:
                model.load()
            except (FileNotFoundError, json.JSONDecodeError, ValueError, KeyError):
                return None
            self._models[prop] = model
            return model

    def predict(self, molecule: NormalizedMolecule, prop: str) -> EngineResult:
        return self.predict_batch([molecule], prop)[0]

    def predict_batch(self, molecules: list[NormalizedMolecule], prop: str) -> list[EngineResult]:
        if not self.supports(prop):
            return [
                error_result(
                    engine=self.name,
                    engine_version=self.version,
                    prop=prop,
                    status=EngineStatus.UNSUPPORTED_PROPERTY,
                    message=f"GNN adapter does not cover {prop!r}",
                )
                for _ in molecules
            ]
        inspect = self._inspect_property(prop)
        issues = list(inspect["issues"])
        if self.version == "not-installed":
            issues.append("torch/torch_geometric not installed in the active environment")
        if issues:
            message = "; ".join(dict.fromkeys(issues))
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

        model = self._get_model(prop)
        if model is None:
            return [
                error_result(
                    engine=self.name,
                    engine_version=self.version,
                    prop=prop,
                    status=EngineStatus.NOT_CONFIGURED,
                    message=f"GNN model at {self._root / prop} could not be loaded",
                )
                for _ in molecules
            ]

        smiles = [m.canonical_smiles for m in molecules]
        t0 = time.monotonic()
        try:
            raw = model.predict_batch(molecules)
            max_sims = model.compute_max_tanimoto(smiles)
        except Exception as e:  # pragma: no cover
            rt = int((time.monotonic() - t0) * 1000)
            return [
                error_result(
                    engine=self.name,
                    engine_version=self.version,
                    prop=prop,
                    status=EngineStatus.ENGINE_ERROR,
                    message=f"GNN inference failed: {e}",
                    runtime_ms=rt,
                )
                for _ in molecules
            ]
        runtime_ms = int((time.monotonic() - t0) * 1000)

        results: list[EngineResult] = []
        for molecule, prediction, max_sim in zip(molecules, raw, max_sims, strict=True):
            results.append(self._build_result(model, molecule, prop, prediction, max_sim, runtime_ms))
        return results

    def _build_result(
        self,
        model: _Model,
        molecule: NormalizedMolecule,
        prop: str,
        prediction: dict | None,
        max_sim: float | None,
        runtime_ms: int,
    ) -> EngineResult:
        if prediction is None:
            return error_result(
                engine=self.name,
                engine_version=self.version,
                prop=prop,
                status=EngineStatus.PARSE_ERROR,
                message="GNN could not featurize SMILES",
                runtime_ms=runtime_ms,
            )

        value = float(prediction["value"])
        sigma = float(prediction["sigma_total"])
        status = EngineStatus.OK
        warnings: list[str] = []
        in_domain: bool | None = None
        ad_score: float | None = None
        predicted_abs_error: float | None = None
        ok_probability: float | None = None
        calibrated_uncertainty: float | None = None
        interval50_low: float | None = None
        interval50_high: float | None = None
        interval80_low: float | None = None
        interval80_high: float | None = None
        interval90_low: float | None = None
        interval90_high: float | None = None

        if max_sim is not None:
            ad_score = float(max_sim)
            if max_sim < model.ood_threshold:
                in_domain = False
                status = EngineStatus.OUT_OF_DOMAIN
                warnings.append(
                    f"max-Tanimoto to training set {max_sim:.2f} below threshold {model.ood_threshold}"
                )
            else:
                in_domain = True

        if model.expected_error_model is not None:
            feature_names = model.expected_error_model.get("feature_names") or []
            regressor = model.expected_error_model.get("expected_error_regressor")
            isotonic = model.expected_error_model.get("ok_probability_isotonic")
            if feature_names and regressor is not None and isotonic is not None:
                features = _expected_error_features(molecule, prediction, max_sim, feature_names)
                predicted_abs_error = float(max(regressor.predict(features)[0], 0.0))
                ok_probability = float(np.clip(isotonic.predict([predicted_abs_error])[0], 0.0, 1.0))
                quantile_regressors = model.expected_error_model.get("interval_quantile_regressors") or {}
                interval_offsets = model.expected_error_model.get("interval_conformal_offsets") or {}
                interval_floor_rules = model.expected_error_model.get("interval_floor_rules") or {}
                display_level = float(model.expected_error_model.get("display_interval_level", 0.80) or 0.80)
                interval_half_widths: dict[float, float] = {}
                floor = 0.0
                for level_key, interval_model in sorted(
                    quantile_regressors.items(),
                    key=lambda item: float(item[0]),
                ):
                    level = float(level_key)
                    offset = float(interval_offsets.get(level_key, interval_offsets.get(str(level), 0.0)) or 0.0)
                    half_width = float(max(interval_model.predict(features)[0], 0.0) + offset)
                    half_width = max(half_width, floor)
                    if level >= display_level:
                        half_width = max(half_width, predicted_abs_error)
                    half_width = max(
                        half_width,
                        _interval_floor_from_rules(molecule, prediction, interval_floor_rules, level),
                    )
                    interval_half_widths[level] = half_width
                    floor = half_width

                if interval_half_widths:
                    selected_level = min(interval_half_widths, key=lambda level: abs(level - display_level))
                    interval80_half_width = interval_half_widths[selected_level]
                else:
                    interval80_half_width = max(predicted_abs_error, 0.0) * 1.605
                calibrated_uncertainty = interval80_half_width
                if 0.50 in interval_half_widths:
                    interval50_low = value - interval_half_widths[0.50]
                    interval50_high = value + interval_half_widths[0.50]
                interval80_low = value - interval80_half_width
                interval80_high = value + interval80_half_width
                if 0.90 in interval_half_widths:
                    interval90_low = value - interval_half_widths[0.90]
                    interval90_high = value + interval_half_widths[0.90]

        if status == EngineStatus.OK:
            if predicted_abs_error is not None and model.expected_error_threshold_c is not None:
                if predicted_abs_error > model.expected_error_threshold_c:
                    status = EngineStatus.LOW_CONFIDENCE
                    warnings.append(
                        f"predicted abs error {predicted_abs_error:.1f} C exceeds threshold {model.expected_error_threshold_c:.1f} C"
                    )
            if (
                status == EngineStatus.OK
                and calibrated_uncertainty is not None
                and model.interval80_status_threshold_c is not None
                and calibrated_uncertainty > model.interval80_status_threshold_c
            ):
                status = EngineStatus.LOW_CONFIDENCE
                warnings.append(
                    f"calibrated interval +/-{calibrated_uncertainty:.1f} C exceeds threshold {model.interval80_status_threshold_c:.1f} C"
                )
            if status == EngineStatus.OK and ok_probability is not None:
                if ok_probability < model.ok_probability_threshold:
                    status = EngineStatus.LOW_CONFIDENCE
                    warnings.append(
                        f"predicted OK probability {ok_probability:.2f} below threshold {model.ok_probability_threshold:.2f}"
                    )
            elif status == EngineStatus.OK and model.low_conf_sigma is not None and sigma > model.low_conf_sigma:
                status = EngineStatus.LOW_CONFIDENCE
                warnings.append(f"sigma {sigma:.3g} exceeds low-confidence threshold {model.low_conf_sigma}")

        version_tag = f"{self.version}::{model.metadata.get('version', 'unversioned')}"
        emitted_value = None if status == EngineStatus.OUT_OF_DOMAIN else value
        if ok_probability is not None:
            confidence_score = ok_probability
        else:
            sigma_scale = model.low_conf_sigma if model.low_conf_sigma is not None and model.low_conf_sigma > 0 else 40.0
            confidence_score = float(1.0 / (1.0 + max(sigma, 0.0) / sigma_scale))
        return EngineResult(
            engine=self.name,
            engine_version=version_tag,
            property=prop,
            status=status,
            value=emitted_value,
            unit=PROPERTY_UNITS[prop],
            uncertainty=calibrated_uncertainty if calibrated_uncertainty is not None else sigma,
            in_domain=in_domain,
            ad_score=ad_score,
            confidence_score=confidence_score,
            runtime_ms=runtime_ms,
            warnings=tuple(warnings),
            raw={
                "sigma_aleatoric": prediction["sigma_aleatoric"],
                "sigma_epistemic": prediction["sigma_epistemic"],
                "sigma_total_raw_calibrated": sigma,
                "max_tanimoto": max_sim,
                "temperature": model.temperature,
                "feature_version": model.feature_version,
                "calibrated_uncertainty_c": calibrated_uncertainty,
                "predicted_abs_error_c": predicted_abs_error,
                "ok_probability": ok_probability,
                "interval50_low_c": interval50_low,
                "interval50_high_c": interval50_high,
                "interval80_low_c": interval80_low,
                "interval80_high_c": interval80_high,
                "interval90_low_c": interval90_low,
                "interval90_high_c": interval90_high,
            },
        )
