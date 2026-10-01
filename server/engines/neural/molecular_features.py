"""Shared molecular featurization primitives for the Gen 7 GNN."""
from __future__ import annotations

import math
from collections.abc import Mapping
from functools import lru_cache

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Descriptors3D, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.rdchem import BondStereo, BondType, ChiralType, HybridizationType
from rdkit.Chem.Scaffolds import MurckoScaffold

FEATURE_VERSION = "phase15_graphs_v7"

ATOM_VOCAB = ("B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I", "H", "other")
ATOM_INDEX = {symbol: idx for idx, symbol in enumerate(ATOM_VOCAB)}

DEGREE_VOCAB = (0, 1, 2, 3, 4, 5, "other")
DEGREE_INDEX = {value: idx for idx, value in enumerate(DEGREE_VOCAB)}

FORMAL_CHARGE_VOCAB = (-2, -1, 0, 1, 2, "other")
FORMAL_CHARGE_INDEX = {value: idx for idx, value in enumerate(FORMAL_CHARGE_VOCAB)}

HYBRIDIZATION_VOCAB = (
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
    "other",
)
HYBRIDIZATION_INDEX = {value: idx for idx, value in enumerate(HYBRIDIZATION_VOCAB)}

HYDROGEN_COUNT_VOCAB = (0, 1, 2, 3, 4, "other")
HYDROGEN_COUNT_INDEX = {value: idx for idx, value in enumerate(HYDROGEN_COUNT_VOCAB)}

CHIRALITY_VOCAB = (
    ChiralType.CHI_UNSPECIFIED,
    ChiralType.CHI_TETRAHEDRAL_CW,
    ChiralType.CHI_TETRAHEDRAL_CCW,
    "other",
)
CHIRALITY_INDEX = {value: idx for idx, value in enumerate(CHIRALITY_VOCAB)}

CIP_CODE_VOCAB = ("R", "S", "other")
CIP_CODE_INDEX = {value: idx for idx, value in enumerate(CIP_CODE_VOCAB)}

BOND_TYPE_VOCAB = (
    BondType.SINGLE,
    BondType.DOUBLE,
    BondType.TRIPLE,
    BondType.AROMATIC,
    "other",
)
BOND_TYPE_INDEX = {value: idx for idx, value in enumerate(BOND_TYPE_VOCAB)}

BOND_STEREO_VOCAB = (
    BondStereo.STEREONONE,
    BondStereo.STEREOANY,
    BondStereo.STEREOZ,
    BondStereo.STEREOE,
    BondStereo.STEREOCIS,
    BondStereo.STEREOTRANS,
    "other",
)
BOND_STEREO_INDEX = {value: idx for idx, value in enumerate(BOND_STEREO_VOCAB)}

BASE_DESCRIPTOR_NAMES = (
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

THREED_DESCRIPTOR_NAMES = (
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

GLOBAL_FEATURE_NAMES = BASE_DESCRIPTOR_NAMES + THREED_DESCRIPTOR_NAMES
GLOBAL_FEATURE_DIM = len(GLOBAL_FEATURE_NAMES)

FINGERPRINT_RADIUS = 2
FINGERPRINT_DIM = 2048

FAST_3D_DEFAULT_CONFS = 2
FAST_3D_DEFAULT_MAX_ITERS = 75
FAST_3D_LARGE_CONFS = 1
FAST_3D_LARGE_MAX_ITERS = 40
FAST_3D_LARGE_HEAVY_ATOMS = 45
FAST_3D_LARGE_ROTATABLE_BONDS = 10
FAST_3D_LARGE_RING_COUNT = 6
FAST_3D_EMBED_ONLY_HEAVY_ATOMS = 55
FAST_3D_EMBED_ONLY_ROTATABLE_BONDS = 18
FAST_3D_EMBED_ONLY_RING_COUNT = 7


def _bucket_index(value: object, index: dict[object, int]) -> int:
    return index.get(value, index["other"])


def _one_hot(index: int, size: int) -> list[float]:
    values = [0.0] * size
    values[index] = 1.0
    return values


def atom_feature_dim() -> int:
    return (
        len(ATOM_VOCAB)
        + len(DEGREE_VOCAB)
        + len(FORMAL_CHARGE_VOCAB)
        + len(HYBRIDIZATION_VOCAB)
        + 1
        + len(HYDROGEN_COUNT_VOCAB)
        + len(CHIRALITY_VOCAB)
        + len(CIP_CODE_VOCAB)
        + 1
    )


def bond_feature_dim() -> int:
    return len(BOND_TYPE_VOCAB) + 1 + 1 + len(BOND_STEREO_VOCAB)


ATOM_FEATURE_DIM = atom_feature_dim()
BOND_FEATURE_DIM = bond_feature_dim()


def atom_features(atom: Chem.Atom) -> tuple[list[float], int]:
    atom_symbol = atom.GetSymbol()
    atom_idx = _bucket_index(atom_symbol, ATOM_INDEX)
    degree_idx = _bucket_index(atom.GetDegree(), DEGREE_INDEX)
    formal_charge_idx = _bucket_index(atom.GetFormalCharge(), FORMAL_CHARGE_INDEX)
    hybridization_idx = _bucket_index(atom.GetHybridization(), HYBRIDIZATION_INDEX)
    hydrogen_count_idx = _bucket_index(atom.GetTotalNumHs(includeNeighbors=True), HYDROGEN_COUNT_INDEX)
    chirality_idx = _bucket_index(atom.GetChiralTag(), CHIRALITY_INDEX)
    cip_code = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else "other"
    cip_code_idx = _bucket_index(cip_code, CIP_CODE_INDEX)
    chirality_possible = float(atom.HasProp("_ChiralityPossible"))

    features = (
        _one_hot(atom_idx, len(ATOM_VOCAB))
        + _one_hot(degree_idx, len(DEGREE_VOCAB))
        + _one_hot(formal_charge_idx, len(FORMAL_CHARGE_VOCAB))
        + _one_hot(hybridization_idx, len(HYBRIDIZATION_VOCAB))
        + [float(atom.GetIsAromatic())]
        + _one_hot(hydrogen_count_idx, len(HYDROGEN_COUNT_VOCAB))
        + _one_hot(chirality_idx, len(CHIRALITY_VOCAB))
        + _one_hot(cip_code_idx, len(CIP_CODE_VOCAB))
        + [chirality_possible]
    )
    return features, atom_idx


def bond_features(bond: Chem.Bond) -> list[float]:
    bond_type_idx = _bucket_index(bond.GetBondType(), BOND_TYPE_INDEX)
    stereo_idx = _bucket_index(bond.GetStereo(), BOND_STEREO_INDEX)
    return [
        *_one_hot(bond_type_idx, len(BOND_TYPE_VOCAB)),
        float(bond.IsInRing()),
        float(bond.GetIsConjugated()),
        *_one_hot(stereo_idx, len(BOND_STEREO_VOCAB)),
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


def _pattern_count(mol: Chem.Mol, smarts: str) -> float:
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        return 0.0
    return float(len(mol.GetSubstructMatches(pattern)))


def _ring_system_descriptors(mol: Chem.Mol) -> tuple[float, float, float]:
    atom_rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
    if not atom_rings:
        return 0.0, 0.0, 0.0

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

    return float(system_count), float(largest_system_atoms), float(fused_ring_count)


def _murcko_descriptors(mol: Chem.Mol) -> tuple[float, float]:
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return 0.0, 0.0
    scaffold_atoms = float(scaffold.GetNumHeavyAtoms()) if scaffold is not None else 0.0
    heavy_atoms = float(max(mol.GetNumHeavyAtoms(), 1))
    return scaffold_atoms, scaffold_atoms / heavy_atoms


def _topological_diameter(mol: Chem.Mol) -> float:
    try:
        distances = Chem.GetDistanceMatrix(mol)
    except Exception:
        return 0.0
    if distances.size == 0:
        return 0.0
    return _safe_float(np.max(distances))


def base_descriptor_dict(mol: Chem.Mol) -> dict[str, float]:
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    hetero_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6))
    ring_system_count, largest_ring_system_size, fused_ring_count = _ring_system_descriptors(mol)
    murcko_scaffold_atoms, murcko_scaffold_fraction = _murcko_descriptors(mol)
    heavy_atoms = float(max(mol.GetNumHeavyAtoms(), 1))
    aromatic_atom_fraction = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic()) / heavy_atoms
    return {
        "mw": _safe_float(Descriptors.MolWt(mol)),
        "heavy_atom_count": float(mol.GetNumHeavyAtoms()),
        "num_rotatable_bonds": _safe_float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "tpsa": _safe_float(rdMolDescriptors.CalcTPSA(mol)),
        "logp": _safe_float(Descriptors.MolLogP(mol)),
        "num_h_acceptors": _safe_float(rdMolDescriptors.CalcNumHBA(mol)),
        "num_h_donors": _safe_float(rdMolDescriptors.CalcNumHBD(mol)),
        "num_aromatic_rings": _safe_float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "fraction_csp3": _safe_float(Descriptors.FractionCSP3(mol)),
        "num_atom_stereo_centers": _safe_float(rdMolDescriptors.CalcNumAtomStereoCenters(mol)),
        "num_unspecified_atom_stereo_centers": _safe_float(
            rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)
        ),
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
        "ring_system_count": ring_system_count,
        "largest_ring_system_size": largest_ring_system_size,
        "fused_ring_count": fused_ring_count,
        "murcko_scaffold_atoms": murcko_scaffold_atoms,
        "murcko_scaffold_fraction": _safe_float(murcko_scaffold_fraction),
        "topological_diameter": _topological_diameter(mol),
        "carbonyl_count": _pattern_count(mol, "[CX3]=[OX1]"),
        "hydroxyl_count": _pattern_count(mol, "[OX2H]"),
        "carboxylic_acid_count": _pattern_count(mol, "[CX3](=O)[OX2H1]"),
        "aromatic_atom_fraction": _safe_float(aromatic_atom_fraction),
    }


def _shape_descriptor_defaults() -> dict[str, float]:
    return {name: 0.0 for name in THREED_DESCRIPTOR_NAMES}


def _shape_embedding_plan(mol: Chem.Mol) -> tuple[int, bool, int]:
    heavy_atoms = mol.GetNumHeavyAtoms()
    rotatable_bonds = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
    ring_count = int(mol.GetRingInfo().NumRings())

    if (
        heavy_atoms >= FAST_3D_EMBED_ONLY_HEAVY_ATOMS
        or rotatable_bonds >= FAST_3D_EMBED_ONLY_ROTATABLE_BONDS
        or ring_count >= FAST_3D_EMBED_ONLY_RING_COUNT
    ):
        return 0, False, 0
    if (
        heavy_atoms >= FAST_3D_LARGE_HEAVY_ATOMS
        or rotatable_bonds >= FAST_3D_LARGE_ROTATABLE_BONDS
        or ring_count >= FAST_3D_LARGE_RING_COUNT
    ):
        return FAST_3D_LARGE_CONFS, True, FAST_3D_LARGE_MAX_ITERS
    return FAST_3D_DEFAULT_CONFS, True, FAST_3D_DEFAULT_MAX_ITERS


def _embed_for_3d(mol: Chem.Mol) -> tuple[Chem.Mol | None, int | None]:
    mol_3d = Chem.AddHs(Chem.Mol(mol))
    num_confs, optimize_geometry, max_iters = _shape_embedding_plan(mol)
    if num_confs <= 0:
        return None, None
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    params.useRandomCoords = False
    params.numThreads = 1
    try:
        conf_ids = list(AllChem.EmbedMultipleConfs(mol_3d, numConfs=num_confs, params=params))
    except Exception:
        return None, None
    if not conf_ids:
        return None, None

    best_conf_id: int | None = int(conf_ids[0])
    best_energy = float("inf")
    if not optimize_geometry:
        return mol_3d, best_conf_id

    try:
        if AllChem.MMFFHasAllMoleculeParams(mol_3d):
            results = AllChem.MMFFOptimizeMoleculeConfs(mol_3d, numThreads=1, maxIters=max_iters)
        else:
            results = AllChem.UFFOptimizeMoleculeConfs(mol_3d, numThreads=1, maxIters=max_iters)
    except Exception:
        return mol_3d, best_conf_id

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

    if best_conf_id is None:
        return None, None
    return mol_3d, best_conf_id


def _shape_descriptor_dict(mol: Chem.Mol) -> dict[str, float]:
    embedded, conf_id = _embed_for_3d(mol)
    if embedded is None or conf_id is None:
        return _shape_descriptor_defaults()

    values = {
        "conformer_success": 1.0,
        "radius_of_gyration": _safe_float(Descriptors3D.RadiusOfGyration(embedded, confId=conf_id)),
        "asphericity": _safe_float(Descriptors3D.Asphericity(embedded, confId=conf_id)),
        "eccentricity": _safe_float(Descriptors3D.Eccentricity(embedded, confId=conf_id)),
        "inertial_shape_factor": _safe_float(Descriptors3D.InertialShapeFactor(embedded, confId=conf_id)),
        "npr1": _safe_float(Descriptors3D.NPR1(embedded, confId=conf_id)),
        "npr2": _safe_float(Descriptors3D.NPR2(embedded, confId=conf_id)),
        "pbf": _safe_float(Descriptors3D.PBF(embedded, confId=conf_id)),
        "pmi1": _safe_float(Descriptors3D.PMI1(embedded, confId=conf_id)),
        "pmi2": _safe_float(Descriptors3D.PMI2(embedded, confId=conf_id)),
        "pmi3": _safe_float(Descriptors3D.PMI3(embedded, confId=conf_id)),
        "spherocity_index": _safe_float(Descriptors3D.SpherocityIndex(embedded, confId=conf_id)),
    }
    return values


def global_feature_dict(
    smiles: str,
    *,
    descriptor_overrides: Mapping[str, object] | None = None,
) -> dict[str, float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse canonical SMILES: {smiles}")
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    base_values = base_descriptor_dict(mol)
    if descriptor_overrides:
        for name in BASE_DESCRIPTOR_NAMES:
            if name in descriptor_overrides:
                base_values[name] = _safe_float(descriptor_overrides[name], default=base_values[name])

    shape_values = _shape_descriptor_dict(mol)
    return {**base_values, **shape_values}


def global_feature_vector(
    smiles: str,
    *,
    descriptor_overrides: Mapping[str, object] | None = None,
) -> list[float]:
    feature_dict = global_feature_dict(smiles, descriptor_overrides=descriptor_overrides)
    return [_safe_float(feature_dict[name]) for name in GLOBAL_FEATURE_NAMES]


def fingerprint_vector(smiles: str) -> list[float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse canonical SMILES: {smiles}")
    arr = np.zeros((FINGERPRINT_DIM,), dtype=np.uint8)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=FINGERPRINT_RADIUS, fpSize=FINGERPRINT_DIM)
    DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), arr)
    return arr.astype(np.float32, copy=False).tolist()


@lru_cache(maxsize=100_000)
def cached_global_feature_vector(smiles: str) -> tuple[float, ...]:
    return tuple(global_feature_vector(smiles))


@lru_cache(maxsize=100_000)
def cached_fingerprint_vector(smiles: str) -> tuple[float, ...]:
    return tuple(fingerprint_vector(smiles))
