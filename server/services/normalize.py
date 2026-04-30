"""RDKit-based SMILES normalization.

Pipeline: parse -> sanitize -> uncharge -> strip isotopes -> pick parent fragment
-> canonical SMILES -> InChIKey -> descriptors.
"""
from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from server.schemas.chemistry import MoleculeDescriptors, NormalizedMolecule

RDLogger.DisableLog("rdApp.*")


class NormalizationError(ValueError):
    """Raised when a SMILES cannot be parsed or sanitized."""


@dataclass(frozen=True, slots=True)
class _Standardizer:
    uncharger: rdMolStandardize.Uncharger
    cleanup_params: rdMolStandardize.CleanupParameters


def _build_standardizer() -> _Standardizer:
    params = rdMolStandardize.CleanupParameters()
    return _Standardizer(
        uncharger=rdMolStandardize.Uncharger(),
        cleanup_params=params,
    )


_STD = _build_standardizer()


def _parse(smiles: str) -> Chem.Mol:
    if not isinstance(smiles, str) or not smiles.strip():
        raise NormalizationError("empty SMILES")
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise NormalizationError(f"unparseable SMILES: {smiles!r}")
    try:
        Chem.SanitizeMol(mol)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError) as e:
        raise NormalizationError(f"sanitization failed: {e}") from e
    return mol


def _strip_isotopes(mol: Chem.Mol) -> Chem.Mol:
    for atom in mol.GetAtoms():
        if atom.GetIsotope() != 0:
            atom.SetIsotope(0)
    return mol


def _standardize(mol: Chem.Mol) -> Chem.Mol:
    try:
        mol = rdMolStandardize.Cleanup(mol, _STD.cleanup_params)
        parent = rdMolStandardize.FragmentParent(mol, _STD.cleanup_params)
        neutral = _STD.uncharger.uncharge(parent)
        _strip_isotopes(neutral)
        Chem.SanitizeMol(neutral)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError) as e:
        raise NormalizationError(f"standardization failed: {e}") from e
    return neutral


def _compute_descriptors(mol: Chem.Mol) -> MoleculeDescriptors:
    return MoleculeDescriptors(
        mw=float(Descriptors.MolWt(mol)),
        heavy_atom_count=int(mol.GetNumHeavyAtoms()),
        num_rotatable_bonds=int(Lipinski.NumRotatableBonds(mol)),
        tpsa=float(Descriptors.TPSA(mol)),
        logp=float(Descriptors.MolLogP(mol)),
        num_h_acceptors=int(Lipinski.NumHAcceptors(mol)),
        num_h_donors=int(Lipinski.NumHDonors(mol)),
        num_aromatic_rings=int(Lipinski.NumAromaticRings(mol)),
    )


def normalize(smiles: str) -> NormalizedMolecule:
    """Normalize a SMILES string into a canonical form with descriptors.

    Raises NormalizationError for unparseable or un-sanitizable input.
    """
    raw = _parse(smiles)
    std = _standardize(raw)
    canonical = Chem.MolToSmiles(std, canonical=True)
    inchi = Chem.MolToInchi(std)
    if not inchi:
        raise NormalizationError("InChI generation failed")
    inchikey = Chem.InchiToInchiKey(inchi)
    formula = rdMolDescriptors.CalcMolFormula(std)
    return NormalizedMolecule(
        input_smiles=smiles,
        canonical_smiles=canonical,
        inchikey=inchikey,
        formula=formula,
        descriptors=_compute_descriptors(std),
    )


def smiles_to_mol(smiles: str) -> Chem.Mol:
    """Parse + standardize a SMILES to a Mol (no SMILES/InChIKey export).

    Useful for engines that want a ready-to-use Mol.
    """
    mol = _standardize(_parse(smiles))
    AllChem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol
