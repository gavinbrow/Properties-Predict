"""Domain filters for v1 chemistry scope.

A molecule passes only if it satisfies every filter. Rejection reasons are
returned so the API layer can surface them in the `unsupported` response.
"""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors

from server.core.constants import MW_MAX_DA, MW_MIN_DA, SUPPORTED_ATOMS
from server.schemas.chemistry import DomainCheckResult, DomainRejection
from server.services.normalize import smiles_to_mol


def _check_fragments(mol: Chem.Mol) -> DomainRejection | None:
    frags = Chem.GetMolFrags(mol, asMols=False)
    if len(frags) > 1:
        return DomainRejection(
            code="disconnected_fragments",
            message="multi-component structure after standardization",
        )
    return None


def _check_atoms(mol: Chem.Mol) -> DomainRejection | None:
    unsupported = sorted(
        {a.GetSymbol() for a in mol.GetAtoms()} - SUPPORTED_ATOMS
    )
    if unsupported:
        return DomainRejection(
            code="unsupported_atoms",
            message=f"atoms outside v1 scope: {', '.join(unsupported)}",
        )
    return None


def _check_mw(mol: Chem.Mol) -> DomainRejection | None:
    mw = Descriptors.MolWt(mol)
    if mw < MW_MIN_DA or mw > MW_MAX_DA:
        return DomainRejection(
            code="mw_out_of_range",
            message=f"MW {mw:.2f} outside [{MW_MIN_DA}, {MW_MAX_DA}] Da",
        )
    return None


def _check_radicals(mol: Chem.Mol) -> DomainRejection | None:
    if any(a.GetNumRadicalElectrons() > 0 for a in mol.GetAtoms()):
        return DomainRejection(
            code="radical",
            message="open-shell species after sanitization",
        )
    return None


_CHECKS = (_check_fragments, _check_atoms, _check_mw, _check_radicals)


def check_mol(mol: Chem.Mol) -> DomainCheckResult:
    """Run every domain filter against a standardized Mol."""
    rejections = tuple(r for r in (c(mol) for c in _CHECKS) if r is not None)
    return DomainCheckResult(in_domain=not rejections, rejections=rejections)


def check_smiles(smiles: str) -> DomainCheckResult:
    """Normalize then run domain filters. Parse failures short-circuit."""
    try:
        mol = smiles_to_mol(smiles)
    except ValueError as e:
        return DomainCheckResult(
            in_domain=False,
            rejections=(DomainRejection(code="parse_error", message=str(e)),),
        )
    return check_mol(mol)
