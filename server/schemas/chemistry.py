"""Schemas for the chemistry preprocessing layer."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MoleculeDescriptors(BaseModel):
    model_config = ConfigDict(frozen=True)

    mw: float
    heavy_atom_count: int
    num_rotatable_bonds: int
    tpsa: float
    logp: float
    num_h_acceptors: int
    num_h_donors: int
    num_aromatic_rings: int


class NormalizedMolecule(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_smiles: str
    canonical_smiles: str
    inchikey: str
    formula: str
    descriptors: MoleculeDescriptors


class DomainRejection(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str = Field(description="Machine-readable reason code.")
    message: str = Field(description="Human-readable reason.")


class DomainCheckResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    in_domain: bool
    rejections: tuple[DomainRejection, ...] = ()
