"""Lookup service — exact match by InChIKey with source-priority selection."""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select

from server.core.constants import PROPERTY_UNITS, SUPPORTED_PROPERTIES
from server.lookup.db import session_scope
from server.lookup.models import Compound, PropertyRecord


@dataclass(frozen=True, slots=True)
class LookupHit:
    property: str
    value: float
    unit: str
    condition: str | None
    source: str
    source_priority: int
    inchikey: str
    canonical_smiles: str
    method: str = "lookup"


def lookup(inchikey: str, prop: str) -> LookupHit | None:
    """Return the preferred (lowest source_priority) record for (inchikey, prop).

    None if no record. Raises ValueError for unknown property.
    """
    if prop not in SUPPORTED_PROPERTIES:
        raise ValueError(f"unsupported property: {prop}")

    stmt = (
        select(PropertyRecord, Compound)
        .join(Compound, PropertyRecord.compound_id == Compound.id)
        .where(Compound.inchikey == inchikey, PropertyRecord.property == prop)
        .order_by(PropertyRecord.source_priority.asc(), PropertyRecord.created_at.asc())
        .limit(1)
    )
    with session_scope() as s:
        row = s.execute(stmt).first()
        if row is None:
            return None
        rec, cmp = row
        return LookupHit(
            property=rec.property,
            value=rec.value,
            unit=rec.unit,
            condition=rec.condition,
            source=rec.source,
            source_priority=rec.source_priority,
            inchikey=cmp.inchikey,
            canonical_smiles=cmp.canonical_smiles,
        )


def lookup_all(inchikey: str) -> dict[str, LookupHit]:
    """Return preferred record per property for the given compound."""
    out: dict[str, LookupHit] = {}
    for prop in SUPPORTED_PROPERTIES:
        hit = lookup(inchikey, prop)
        if hit is not None:
            out[prop] = hit
    return out


def expected_unit(prop: str) -> str:
    return PROPERTY_UNITS[prop]
