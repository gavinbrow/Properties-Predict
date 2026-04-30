"""ORM models for the lookup DB.

`Compound` is keyed by InChIKey (unique). `PropertyRecord` holds a value per
(compound, property, source) tuple; `source_priority` (lower = better) is
used to pick the preferred record on lookup.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from server.lookup.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Compound(Base):
    __tablename__ = "compounds"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    inchikey: Mapped[str] = mapped_column(String(27), unique=True, index=True, nullable=False)
    canonical_smiles: Mapped[str] = mapped_column(String(500), index=True, nullable=False)
    formula: Mapped[str | None] = mapped_column(String(120), nullable=True)
    mw: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    records: Mapped[list[PropertyRecord]] = relationship(
        back_populates="compound",
        cascade="all, delete-orphan",
    )


class PropertyRecord(Base):
    __tablename__ = "property_records"
    __table_args__ = (
        UniqueConstraint("compound_id", "property", "source", name="uq_record_per_source"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    compound_id: Mapped[int] = mapped_column(
        ForeignKey("compounds.id", ondelete="CASCADE"), index=True, nullable=False
    )
    property: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(String(16), nullable=False)
    condition: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_priority: Mapped[int] = mapped_column(Integer, default=100, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    compound: Mapped[Compound] = relationship(back_populates="records")


class IngestAudit(Base):
    """One row per ingest run; supports traceable provenance."""

    __tablename__ = "ingest_audit"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    source_file: Mapped[str] = mapped_column(Text, nullable=False)
    source_name: Mapped[str] = mapped_column(String(64), nullable=False)
    rows_read: Mapped[int] = mapped_column(Integer, default=0)
    rows_inserted: Mapped[int] = mapped_column(Integer, default=0)
    rows_updated: Mapped[int] = mapped_column(Integer, default=0)
    rows_skipped: Mapped[int] = mapped_column(Integer, default=0)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
