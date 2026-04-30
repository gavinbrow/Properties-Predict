"""Project-wide constants. Values locked by docs/scope.md."""
from __future__ import annotations

SUPPORTED_ATOMS: frozenset[str] = frozenset(
    {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}
)

SUPPORTED_PROPERTIES: frozenset[str] = frozenset({"mp", "bp", "density"})

PROPERTY_UNITS: dict[str, str] = {
    "mp": "degC",
    "bp": "degC",
    "density": "g/mL",
}

PROPERTY_CONDITIONS: dict[str, str] = {
    "mp": "atmospheric",
    "bp": "1 atm",
    "density": "liquid, 25 degC, 1 atm",
}

MW_MIN_DA: float = 10.0
MW_MAX_DA: float = 1000.0
