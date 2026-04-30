"""Offline density estimator built from thermo/chemicals correlations.

This adapter is a pragmatic fallback for liquid density at 25 C, 1 atm when
Chemprop model artifacts are not present. It uses Joback group contributions to
estimate critical properties from the molecule, then applies CSP liquid-density
correlations (COSTALD and Rackett). The result is conservative by design:
warnings are surfaced when fragmentation is partial, the molecule is likely not
liquid at 25 C, or the two correlations disagree materially.
"""
from __future__ import annotations

import importlib.metadata as im
import math
import time

from server.core.constants import PROPERTY_UNITS
from server.engines.base import BaseEngine
from server.schemas.chemistry import NormalizedMolecule
from server.schemas.engines import EngineResult, EngineStatus, error_result

ROOM_TEMPERATURE_K = 298.15
LOW_CONF_SIGMA_G_ML = 0.08
MIN_REASONABLE_DENSITY = 0.30
MAX_REASONABLE_DENSITY = 2.20
R_GAS_CONSTANT = 8.314462618


class ThermoDensityEngine(BaseEngine):
    """Density fallback using Joback + CSP liquid-density equations."""

    name = "thermo"

    def __init__(self) -> None:
        self.version = self._discover_version()

    def _discover_version(self) -> str:
        versions: list[str] = []
        for package in ("thermo", "chemicals"):
            try:
                versions.append(f"{package}-{im.version(package)}")
            except im.PackageNotFoundError:
                continue
        return "::".join(versions) if versions else "not-installed"

    def supports(self, prop: str) -> bool:
        return prop == "density"

    def status(self) -> dict:
        ready = self.version != "not-installed"
        issues = ()
        if not ready:
            issues = ("thermo and chemicals packages are not installed in the active Python environment",)
        return {
            "ready": ready,
            "ready_for": ["density"],
            "issues": issues,
        }

    def predict(self, molecule: NormalizedMolecule, prop: str) -> EngineResult:
        if not self.supports(prop):
            return error_result(
                engine=self.name,
                engine_version=self.version,
                prop=prop,
                status=EngineStatus.UNSUPPORTED_PROPERTY,
                message=f"Thermo adapter does not cover {prop!r}",
            )
        if self.version == "not-installed":
            return error_result(
                engine=self.name,
                engine_version=self.version,
                prop=prop,
                status=EngineStatus.NOT_CONFIGURED,
                message="thermo and chemicals packages are not installed in the active Python environment",
            )

        t0 = time.monotonic()
        try:
            estimate = self._estimate_density(molecule.canonical_smiles)
        except ValueError as e:
            rt = int((time.monotonic() - t0) * 1000)
            return error_result(
                engine=self.name,
                engine_version=self.version,
                prop=prop,
                status=EngineStatus.OUT_OF_DOMAIN,
                message=str(e),
                runtime_ms=rt,
            )
        except Exception as e:  # pragma: no cover - depends on optional runtime
            rt = int((time.monotonic() - t0) * 1000)
            return error_result(
                engine=self.name,
                engine_version=self.version,
                prop=prop,
                status=EngineStatus.ENGINE_ERROR,
                message=f"thermo density calculation failed: {e}",
                runtime_ms=rt,
            )

        runtime_ms = int((time.monotonic() - t0) * 1000)
        return EngineResult(
            engine=self.name,
            engine_version=self.version,
            property=prop,
            status=estimate["status"],
            value=estimate["value"],
            unit=PROPERTY_UNITS[prop],
            uncertainty=estimate["uncertainty"],
            in_domain=estimate["in_domain"],
            runtime_ms=runtime_ms,
            warnings=estimate["warnings"],
            raw=estimate["raw"],
        )

    def _estimate_density(self, smiles: str) -> dict:
        from chemicals.acentric import LK_omega
        from chemicals.utils import Vm_to_rho
        from chemicals.volume import COSTALD, Rackett
        from thermo import Joback

        joback = Joback(smiles)
        estimates = joback.estimate()

        required = ("Tb", "Tc", "Pc", "Vc")
        missing = [name for name in required if estimates.get(name) in (None, 0)]
        if missing:
            raise ValueError(
                "thermo density fallback could not estimate "
                + ", ".join(missing)
                + f" from Joback groups for {smiles}"
            )

        tb = float(estimates["Tb"])
        tc = float(estimates["Tc"])
        pc = float(estimates["Pc"])
        vc = float(estimates["Vc"])
        if not all(math.isfinite(x) and x > 0 for x in (tb, tc, pc, vc)):
            raise ValueError(f"thermo density fallback produced invalid critical properties for {smiles}")

        warnings: list[str] = []
        status = EngineStatus.OK
        if not joback.success or str(joback.status).upper() != "OK":
            status = EngineStatus.LOW_CONFIDENCE
            warnings.append(f"Joback fragmentation status: {joback.status}")

        if tb <= ROOM_TEMPERATURE_K:
            status = EngineStatus.LOW_CONFIDENCE
            warnings.append("estimated normal boiling point is below 25 C; liquid density at 1 atm may be out of scope")

        tr = ROOM_TEMPERATURE_K / tc
        if not 0.25 <= tr <= 0.95:
            status = EngineStatus.LOW_CONFIDENCE
            warnings.append(f"reduced temperature {tr:.2f} is outside COSTALD's preferred range")

        methods: dict[str, float] = {}
        omega = LK_omega(tb, tc, pc)
        if math.isfinite(omega):
            vm_costald = COSTALD(ROOM_TEMPERATURE_K, tc, vc, omega)
            methods["COSTALD"] = Vm_to_rho(vm_costald, joback.MW) / 1000.0

        zc = pc * vc / (R_GAS_CONSTANT * tc)
        if math.isfinite(zc) and zc > 0:
            vm_rackett = Rackett(ROOM_TEMPERATURE_K, tc, pc, zc)
            methods["Rackett"] = Vm_to_rho(vm_rackett, joback.MW) / 1000.0

        if not methods:
            raise ValueError(f"thermo density fallback could not compute a liquid-density correlation for {smiles}")

        value = methods.get("COSTALD")
        if value is None:
            value = next(iter(methods.values()))

        sigma = None
        if len(methods) >= 2:
            vals = list(methods.values())
            mean = sum(vals) / len(vals)
            sigma = (sum((x - mean) ** 2 for x in vals) / max(1, len(vals) - 1)) ** 0.5
            if sigma > LOW_CONF_SIGMA_G_ML:
                status = EngineStatus.LOW_CONFIDENCE
                warnings.append(
                    f"density correlations disagree (sigma {sigma:.3f} g/mL exceeds {LOW_CONF_SIGMA_G_ML:.2f})"
                )

        if not MIN_REASONABLE_DENSITY <= value <= MAX_REASONABLE_DENSITY:
            status = EngineStatus.LOW_CONFIDENCE
            warnings.append(
                f"estimated density {value:.3f} g/mL falls outside the typical organic range"
            )

        return {
            "status": status,
            "value": value,
            "uncertainty": sigma,
            "in_domain": bool(joback.success),
            "warnings": tuple(dict.fromkeys(warnings)),
            "raw": {
                "joback_status": str(joback.status),
                "Tb_K": tb,
                "Tc_K": tc,
                "Pc_Pa": pc,
                "Vc_m3_mol": vc,
                "Tr": tr,
                "omega": omega,
                "methods_g_mL": methods,
            },
        }
