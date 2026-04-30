"""Uncertainty calibration.

Chemprop's raw sigma is often miscalibrated — under- or over-confident relative
to actual residual distributions. We correct it with a per-model scalar and
optional isotonic curve fitted on a held-out split and stored on disk at
`<model_dir>/calibration.json`.

This module provides:
- `Calibration` — an immutable container with `calibrate_sigma()`.
- `load_calibration()` — reads the JSON file produced by `fit_calibration()`.
- `fit_calibration()` — estimates scale (+ optional isotonic) from residuals
  vs raw sigma. Meant to be run offline by a training script, not at request
  time.

File format (calibration.json):
    {
        "method": "scale" | "isotonic",
        "scale": 1.83,                      # multiplier on raw sigma
        "knots_sigma": [...], "knots_cal": [...],  # present if isotonic
        "n_samples": 420,
        "fit_date": "2026-01-15"
    }
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Calibration:
    method: str
    scale: float
    knots_sigma: tuple[float, ...] = ()
    knots_cal: tuple[float, ...] = ()
    n_samples: int = 0

    def calibrate_sigma(self, sigma: float) -> float:
        if self.method == "isotonic" and self.knots_sigma:
            return _interp(sigma, self.knots_sigma, self.knots_cal)
        return sigma * self.scale


def _interp(x: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = ys[i - 1], ys[i]
            if x1 == x0:
                return float(y1)
            return float(y0 + (y1 - y0) * (x - x0) / (x1 - x0))
    return float(ys[-1])


def load_calibration(path: Path) -> Calibration:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return Calibration(
        method=str(data.get("method", "scale")),
        scale=float(data.get("scale", 1.0)),
        knots_sigma=tuple(float(x) for x in data.get("knots_sigma", [])),
        knots_cal=tuple(float(x) for x in data.get("knots_cal", [])),
        n_samples=int(data.get("n_samples", 0)),
    )


def fit_calibration(
    residuals: Sequence[float],
    raw_sigmas: Sequence[float],
    *,
    method: str = "scale",
    n_bins: int = 10,
) -> Calibration:
    """Fit a calibration from validation residuals.

    `method="scale"` picks the single multiplier `s` that makes the ratio
    stddev(residual) / stddev(raw_sigma * s) equal to 1 — i.e. it matches the
    predicted sigma spread to the observed residual spread.

    `method="isotonic"` bins samples by raw sigma, fits a monotone step
    function of (raw_sigma -> empirical residual stddev per bin), and stores
    the knots for linear interpolation at inference time.
    """
    if len(residuals) != len(raw_sigmas):
        raise ValueError("residuals and raw_sigmas length mismatch")
    if not residuals:
        raise ValueError("no samples to fit calibration")

    abs_res = [abs(r) for r in residuals]
    if method == "scale":
        mean_abs_res = sum(abs_res) / len(abs_res)
        mean_sigma = sum(raw_sigmas) / len(raw_sigmas)
        scale = (mean_abs_res / mean_sigma) if mean_sigma > 0 else 1.0
        return Calibration(method="scale", scale=scale, n_samples=len(residuals))

    if method == "isotonic":
        pairs = sorted(zip(raw_sigmas, abs_res, strict=True), key=lambda p: p[0])
        n = len(pairs)
        bin_size = max(1, n // n_bins)
        knots_sigma: list[float] = []
        knots_cal: list[float] = []
        i = 0
        while i < n:
            chunk = pairs[i : i + bin_size]
            sig_mean = sum(s for s, _ in chunk) / len(chunk)
            res_mean = sum(r for _, r in chunk) / len(chunk)
            knots_sigma.append(sig_mean)
            knots_cal.append(res_mean)
            i += bin_size
        # enforce monotonicity (pool adjacent violators, simplified)
        for j in range(1, len(knots_cal)):
            if knots_cal[j] < knots_cal[j - 1]:
                knots_cal[j] = knots_cal[j - 1]
        scale = knots_cal[-1] / knots_sigma[-1] if knots_sigma[-1] > 0 else 1.0
        return Calibration(
            method="isotonic",
            scale=scale,
            knots_sigma=tuple(knots_sigma),
            knots_cal=tuple(knots_cal),
            n_samples=len(residuals),
        )

    raise ValueError(f"unknown calibration method: {method!r}")


def save_calibration(cal: Calibration, path: Path) -> None:
    payload = {
        "method": cal.method,
        "scale": cal.scale,
        "knots_sigma": list(cal.knots_sigma),
        "knots_cal": list(cal.knots_cal),
        "n_samples": cal.n_samples,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
