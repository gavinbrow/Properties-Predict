"""Production Flask entrypoint for Properties Predict.

Run with:
    python app.py
"""
from __future__ import annotations

import importlib.metadata as im
import json
import logging
import logging.handlers
import os
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request, send_from_directory
from pydantic import ValidationError
from waitress import serve
from werkzeug.exceptions import HTTPException

PROD_ROOT = Path(__file__).resolve().parent
FRONTEND_DIST = PROD_ROOT / "frontend" / "dist"
LOG_DIR = PROD_ROOT / "logs"

# Make the bundled `server` package importable regardless of cwd.
if str(PROD_ROOT) not in sys.path:
    sys.path.insert(0, str(PROD_ROOT))

from server.core.config import get_settings  # noqa: E402
from server.core.constants import PROPERTY_CONDITIONS, PROPERTY_UNITS, SUPPORTED_PROPERTIES  # noqa: E402
from server.engines.gnn import GnnEngine  # noqa: E402
from server.engines.knn import KnnEngine  # noqa: E402
from server.engines.thermo import ThermoDensityEngine  # noqa: E402
from server.schemas.predict import BatchPredictionRequest, PredictionRequest  # noqa: E402
from server.services.normalize import normalize  # noqa: E402
from server.services.orchestrator import EngineRegistry, predict_for_smiles  # noqa: E402

ACTIVE_ENGINES = ("knn", "gnn", "thermo")
PROD_ROUTING: dict[str, tuple[str, ...]] = {
    "mp": ("knn", "gnn"),
    "bp": ("knn", "gnn"),
    "density": ("thermo",),
}
REQUIRED_READY_FOR: dict[str, tuple[str, ...]] = {
    "knn": ("mp", "bp"),
    "gnn": ("mp", "bp"),
    "thermo": ("density",),
}

logger = logging.getLogger("properties_predict.prod")
audit_logger = logging.getLogger("properties_predict.audit")
audit_logger.propagate = False  # keep audit JSONL out of the console stream


class ApiError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def create_app() -> Flask:
    # static_folder=None: we serve frontend assets ourselves via the routes
    # below. Flask's auto static handler would otherwise intercept every URL
    # and 404 on unknown asset paths instead of falling back to index.html.
    app = Flask(__name__, static_folder=None)
    _configure_server_log_file(os.getenv("LOG_LEVEL", get_settings().log_level))
    _configure_audit_logger()

    @app.after_request
    def add_security_headers(response: Response) -> Response:
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.errorhandler(ApiError)
    def handle_api_error(exc: ApiError):
        return jsonify({"error": exc.message}), exc.status_code

    @app.errorhandler(ValidationError)
    def handle_validation_error(exc: ValidationError):
        return jsonify({"error": "invalid request", "details": exc.errors()}), 422

    @app.errorhandler(HTTPException)
    def handle_http_exception(exc: HTTPException):
        # Standard HTTP errors (404, 405, etc.) — let Flask render its
        # normal response and don't log a traceback as if it were a bug.
        return exc

    @app.errorhandler(Exception)
    def handle_unexpected(exc: Exception):
        logger.exception("Unhandled prod API error")
        return jsonify({"error": "internal server error"}), 500

    @app.post("/predict")
    def predict():
        t0 = time.monotonic()
        payload = _request_json()
        req = PredictionRequest.model_validate(payload)
        _enforce_smiles(req.smiles)
        engines = _sanitize_engines(req.engines)
        response = predict_for_smiles(
            req.smiles,
            req.properties,
            registry=get_registry(),
            engines=engines,
        )
        body = response.model_dump(mode="json")
        _audit_prediction(
            endpoint="/predict",
            smiles=req.smiles,
            requested_engines=engines,
            requested_properties=req.properties,
            response=body,
            runtime_ms=int((time.monotonic() - t0) * 1000),
        )
        return jsonify(body)

    @app.post("/predict/batch")
    def predict_batch():
        t0 = time.monotonic()
        payload = _request_json()
        req = BatchPredictionRequest.model_validate(payload)
        _enforce_batch_size(len(req.items))
        results = []
        registry = get_registry()
        for item in req.items:
            _enforce_smiles(item.smiles)
            engines = _sanitize_engines(item.engines)
            response = predict_for_smiles(
                item.smiles,
                item.properties,
                registry=registry,
                engines=engines,
            )
            body = response.model_dump(mode="json")
            results.append(body)
            _audit_prediction(
                endpoint="/predict/batch",
                smiles=item.smiles,
                requested_engines=engines,
                requested_properties=item.properties,
                response=body,
                runtime_ms=int((time.monotonic() - t0) * 1000),
            )
        return jsonify({"results": results})

    @app.get("/properties")
    def list_properties():
        return jsonify(
            {
                "properties": sorted(SUPPORTED_PROPERTIES),
                "units": PROPERTY_UNITS,
                "conditions": PROPERTY_CONDITIONS,
            }
        )

    @app.get("/engines")
    def list_engines():
        registry = get_registry()
        return jsonify(
            {
                "engines": [
                    _engine_payload(name, registry.by_name[name])
                    for name in ACTIVE_ENGINES
                    if name in registry.by_name
                ],
                "routing": {key: list(value) for key, value in PROD_ROUTING.items()},
            }
        )

    @app.get("/version")
    def version():
        registry = get_registry()
        return jsonify(
            {
                "backend": "0.1.0",
                "flask": _pkg("flask"),
                "waitress": _pkg("waitress"),
                "pydantic": _pkg("pydantic"),
                "rdkit": _pkg("rdkit"),
                "knn": _engine_version(registry, "knn"),
                "knn_artifact_digest": _artifact_digest(registry, "knn"),
                "gnn": _engine_version(registry, "gnn"),
                "gnn_checkpoint_digest": _artifact_digest(registry, "gnn"),
                "gnn_dataset_manifest_digest": getattr(
                    registry.get("gnn"), "dataset_manifest_digest", lambda: None
                )(),
                "thermo": _engine_version(registry, "thermo"),
            }
        )

    @app.get("/health")
    def health():
        return jsonify(_health_payload(get_registry()))

    @app.get("/")
    def index():
        return _serve_frontend("index.html")

    @app.get("/<path:path>")
    def frontend(path: str):
        candidate = FRONTEND_DIST / path
        if candidate.is_file():
            return send_from_directory(FRONTEND_DIST, path)
        return _serve_frontend("index.html")

    return app


@lru_cache(maxsize=1)
def get_registry() -> EngineRegistry:
    return EngineRegistry(
        by_name={
            "knn": KnnEngine(),
            "gnn": GnnEngine(),
            "thermo": ThermoDensityEngine(),
        }
    )


def _request_json() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ApiError(400, "request body must be a JSON object")
    return payload


def _sanitize_engines(engines: tuple[str, ...] | None) -> tuple[str, ...] | None:
    if engines is None:
        return None
    cleaned = tuple(dict.fromkeys(name for name in engines if name in ACTIVE_ENGINES))
    return cleaned or None


def _enforce_smiles(smiles: str) -> None:
    settings = get_settings()
    if not isinstance(smiles, str) or not smiles.strip():
        raise ApiError(422, "smiles must be a non-empty string")
    if len(smiles) > settings.max_smiles_length:
        raise ApiError(413, f"smiles exceeds max length {settings.max_smiles_length}")


def _enforce_batch_size(size: int) -> None:
    settings = get_settings()
    if size == 0:
        raise ApiError(422, "batch must contain at least one item")
    if size > settings.max_batch_size:
        raise ApiError(413, f"batch size {size} exceeds max {settings.max_batch_size}")


def _engine_payload(name: str, engine: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": name,
        "version": getattr(engine, "version", "unknown"),
        "supports": [prop for prop in sorted(SUPPORTED_PROPERTIES) if engine.supports(prop)],
    }
    status_fn = getattr(engine, "status", None)
    if callable(status_fn):
        payload.update(status_fn())
    return payload


def _health_payload(registry: EngineRegistry) -> dict[str, Any]:
    engines: dict[str, dict[str, Any]] = {}
    for name in ACTIVE_ENGINES:
        engine = registry.get(name)
        if engine is None:
            engines[name] = {
                "ready": False,
                "ready_for": [],
                "issues": [f"{name} engine is not registered"],
            }
            continue

        status_fn = getattr(engine, "status", None)
        details = status_fn() if callable(status_fn) else {"ready": True, "ready_for": []}
        ready_for = set(details.get("ready_for", []))
        required = set(REQUIRED_READY_FOR[name])
        issues = list(details.get("issues", ()))
        missing = sorted(required - ready_for)
        ready = bool(details.get("ready", True)) and not missing
        if missing:
            issues.append(f"missing ready properties: {', '.join(missing)}")
        engines[name] = {
            "ready": ready,
            "ready_for": sorted(ready_for),
            "required_for": sorted(required),
            "issues": issues,
        }

    degraded = not all(item["ready"] for item in engines.values())
    return {"status": "ok", "degraded": degraded, "engines": engines}


def _assert_ready(registry: EngineRegistry) -> None:
    health = _health_payload(registry)
    if not health["degraded"]:
        return
    problems = []
    for name, info in health["engines"].items():
        if not info["ready"]:
            issues = "; ".join(info.get("issues") or ["not ready"])
            problems.append(f"{name}: {issues}")
    raise RuntimeError("production readiness check failed: " + " | ".join(problems))


def _warm_models(registry: EngineRegistry) -> None:
    molecule = normalize("CCO")
    for engine_name, props in (("knn", ("mp", "bp")), ("gnn", ("mp", "bp")), ("thermo", ("density",))):
        engine = registry.get(engine_name)
        if engine is None:
            continue
        for prop in props:
            result = engine.predict(molecule, prop)
            if getattr(result.status, "value", str(result.status)) in {
                "engine_error", "not_configured", "parse_error", "timeout",
            }:
                raise RuntimeError(f"startup inference failed for {engine_name}/{prop}: {result.warnings}")
            logger.info(
                "warmed engine=%s prop=%s status=%s",
                engine_name,
                prop,
                getattr(result.status, "value", str(result.status)),
            )


def _serve_frontend(filename: str):
    if not (FRONTEND_DIST / filename).is_file():
        raise ApiError(503, f"frontend build not found at {FRONTEND_DIST}")
    return send_from_directory(FRONTEND_DIST, filename)


def _engine_version(registry: EngineRegistry, name: str) -> str:
    engine = registry.get(name)
    return getattr(engine, "version", "not-loaded") if engine else "not-loaded"


def _artifact_digest(registry: EngineRegistry, name: str) -> str | None:
    engine = registry.get(name)
    if engine is None:
        return None
    digest_fn = getattr(engine, "artifact_digest", None)
    return digest_fn() if callable(digest_fn) else None


def _client_ip() -> str:
    # Honor X-Forwarded-For when present (only the left-most entry is the
    # original client; the rest are proxy hops). Fall back to remote_addr.
    fwd = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    return fwd or (request.remote_addr or "unknown")


def _audit_prediction(
    *,
    endpoint: str,
    smiles: str,
    requested_engines: tuple[str, ...] | None,
    requested_properties: tuple[str, ...] | None,
    response: dict[str, Any],
    runtime_ms: int,
) -> None:
    try:
        compound = response.get("compound") or {}
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "ip": _client_ip(),
            "user_agent": request.headers.get("User-Agent", ""),
            "endpoint": endpoint,
            "runtime_ms": runtime_ms,
            "smiles_input": smiles,
            "smiles_canonical": compound.get("canonical_smiles"),
            "inchikey": compound.get("inchikey"),
            "formula": compound.get("formula"),
            "mw": compound.get("mw"),
            "requested_engines": list(requested_engines) if requested_engines else None,
            "requested_properties": list(requested_properties) if requested_properties else None,
            "supported": response.get("supported"),
            "rejections": response.get("rejections") or [],
            "predictions": [
                {
                    "property": pred.get("property"),
                    "value": pred.get("value"),
                    "unit": pred.get("unit"),
                    "uncertainty": pred.get("uncertainty"),
                    "confidence": pred.get("confidence"),
                    "method": pred.get("method"),
                    "selected_engine": pred.get("selected_engine"),
                    "source": pred.get("source"),
                    "warnings": pred.get("warnings") or [],
                    "engines": [
                        {
                            "engine": d.get("engine"),
                            "engine_version": d.get("engine_version"),
                            "status": d.get("status"),
                            "value": d.get("value"),
                            "unit": d.get("unit"),
                            "uncertainty": d.get("uncertainty"),
                            "in_domain": d.get("in_domain"),
                            "runtime_ms": d.get("runtime_ms"),
                        }
                        for d in (pred.get("engine_details") or [])
                    ],
                }
                for pred in response.get("predictions") or []
            ],
        }
        audit_logger.info(json.dumps(record, ensure_ascii=False, default=str))
    except Exception:
        # Audit must never break a real request.
        logger.exception("failed to write prediction audit record")


def _pkg(name: str) -> str:
    try:
        return im.version(name)
    except im.PackageNotFoundError:
        return "not-installed"


def _configure_audit_logger() -> None:
    if any(getattr(h, "_pp_audit", False) for h in audit_logger.handlers):
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # backupCount=0 -> rotate daily but never delete old files (kept forever).
    handler = logging.handlers.TimedRotatingFileHandler(
        LOG_DIR / "predictions.jsonl",
        when="midnight",
        backupCount=0,
        encoding="utf-8",
        utc=True,
    )
    handler.suffix = "%Y-%m-%d"
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler._pp_audit = True  # type: ignore[attr-defined]
    audit_logger.addHandler(handler)
    audit_logger.setLevel(logging.INFO)


def _configure_server_log_file(level: str) -> None:
    root = logging.getLogger()
    if any(getattr(h, "_pp_server_file", False) for h in root.handlers):
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.TimedRotatingFileHandler(
        LOG_DIR / "server.log",
        when="midnight",
        backupCount=0,  # keep forever
        encoding="utf-8",
        utc=True,
    )
    handler.suffix = "%Y-%m-%d"
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    handler.setLevel(level)
    handler._pp_server_file = True  # type: ignore[attr-defined]
    root.addHandler(handler)


def main() -> None:
    level = os.getenv("LOG_LEVEL", get_settings().log_level)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _configure_server_log_file(level)
    _configure_audit_logger()
    registry = get_registry()
    _assert_ready(registry)
    _warm_models(registry)
    _assert_ready(registry)

    settings = get_settings()
    host = os.getenv("PROD_HOST") or settings.host
    port = int(os.getenv("PORT") or os.getenv("PROD_PORT") or settings.port)
    logger.info("serving Properties Predict prod app on http://%s:%s", host, port)
    serve(create_app(), host=host, port=port, threads=int(os.getenv("PROD_THREADS", "8")))


if __name__ == "__main__":
    main()
