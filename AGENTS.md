# AGENTS — Prod/

Self-contained production deployment of Properties Predict: a single-process Flask + waitress app that bundles the Python API, the prebuilt frontend, and all model artifacts so it can be cloned to a server and run with no other repos. Different stack from the dev backend (Flask/waitress vs FastAPI/uvicorn) and a reduced engine set (knn, gnn, thermo — no OPERA, no chemprop). It is a NESTED git repo with its own `.git/` and its own GitHub remote. Not referenced from the root README.

## Layout

```
Prod/
  .git/                   NESTED git repo (own remote, own history) — see Gotchas
  .gitignore              ignores __pycache__, logs/, .env, frontend/node_modules, *.png
  app.py                  Flask entrypoint; waitress-served; serves API + frontend/dist/
  requirements.txt        flask, waitress, pydantic-settings, rdkit, thermo, chemicals,
                          scikit-learn, torch, torch-geometric (no fastapi, no uvicorn,
                          no chemprop)
  start server.bat        Windows launcher: cd to dir, `python app.py`
  README.md               Prod-specific run + API docs
  server/                 Internal Python package (importable as `server.*`)
    core/                 config.py (Settings, paths default to Prod/data/), constants.py
    engines/             base.py, gnn.py, knn.py, thermo.py  (NO opera.py, NO chemprop.py)
    lookup/              service.py, models.py, db.py + data/lookup/compounds.db
    schemas/             predict.py, engines.py, chemistry.py
    services/            orchestrator.py, normalize.py, domain.py, consensus.py, calibration.py
  data/
    trained_models/gnn/   v7 GNN ensembles (5 seeds: 7,17,29,41,53) + calibration
    trained_models/knn/   kNN BP/MP fingerprint stores (data.npz + metadata.json)
    lookup/compounds.db   SQLite lookup DB
  frontend/
    dist/                 PREBUILT UI (index.html, assets/, favicon) — checked in, served by Flask
    public/, index.html, src/, package.json, vite.config.*, tsconfig.*, node_modules/
    .gitignore            ignores node_modules, .vite, .env
  logs/                   Runtime only (gitignored): server.log, predictions.jsonl
  __pycache__/            stale bytecode (gitignored)
```

## What this is

Production deployment bundle, NOT legacy. It is a deployable re-packaging of the main backend's prediction logic for a single host: `app.py` is a Flask app (importing from the bundled `server/` package) that validates all three engines are ready at boot, warms the kNN/GNN/thermo artifacts in-process, then serves both the REST API and the prebuilt SPA from `frontend/dist/` under one waitress process. It diverges from the dev backend in three ways: (1) web framework — Flask + waitress instead of FastAPI + uvicorn; (2) engine set — only knn, gnn, thermo (PROD_ROUTING routes mp/bp to knn+gnn, density to thermo); OPERA and chemprop are intentionally absent; (3) config — `server/core/config.py` resolves all artifact paths relative to `Prod/data/` so the bundle is self-contained. The `server/` package is a cleaned copy of the dev `backend/app/` package (same module names under core/engines/lookup/schemas/services, minus opera/chemprop and minus the dev-only `api/` and `tests/` subpackages).

## Key files

- `Prod/app.py` — Flask entrypoint (479 lines): routes, readiness check, audit JSONL logging, security headers, SPA fallback, `main()` boots waitress on PROD_HOST:PROD_PORT (defaults 127.0.0.1:8000, 8 threads).
- `Prod/requirements.txt` — prod-only deps; note torch + torch-geometric (heavy), no fastapi/uvicorn/chemprop.
- `Prod/server/core/config.py` — `Settings` (pydantic-settings); all paths default to `Prod/data/...`; reads `Prod/.env`.
- `Prod/server/core/constants.py` — supported atoms/properties, units, MW bounds (locked by docs/scope.md).
- `Prod/server/services/orchestrator.py` — per-request orchestration: normalize -> domain filter -> lookup -> engines -> consensus.
- `Prod/server/engines/{gnn,knn,thermo}.py` — the three prod engines.
- `Prod/frontend/dist/` — the checked-in prebuilt UI that `app.py` serves; rebuild via `cd frontend && npx vite build` (optional).
- `Prod/start server.bat` — Windows one-liner launcher.
- `Prod/README.md` — prod run/API docs (endpoints: /predict, /predict/batch, /engines, /properties, /version, /health, /).

## Gotchas

- Not referenced from the top-level `README.md` (stale reference — the root README only documents `backend/`, `frontend/`, `docs/`). An agent reading only the root README will not learn `Prod/` exists.
- `Prod/` is a NESTED git repository: it has its own `Prod/.git/` with remote `https://github.com/gavinbrow/Properties-Predict.git` on branch `main`. Do NOT run git commands inside `Prod/` from the root repo's context (e.g. `git add Prod/...` at the root will treat it as a gitlink/submodule-like entry, not ordinary files). Treat `Prod/` as an independent repo; only run git inside it if you explicitly intend to affect the prod deployment's history. Never push or commit here unless explicitly asked.
- `frontend/dist/` is checked into the Prod repo on purpose (so the app runs without an npm build). Do not delete it or gitignore it. `frontend/node_modules/` IS present on disk but is gitignored — it is not authoritative.
- `logs/` (server.log, predictions.jsonl) is runtime-generated and gitignored; `backupCount=0` means daily rotation keeps files forever — they can grow without bound. Do not assume logs reflect committed state.
- Prod's engine set is a subset of dev's: no `opera.py`, no `chemprop.py` in `Prod/server/engines/`. If a fix touches those engines, it belongs in the dev `backend/`, not here.
- `app.py` hard-fails at boot (`_assert_ready`) if any of knn/gnn/thermo is not ready, so a missing or corrupt `data/trained_models/*` artifact will stop the server from starting rather than degrade silently.
- Do not edit anything here unless you know why — this is the production deployable. Changes should flow from the dev backend with deliberate re-packaging, not ad-hoc edits inside `Prod/`.
