# Properties Predict — Prod

Self-contained production deployment of the Properties Predict web app.
Bundles the Flask API, the prebuilt frontend, and all model artifacts so it
can be cloned to a server and run with no other repos.

## Layout

```
.
├── app.py                  # Flask entrypoint (waitress-served)
├── server/                 # Internal Python package (engines, schemas, services, lookup)
├── data/
│   ├── trained_models/
│   │   ├── gnn/            # v7 GNN ensembles (BP distill, MP specialist) + calibration
│   │   └── knn/            # kNN BP/MP fingerprint stores
│   └── lookup/compounds.db # SQLite lookup DB
├── frontend/
│   ├── dist/               # Prebuilt UI (served by Flask)
│   ├── public/             # Static assets (favicon, etc.)
│   └── src/                # React/TS source if you want to rebuild
├── logs/                   # Runtime: server.log + predictions.jsonl (gitignored)
├── requirements.txt
└── start server.bat        # Windows convenience launcher
```

## Run

```bash
cd Properties-Predict
python -m pip install -r requirements.txt
python app.py
```

Defaults to `http://127.0.0.1:8000`. The single Flask process serves both
the API and the prebuilt UI from `frontend/dist/`.

### Environment overrides

| Var            | Default       | Purpose                                |
|----------------|---------------|----------------------------------------|
| `PROD_HOST`    | `127.0.0.1`   | Bind address                           |
| `PROD_PORT`    | `8000`        | Bind port                              |
| `PROD_THREADS` | `8`           | Waitress worker threads                |
| `LOG_LEVEL`    | `INFO`        | stdlib logging level                   |

## Engines

`app.py` validates that all three engines are ready before serving traffic
and warms the kNN + GNN artifacts in-process:

- `gnn` — v7 ensemble (5 seeds), BP distill + MP specialist, with calibrated
  uncertainty
- `knn` — Tanimoto kNN over Morgan fingerprints (BP + MP)
- `thermo` — Joback + COSTALD/Rackett offline density estimator

## Logging

Two files are written to `logs/` (gitignored, kept forever, daily rotation
at UTC midnight):

- `server.log` — application + access logs (Flask, waitress, internal)
- `predictions.jsonl` — one JSON object per `/predict` request:
  timestamp, client IP, SMILES (input + canonical), per-property predicted
  values + uncertainties + per-engine breakdown.

## Rebuilding the frontend (optional)

The `frontend/dist/` folder is checked in so the app is ready to run. If you
edit `frontend/src/` and need to rebuild:

```bash
cd frontend
npm install
npx vite build
```

Vite copies `frontend/public/favicon.ico` and `frontend/index.html` into
`frontend/dist/`.

## API

- `POST /predict`        — `{ "smiles": "CCO" }`
- `POST /predict/batch`  — `{ "items": [{ "smiles": "CCO" }, ...] }`
- `GET  /engines`        — engine inventory + readiness
- `GET  /properties`     — supported properties + units
- `GET  /version`        — engine versions + artifact digests
- `GET  /health`         — readiness check
- `GET  /`               — frontend SPA
