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
│   │   ├── gnn/            # Preserved Gen 7 models (optional explicit override)
│   │   ├── gnn_v14/        # Locked Gen 14 blend, nine models per property
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

- `gnn` — Gen 14 locked blend, nine neural models per property. In-domain
  estimates carry low confidence and null uncertainty until blend calibration is
  established; out-of-domain estimates emit no value.
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

## Updating to Gen 14

Stop the service, pull `main`, install the updated requirements, and restart:

```powershell
git pull --ff-only origin main
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python app.py
```

Use Python 3.10–3.12. All model files and frozen inference sources are bundled;
no research checkout is needed. If `GNN_MODELS_ROOT` in the environment or `.env`
points to the old `gnn` directory, remove that override or change it to
`data/trained_models/gnn_v14`. Confirm `/version` reports `gnn: v14.0` and
`/health` is ready after restart. CPU inference defaults to four threads;
`GNN_CPU_THREADS` can adjust it.

MP uses 0.28 GINE + 0.22 MAE-only GINE + 0.50 descriptor Chemprop; BP uses
0.18 GINE + 0.28 plain Chemprop + 0.54 descriptor Chemprop. Each constituent
averages seeds 7, 17 and 29 equally. The selection lock and every model hash
are recorded in `data/trained_models/gnn_v14/release.json`; files are verified
before loading. Structure-only cached descriptors preserve the exact established
training features, and new structures use the same frozen feature code.

Saved internal historical-holdout MAE: MP 26.640 °C, BP 28.759 °C. These are
not independent external validation. kNN and exact reference lookups can still
short-circuit the neural engine; disable kNN to inspect neural predictions.
Density remains the thermo estimator.
