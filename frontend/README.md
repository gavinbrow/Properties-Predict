# Properties Predict — Frontend

Vite + React + TypeScript client for the Properties Predict API. Uses
[Ketcher](https://github.com/epam/ketcher) (standalone mode) for molecule
drawing; the drawing is exported to SMILES and posted to the backend.

## Prereqs

- Node 20+
- The backend running locally (see `backend/README.md` — defaults to
  `http://127.0.0.1:8000`).

## Install and run

```bash
cd frontend
npm install
npm run dev
```

Then open <http://localhost:5173>.

Vite proxies `/predict`, `/predict/batch`, `/properties`, `/engines`,
`/version`, and `/health` to the backend. Override the backend target via
`VITE_BACKEND_URL` when starting Vite.

## Contract with the backend

The client is a thin shell over the API. It posts `{smiles}` to `/predict` and
renders the returned `PredictionResponse`:

- `compound` — canonical SMILES, InChIKey, formula, molecular weight.
- `supported` — `false` when the input failed parsing or domain filters; in
  that case `rejections` explains why.
- `predictions[]` — one card per property (MP, BP, density). Each carries:
  - `value` or `range` (range when engines disagree and no single value can be
    chosen),
  - `unit` (`degC` or `g/mL`) and implicit condition (MP atmospheric, BP 1 atm,
    density liquid at 25 °C),
  - `confidence` tier: `reference` / `high` / `medium` / `low` / `unsupported`,
  - `method`: `lookup` / `knn` / `gnn` / `thermo` / `consensus` / `none`,
  - `uncertainty` (1-sigma) when known,
  - `source` (for lookup hits),
  - `warnings`,
  - `engine_details[]` — per-engine raw outputs (status, value, sigma, AD,
    runtime) for an optional debug panel.

See [src/api/types.ts](src/api/types.ts) for the full TypeScript mirror.

## Structure

```
src/
├── api/
│   ├── client.ts      # fetch wrapper, typed endpoints
│   ├── format.ts      # label + value formatters
│   └── types.ts       # TS mirror of Pydantic schemas
├── components/
│   ├── CompoundHeader.tsx
│   ├── ConfidenceBadge.tsx
│   ├── EngineDetails.tsx
│   ├── KetcherEditor.tsx
│   ├── MethodBadge.tsx
│   ├── PropertyCard.tsx
│   └── RejectionList.tsx
├── App.tsx
├── main.tsx
└── styles.css
```
