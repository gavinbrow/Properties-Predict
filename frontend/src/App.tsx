import { Suspense, lazy, useCallback, useEffect, useRef, useState } from "react";
import type { Ketcher } from "ketcher-core";
import { ApiError, getEngines, predict } from "./api/client";
import type { EngineInfo, PredictionResponse } from "./api/types";
import { CompoundHeader } from "./components/CompoundHeader";
import { EngineSelector } from "./components/EngineSelector";
import { PropertyCard } from "./components/PropertyCard";
import { RejectionList } from "./components/RejectionList";

type BrowserRequire = (id: string) => unknown;

const KetcherEditor = lazy(async () => {
  const raphaelModule = await import("raphael");
  const raphael = raphaelModule.default ?? raphaelModule;
  const browserWindow = window as unknown as { require?: BrowserRequire };
  const browserGlobal = globalThis as unknown as { require?: BrowserRequire };
  const previousRequire = browserWindow.require;
  const browserRequire: BrowserRequire = (id) => {
    if (id === "raphael") return raphael;
    if (previousRequire) return previousRequire(id);
    throw new Error(`Unsupported browser require: ${id}`);
  };

  browserWindow.require = browserRequire;
  browserGlobal.require = browserRequire;

  return import("./components/KetcherEditor").then((module) => ({
    default: module.KetcherEditor,
  }));
});

const ACTIVE_ENGINE_NAMES = new Set(["gnn", "thermo", "knn"]);

type Status =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "result"; data: PredictionResponse };

type ActiveTab = "predict" | "how";

export default function App() {
  const ketcherRef = useRef<Ketcher | null>(null);
  const [activeTab, setActiveTab] = useState<ActiveTab>("predict");
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [smiles, setSmiles] = useState<string>("");
  const [smilesInput, setSmilesInput] = useState<string>("");
  const [availableEngines, setAvailableEngines] = useState<EngineInfo[]>([]);
  const [selectedEngines, setSelectedEngines] = useState<string[] | null>(null);
  const [editorReady, setEditorReady] = useState(false);

  const refreshEngines = useCallback(async () => {
    try {
      const data = await getEngines();
      const activeEngines = data.engines.filter((engine) =>
        ACTIVE_ENGINE_NAMES.has(engine.name),
      );
      setAvailableEngines(activeEngines);
      setSelectedEngines((current) => {
        if (current !== null) {
          return current.filter((name) => ACTIVE_ENGINE_NAMES.has(name));
        }
        return activeEngines
          .filter((engine) => engine.ready !== false)
          .map((engine) => engine.name);
      });
    } catch {
      // Leave existing state alone on transient failures so the user can
      // still toggle/deselect engines that are stuck.
    }
  }, []);

  useEffect(() => {
    void refreshEngines();
    // Re-poll readiness so engines that go down (thermo/KNN can hiccup) are
    // reflected in the chip state without a page refresh.
    const id = window.setInterval(() => {
      void refreshEngines();
    }, 15000);
    return () => window.clearInterval(id);
  }, [refreshEngines]);

  const handleReady = useCallback((k: Ketcher) => {
    ketcherRef.current = k;
    setEditorReady(true);
  }, []);

  const handleToggleEngine = useCallback((engine: string) => {
    if (!ACTIVE_ENGINE_NAMES.has(engine)) return;
    setSelectedEngines((current) => {
      const base = current ?? [];
      return base.includes(engine)
        ? base.filter((name) => name !== engine)
        : [...base, engine];
    });
  }, []);

  const handleLoadSmiles = useCallback(
    async (value = smilesInput): Promise<string | null> => {
      const ketcher = ketcherRef.current;
      if (!ketcher) {
        setStatus({ kind: "error", message: "Editor not ready yet." });
        return null;
      }

      const nextSmiles = value.trim();
      if (!nextSmiles) {
        setStatus({ kind: "error", message: "Enter a SMILES string first." });
        return null;
      }

      try {
        await ketcher.setMolecule(nextSmiles);
      } catch (err) {
        setStatus({
          kind: "error",
          message: `Failed to load SMILES into editor: ${String(err)}`,
        });
        return null;
      }

      let editorSmiles = nextSmiles;
      try {
        editorSmiles = (await ketcher.getSmiles()).trim() || nextSmiles;
      } catch {
        editorSmiles = nextSmiles;
      }

      setSmiles(editorSmiles);
      setSmilesInput(editorSmiles);
      setStatus({ kind: "idle" });
      return editorSmiles;
    },
    [smilesInput],
  );

  const handlePredict = useCallback(async () => {
    const ketcher = ketcherRef.current;
    if (!ketcher) {
      setStatus({ kind: "error", message: "Editor not ready yet." });
      return;
    }

    const enginesToRun = selectedEngines ?? Array.from(ACTIVE_ENGINE_NAMES);
    if (enginesToRun.length === 0) {
      setStatus({ kind: "error", message: "Select at least one engine." });
      return;
    }

    let drawnSmiles = "";
    try {
      drawnSmiles = (await ketcher.getSmiles()).trim();
    } catch (err) {
      setStatus({
        kind: "error",
        message: `Failed to read SMILES from editor: ${String(err)}`,
      });
      return;
    }
    if (!drawnSmiles) {
      const loadedSmiles = await handleLoadSmiles(smilesInput);
      if (!loadedSmiles) {
        setStatus({
          kind: "error",
          message: "Draw a structure or enter a SMILES string first.",
        });
        return;
      }
      drawnSmiles = loadedSmiles;
    }
    setSmiles(drawnSmiles);
    setSmilesInput(drawnSmiles);
    setStatus({ kind: "loading" });
    try {
      const data = await predict({
        smiles: drawnSmiles,
        engines: enginesToRun,
      });
      setStatus({ kind: "result", data });
    } catch (err) {
      const message =
        err instanceof ApiError
          ? `API ${err.status}: ${err.message}`
          : `Request failed: ${String(err)}`;
      setStatus({ kind: "error", message });
    } finally {
      void refreshEngines();
    }
  }, [handleLoadSmiles, refreshEngines, selectedEngines, smilesInput]);

  const allVisibleReady =
    availableEngines.length > 0 &&
    availableEngines.every((engine) => engine.ready !== false);
  const statusText = allVisibleReady
    ? "All engines nominal"
    : "Engine checks degraded";
  const canPredict =
    editorReady &&
    status.kind !== "loading" &&
    (selectedEngines === null || selectedEngines.length > 0);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <span className="brand-name">predict</span>
          <span className="brand-badge">ALPHA</span>
        </div>
        <nav className="site-tabs" aria-label="Primary">
          <button
            type="button"
            className={`site-tab${activeTab === "predict" ? " site-tab-active" : ""}`}
            aria-controls="predict-panel"
            aria-selected={activeTab === "predict"}
            onClick={() => setActiveTab("predict")}
          >
            Predict
          </button>
          <button
            type="button"
            className={`site-tab${activeTab === "how" ? " site-tab-active" : ""}`}
            aria-controls="how-panel"
            aria-selected={activeTab === "how"}
            onClick={() => setActiveTab("how")}
          >
            How it works
          </button>
        </nav>
        <div
          className={`system-status${allVisibleReady ? "" : " system-status-warn"}`}
        >
          <span className="status-dot" aria-hidden="true" />
          <span>{statusText}</span>
        </div>
      </header>

      <main
        id="predict-panel"
        className="workbench"
        hidden={activeTab !== "predict"}
      >
        <section className="molecule-pane">
          <EngineSelector
            engines={availableEngines}
            selected={selectedEngines ?? []}
            onToggle={handleToggleEngine}
            disabled={status.kind === "loading"}
          />
          <Suspense
            fallback={
              <div className="ketcher-host ketcher-loading">
                Loading editor...
              </div>
            }
          >
            <KetcherEditor
              onReady={handleReady}
              onError={(e) =>
                setStatus({
                  kind: "error",
                  message: `Editor error: ${String(e)}`,
                })
              }
            />
          </Suspense>
          <div className="run-panel">
            <div className="smiles-readout">
              <label htmlFor="smiles-input">SMILES</label>
              <input
                id="smiles-input"
                value={smilesInput}
                placeholder={smiles || "Paste SMILES"}
                spellCheck={false}
                onChange={(event) => setSmilesInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    void handleLoadSmiles();
                  }
                }}
              />
              <button
                type="button"
                className="smiles-load-btn"
                onClick={() => void handleLoadSmiles()}
                disabled={!editorReady || !smilesInput.trim() || status.kind === "loading"}
              >
                Load
              </button>
            </div>
            <button
              type="button"
              className="predict-btn"
              onClick={handlePredict}
              disabled={!canPredict}
            >
              {status.kind === "loading" ? "Predicting..." : "Predict Properties"}
              {status.kind !== "loading" && <span aria-hidden="true">-&gt;</span>}
            </button>
          </div>
        </section>

        <section className="results-pane">
          {status.kind === "idle" && <EmptyState />}
          {status.kind === "loading" && <LoadingState />}
          {status.kind === "error" && (
            <div className="error-panel">{status.message}</div>
          )}
          {status.kind === "result" && <ResultView data={status.data} />}
        </section>
      </main>
      <main id="how-panel" className="info-page" hidden={activeTab !== "how"}>
        <HowItWorks />
      </main>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="empty-state">
      <h2>Awaiting structure</h2>
      <p>Draw a molecule, then run the selected property engine.</p>
    </div>
  );
}

function LoadingState() {
  return (
    <div className="loading-state">
      <div className="skeleton skeleton-header" />
      <div className="cards">
        <div className="skeleton skeleton-card" />
        <div className="skeleton skeleton-card" />
        <div className="skeleton skeleton-card" />
      </div>
    </div>
  );
}

function ResultView({ data }: { data: PredictionResponse }) {
  return (
    <div className="result-view">
      <CompoundHeader c={data.compound} />
      {!data.supported ? (
        <RejectionList rejections={data.rejections} />
      ) : (
        <div className="cards">
          {data.predictions.map((p) => (
            <PropertyCard key={p.property} p={p} />
          ))}
        </div>
      )}
    </div>
  );
}

function HowItWorks() {
  return (
    <div className="info-content">
      <section className="how-hero">
        <p className="eyebrow">How it works</p>
        <h1>What happens when Properties Predict estimates a molecule</h1>
        <p>
          The site converts a drawn structure or SMILES string into a standardized
          molecule, checks whether it fits the supported chemistry scope, looks
          for measured reference data, runs the relevant local prediction engine
          for each property, and then reports only the result the backend can
          defend.
        </p>
      </section>

      <section className="how-section">
        <div className="section-heading">
          <p className="eyebrow">Prediction pipeline</p>
          <h2>From structure to answer</h2>
        </div>
        <div className="flow-grid">
          <article className="info-card">
            <span className="step-number">01</span>
            <h3>Read and standardize the molecule</h3>
            <p>
              The editor output is read as SMILES. RDKit parses the structure,
              canonicalizes the SMILES, calculates formula and molecular weight,
              and creates an InChIKey so equivalent drawings resolve to the same
              compound identity.
            </p>
          </article>
          <article className="info-card">
            <span className="step-number">02</span>
            <h3>Apply chemistry guardrails</h3>
            <p>
              The backend rejects molecules outside the current model scope:
              disconnected mixtures, radicals, molecular weight outside 10 to
              1000 Da, and atoms outside H, B, C, N, O, F, Si, P, S, Cl, Br,
              and I. Unsupported structures return reasons instead of guesses.
            </p>
          </article>
          <article className="info-card">
            <span className="step-number">03</span>
            <h3>Check measured reference data</h3>
            <p>
              For each requested property, the InChIKey is checked against the
              bundled lookup database. A hit is returned as a reference value
              with its source and the prediction engine is skipped for that
              property.
            </p>
          </article>
          <article className="info-card">
            <span className="step-number">04</span>
            <h3>Run the property engine</h3>
            <p>
              Each property is handled by one active prediction route. Melting
              point, boiling point, and density each return one engine answer
              with that engine's quality checks, warnings, and uncertainty.
              Engine chips in the prediction view control which routes are
              allowed to run.
            </p>
          </article>
          <article className="info-card">
            <span className="step-number">05</span>
            <h3>Qualify the result</h3>
            <p>
              The selected property engine either returns a value, returns a
              value with lower confidence, or declines to answer. Domain checks,
              similarity checks, uncertainty, and model warnings decide whether
              the result is shown as high, medium, low, or unsupported.
            </p>
          </article>
        </div>
      </section>

      <section className="how-section">
        <div className="section-heading">
          <p className="eyebrow">Engines</p>
          <h2>What each property engine contributes</h2>
        </div>
        <div className="engine-explain-grid">
          <article className="info-card info-card-accent">
            <h3>kNN similarity model</h3>
            <p>
              The kNN engine fingerprints the query molecule with Morgan
              fingerprints and compares it to the curated training set using
              Tanimoto similarity. If the nearest neighbor is at or above the
              short-circuit threshold stored in the model artifact, the top
              neighbor's measured value is used directly. Lower similarity can
              still produce a lower-confidence estimate, while very low
              similarity is treated as out of domain.
            </p>
          </article>
          <article className="info-card info-card-accent">
            <h3>GNN ensemble</h3>
            <p>
              The GNN engine represents the molecule as a graph with atom, bond,
              global descriptor, fingerprint, and shape features. Multiple v7
              model checkpoints vote as an ensemble. The engine reports a value,
              uncertainty, similarity to the training set, and calibrated error
              checks that can mark a prediction low confidence or out of domain.
            </p>
          </article>
          <article className="info-card info-card-accent">
            <h3>Thermo density model</h3>
            <p>
              Density is estimated offline from chemistry correlations rather
              than a learned neural model. Joback group contributions estimate
              critical properties, then COSTALD and Rackett liquid-density
              equations estimate density. The result is downgraded when group
              fragmentation is incomplete, the molecule may not be liquid at
              25 degC, or the correlations disagree.
            </p>
          </article>
        </div>
      </section>

      <section className="how-section">
        <div className="section-heading">
          <p className="eyebrow">Confidence</p>
          <h2>How to read the answer</h2>
        </div>
        <div className="confidence-grid">
          <article className="confidence-row">
            <h3>Reference</h3>
            <p>A measured lookup value was found for that compound and property.</p>
          </article>
          <article className="confidence-row">
            <h3>High</h3>
            <p>
              The assigned engine returned one of its strongest signals, such
              as a kNN short-circuit match or a warning-free result with high
              native confidence.
            </p>
          </article>
          <article className="confidence-row">
            <h3>Medium</h3>
            <p>
              The assigned engine returned a usable value, but the quality
              signals are more typical than exceptional.
            </p>
          </article>
          <article className="confidence-row">
            <h3>Low</h3>
            <p>
              The assigned engine produced an estimate, but it flagged weak
              similarity, higher uncertainty, or another warning. Treat it as a
              rough estimate.
            </p>
          </article>
          <article className="confidence-row">
            <h3>Unsupported</h3>
            <p>
              The molecule or property is outside scope, or the assigned engine
              could not produce a trustworthy value.
            </p>
          </article>
        </div>
      </section>

      <section className="how-section">
        <div className="section-heading">
          <p className="eyebrow">Scope</p>
          <h2>What the numbers mean</h2>
        </div>
        <div className="scope-grid">
          <article className="info-card">
            <h3>Supported properties</h3>
            <ul>
              <li>Melting point, reported in degC under atmospheric conditions.</li>
              <li>Boiling point, reported in degC at 1 atm.</li>
              <li>Liquid density, reported in g/mL at 25 degC and 1 atm.</li>
            </ul>
          </article>
          <article className="info-card">
            <h3>Important limits</h3>
            <ul>
              <li>Predictions are estimates, not lab measurements.</li>
              <li>The app prefers an unsupported result over a confident guess.</li>
              <li>
                Model quality depends on similarity to the training data and
                whether the molecule fits the stated chemistry scope.
              </li>
            </ul>
          </article>
        </div>
      </section>
    </div>
  );
}
