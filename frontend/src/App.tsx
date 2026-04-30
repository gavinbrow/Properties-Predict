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

export default function App() {
  const ketcherRef = useRef<Ketcher | null>(null);
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
        <div
          className={`system-status${allVisibleReady ? "" : " system-status-warn"}`}
        >
          <span className="status-dot" aria-hidden="true" />
          <span>{statusText}</span>
        </div>
      </header>

      <main className="workbench">
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
    </div>
  );
}

function EmptyState() {
  return (
    <div className="empty-state">
      <h2>Awaiting structure</h2>
      <p>Draw a molecule, then run the selected prediction engines.</p>
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
