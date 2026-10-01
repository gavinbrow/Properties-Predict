import {
  displayEngineBadgeVersion,
  engineLabel,
  propertyLabel,
} from "../api/format";
import type { EngineInfo } from "../api/types";

const ENGINE_ORDER: Record<string, number> = {
  gnn: 1,
  thermo: 2,
  knn: 3,
};

function orderedEngines(engines: EngineInfo[]): EngineInfo[] {
  return [...engines].sort((a, b) => {
    const orderDelta =
      (ENGINE_ORDER[a.name] ?? Number.MAX_SAFE_INTEGER) -
      (ENGINE_ORDER[b.name] ?? Number.MAX_SAFE_INTEGER);
    if (orderDelta !== 0) return orderDelta;
    return a.name.localeCompare(b.name);
  });
}

function propertySummary(engine: EngineInfo): string {
  const props = engine.ready_for ?? engine.supports ?? [];
  if (props.length === 0) return "No ready properties";
  return props.map((prop) => propertyLabel(prop)).join(", ");
}

export function EngineSelector({
  engines,
  selected,
  onToggle,
  disabled = false,
}: {
  engines: EngineInfo[];
  selected: string[];
  onToggle: (engine: string) => void;
  disabled?: boolean;
}) {
  if (engines.length === 0) return null;

  return (
    <section className="engine-selector">
      <h2>Prediction Engines</h2>

      <div className="engine-chip-row">
        {orderedEngines(engines).map((engine) => {
          const ready = engine.ready ?? true;
          const checked = selected.includes(engine.name);
          const version = displayEngineBadgeVersion(engine.name, engine.version);

          // Allow toggling off even if the engine reports not-ready, so the
          // user can always disable a stuck engine (e.g. KNN/thermo) without
          // refreshing the page.
          const canToggle = ready || checked;
          return (
            <button
              type="button"
              key={engine.name}
              className={`engine-chip${checked ? " engine-chip-selected" : ""}${!ready ? " engine-chip-disabled" : ""}`}
              disabled={disabled || !canToggle}
              onClick={() => onToggle(engine.name)}
              title={
                ready
                  ? propertySummary(engine)
                  : checked
                    ? `${engine.issues?.[0] ?? "Engine unavailable"} - click to disable`
                    : engine.issues?.[0] ?? "Engine unavailable"
              }
            >
              <span className="engine-chip-dot" aria-hidden="true" />
              <span className="engine-chip-name">{engineLabel(engine.name)}</span>
              {version && (
                <span className="engine-chip-version">{version}</span>
              )}
            </button>
          );
        })}
      </div>
      {engines.filter((engine) => engine.name === "gnn").map((engine) => (
        <p className="engine-model-summary" key={engine.name}>
          {Object.entries(engine.properties ?? {}).filter(([, model]) => model.model_label)
            .map(([prop, model]) => `${propertyLabel(prop)}: ${model.model_label}${model.members ? ` (${model.members} models)` : ""}`)
            .join(" · ")}
        </p>
      ))}
    </section>
  );
}
