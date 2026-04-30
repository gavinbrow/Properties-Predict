import type { EngineResult } from "../api/types";
import { displayEngineVersion, engineLabel, unitLabel } from "../api/format";

const ENGINE_ORDER: Record<string, number> = {
  lookup: 0,
  knn: 1,
  gnn: 2,
  thermo: 3,
};

const STATUS_LABELS: Record<string, string> = {
  ok: "OK",
  low_confidence: "Low confidence",
  out_of_domain: "Out of domain",
  unsupported_property: "Unsupported property",
  parse_error: "Parse error",
  timeout: "Timed out",
  engine_error: "Engine error",
  not_configured: "Not configured",
};

type Metric = {
  label: string;
  value: string;
};

function numberDigits(property: string): number {
  return property === "density" ? 3 : 1;
}

function formatMetricValue(n: number | null, digits = 2): string | null {
  return n == null ? null : n.toFixed(digits);
}

function formatEngineValue(result: EngineResult): string {
  if (result.value == null) return "No usable value";
  const unit = unitLabel(result.unit);
  return `${result.value.toFixed(numberDigits(result.property))} ${unit}`.trim();
}

function formatSigma(result: EngineResult): string | null {
  if (result.uncertainty == null) return null;
  const unit = unitLabel(result.unit);
  return `\u00B1${result.uncertainty.toFixed(numberDigits(result.property))} ${unit}`.trim();
}

function formatRuntime(runtimeMs: number | null): string | null {
  return runtimeMs == null ? null : `${runtimeMs} ms`;
}

function statusLabel(status: string): string {
  return STATUS_LABELS[status] ?? status.replaceAll("_", " ");
}

function sortResults(
  results: EngineResult[],
  selectedEngine: string | null,
): EngineResult[] {
  return [...results].sort((a, b) => {
    const aSelected = selectedEngine != null && a.engine === selectedEngine;
    const bSelected = selectedEngine != null && b.engine === selectedEngine;
    if (aSelected !== bSelected) return aSelected ? -1 : 1;

    const orderDelta =
      (ENGINE_ORDER[a.engine] ?? Number.MAX_SAFE_INTEGER) -
      (ENGINE_ORDER[b.engine] ?? Number.MAX_SAFE_INTEGER);
    if (orderDelta !== 0) return orderDelta;

    return a.engine.localeCompare(b.engine);
  });
}

export function EngineDetails({
  results,
  selectedEngine = null,
}: {
  results: EngineResult[];
  selectedEngine?: string | null;
}) {
  if (results.length === 0) return null;

  const orderedResults = sortResults(results, selectedEngine);

  return (
    <details className="engine-details">
      <summary>
        <span aria-hidden="true">&gt;</span>
        {results.length} model output{results.length === 1 ? "" : "s"}
      </summary>

      <div className="engine-grid">
        {orderedResults.map((result, index) => {
          const sigma = formatSigma(result);
          const adScore = formatMetricValue(result.ad_score, 2);
          const confidenceScore = formatMetricValue(result.confidence_score, 2);
          const runtime = formatRuntime(result.runtime_ms);
          const metrics: Metric[] = [
            sigma ? { label: "Uncertainty", value: sigma } : null,
            result.in_domain == null
              ? null
              : {
                  label: "Domain",
                  value: result.in_domain ? "In domain" : "Out of domain",
                },
            adScore ? { label: "AD score", value: adScore } : null,
            confidenceScore
              ? { label: "Conf.", value: confidenceScore }
              : null,
            runtime ? { label: "Runtime", value: runtime } : null,
          ].filter((metric): metric is Metric => metric !== null);

          const isSelected =
            selectedEngine != null && result.engine === selectedEngine;

          return (
            <article
              key={`${result.engine}-${index}`}
              className={`engine-card engine-card-${result.status}${isSelected ? " engine-card-selected" : ""}`}
            >
              <header className="engine-card-header">
                <div className="engine-card-title">
                  <span className="engine-name">{engineLabel(result.engine)}</span>
                  {result.engine_version && (
                    <span className="engine-version">{displayEngineVersion(result.engine_version)}</span>
                  )}
                </div>

                <div className="engine-card-flags">
                  {isSelected && <span className="engine-flag">Selected</span>}
                  <span className={`status-pill status-pill-${result.status}`}>
                    {statusLabel(result.status)}
                  </span>
                </div>
              </header>

              <div
                className={`engine-card-value${result.value == null ? " engine-card-value-empty" : ""}`}
              >
                {formatEngineValue(result)}
              </div>

              {metrics.length > 0 && (
                <dl className="engine-metrics">
                  {metrics.map((metric) => (
                    <div key={metric.label} className="engine-metric">
                      <dt>{metric.label}</dt>
                      <dd>{metric.value}</dd>
                    </div>
                  ))}
                </dl>
              )}

              {result.warnings.length > 0 && (
                <ul className="engine-warnings">
                  {result.warnings.map((warning, warningIndex) => (
                    <li key={warningIndex}>{warning}</li>
                  ))}
                </ul>
              )}
            </article>
          );
        })}
      </div>
    </details>
  );
}
