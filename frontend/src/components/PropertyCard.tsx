import type { ConfidenceTier, PropertyPrediction } from "../api/types";
import {
  conditionLabel,
  engineLabel,
  formatRange,
  formatValue,
  isLookup,
  methodLabel,
  propertyLabel,
  tierLabel,
  uncertaintyLabel,
} from "../api/format";
import { EngineDetails } from "./EngineDetails";

const CONFIDENCE_LEVELS: Record<ConfidenceTier, number> = {
  unsupported: 0,
  low: 2,
  medium: 3,
  high: 4,
  reference: 5,
};

function ConfidenceMeter({ tier }: { tier: ConfidenceTier }) {
  const lit = CONFIDENCE_LEVELS[tier] ?? 0;
  return (
    <div className="confidence-meter" aria-label={`Confidence: ${tierLabel(tier)}`}>
      <span className="meter-bars" aria-hidden="true">
        {Array.from({ length: 5 }).map((_, index) => (
          <span
            key={index}
            className={index < lit ? "meter-bar meter-bar-on" : "meter-bar"}
          />
        ))}
      </span>
      <span className="meter-label">{tierLabel(tier)}</span>
    </div>
  );
}

export function PropertyCard({ p }: { p: PropertyPrediction }) {
  const cond = conditionLabel(p.property);
  const uncertainty = uncertaintyLabel(p);
  const engineSpan = p.value !== null ? formatRange(p) : null;
  const via = p.selected_engine ? engineLabel(p.selected_engine) : null;

  return (
    <article className={`card card-${p.confidence}`}>
      <header className="card-header">
        <h3>{propertyLabel(p.property)}</h3>
        <ConfidenceMeter tier={p.confidence} />
      </header>

      <div className="card-main">
        <div className="card-value">{formatValue(p)}</div>
      </div>

      <div className="card-chips">
        <span className="chip">
          <span>method</span>
          <strong>{methodLabel(p.method)}</strong>
        </span>
        {uncertainty && (
          <span className="chip">
            <span>err</span>
            <strong>{uncertainty}</strong>
          </span>
        )}
        {cond && (
          <span className="chip">
            <span>at</span>
            <strong>{cond}</strong>
          </span>
        )}
        {via && !isLookup(p) && (
          <span className="chip">
            <span>via</span>
            <strong>{via}</strong>
          </span>
        )}
        {p.source && (
          <span className="chip chip-source">
            <span>src</span>
            <strong>{p.source}</strong>
          </span>
        )}
        {engineSpan && (
          <span className="chip">
            <span>range</span>
            <strong>{engineSpan}</strong>
          </span>
        )}
      </div>

      {p.warnings.length > 0 && (
        <ul className="warnings">
          {p.warnings.map((w, i) => (
            <li key={i}>{w}</li>
          ))}
        </ul>
      )}

      <EngineDetails
        results={p.engine_details}
        selectedEngine={p.selected_engine}
      />
    </article>
  );
}
