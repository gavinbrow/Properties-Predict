// Display helpers - keep frontend-visible labels in one place.

import type { ConfidenceTier, Method, PropertyPrediction } from "./types";

const PROPERTY_LABELS: Record<string, string> = {
  mp: "Melting point",
  bp: "Boiling point",
  density: "Density",
};

const UNIT_LABELS: Record<string, string> = {
  degC: "\u00B0C",
  "g/mL": "g/mL",
};

const CONDITIONS: Record<string, string> = {
  mp: "atmospheric",
  bp: "1 atm",
  density: "liquid, 25 \u00B0C, 1 atm",
};

const TIER_LABELS: Record<ConfidenceTier, string> = {
  reference: "Reference",
  high: "High",
  medium: "Medium",
  low: "Low",
  unsupported: "Unsupported",
};

const METHOD_LABELS: Record<Method, string> = {
  lookup: "Reference lookup",
  knn: "kNN match",
  gnn: "Neural ensemble",
  thermo: "Thermo",
  consensus: "Consensus",
  none: "No result",
};

const ENGINE_LABELS: Record<string, string> = {
  lookup: "Reference lookup",
  knn: "kNN match",
  gnn: "Neural ensemble",
  thermo: "Thermo",
};

export function propertyLabel(key: string): string {
  return PROPERTY_LABELS[key] ?? key;
}

export function unitLabel(unit: string | null | undefined): string {
  if (!unit) return "";
  return UNIT_LABELS[unit] ?? unit;
}

export function conditionLabel(prop: string): string | null {
  return CONDITIONS[prop] ?? null;
}

export function tierLabel(tier: ConfidenceTier): string {
  return TIER_LABELS[tier];
}

export function methodLabel(method: Method): string {
  return METHOD_LABELS[method];
}

export function engineLabel(engine: string): string {
  return ENGINE_LABELS[engine] ?? engine;
}

// EngineResult.engine_version carries an audit string of the form
// "<friendly-version>::<model-hash>" (e.g. "v7.0::gen7_mp_specialist_cuda"). The UI
// only wants the friendly prefix.
export function displayEngineVersion(engineVersion: string | null | undefined): string {
  if (!engineVersion) return "";
  const idx = engineVersion.indexOf("::");
  return idx === -1 ? engineVersion : engineVersion.slice(0, idx);
}

export function displayEngineBadgeVersion(
  engine: string,
  version: string | null | undefined,
): string {
  const shortVersion = displayEngineVersion(version);
  if (engine === "thermo") {
    const match = shortVersion.match(/^thermo-(\d+(?:\.\d+)*)$/);
    return match ? `v${match[1]}` : shortVersion;
  }
  return shortVersion;
}

export function formatValue(p: PropertyPrediction): string {
  const unit = unitLabel(p.unit);
  if (p.value !== null) {
    const digits = p.property === "density" ? 3 : 1;
    return `${p.value.toFixed(digits)} ${unit}`.trim();
  }
  if (p.range) {
    const digits = p.property === "density" ? 3 : 1;
    return `${p.range[0].toFixed(digits)} \u2013 ${p.range[1].toFixed(digits)} ${unit}`.trim();
  }
  return "\u2014";
}

export function formatRange(p: PropertyPrediction): string | null {
  if (!p.range) return null;
  const digits = p.property === "density" ? 3 : 1;
  const unit = unitLabel(p.unit);
  return `${p.range[0].toFixed(digits)} \u2013 ${p.range[1].toFixed(digits)} ${unit}`.trim();
}

export function uncertaintyLabel(p: PropertyPrediction): string | null {
  if (p.uncertainty == null) return null;
  const digits = p.property === "density" ? 3 : 1;
  const unit = unitLabel(p.unit);
  return `\u00B1${p.uncertainty.toFixed(digits)} ${unit}`.trim();
}

export function isLookup(p: PropertyPrediction): boolean {
  return p.method === "lookup";
}
