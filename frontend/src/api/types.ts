// Type mirrors of the Pydantic schemas in backend/app/schemas/.
// Keep in sync with predict.py, engines.py, chemistry.py.

export type PropertyKey = "mp" | "bp" | "density";

export type ConfidenceTier =
  | "reference"
  | "high"
  | "medium"
  | "low"
  | "unsupported";

export type Method =
  | "lookup"
  | "knn"
  | "gnn"
  | "thermo"
  | "consensus"
  | "none";

export type EngineStatus =
  | "ok"
  | "out_of_domain"
  | "low_confidence"
  | "unsupported_property"
  | "parse_error"
  | "timeout"
  | "engine_error"
  | "not_configured";

export interface EngineResult {
  engine: string;
  engine_version: string;
  property: string;
  status: EngineStatus;
  value: number | null;
  unit: string | null;
  uncertainty: number | null;
  in_domain: boolean | null;
  ad_score: number | null;
  confidence_score: number | null;
  runtime_ms: number | null;
  warnings: string[];
  raw: Record<string, unknown> | null;
}

export interface DomainRejection {
  code: string;
  message: string;
}

export interface CompoundInfo {
  input_smiles: string;
  canonical_smiles: string | null;
  inchikey: string | null;
  formula: string | null;
  mw: number | null;
}

export interface PropertyPrediction {
  property: string;
  value: number | null;
  unit: string | null;
  confidence: ConfidenceTier;
  method: Method;
  selected_engine: string | null;
  range: [number, number] | null;
  uncertainty: number | null;
  source: string | null;
  warnings: string[];
  engine_details: EngineResult[];
}

export interface PredictionResponse {
  compound: CompoundInfo;
  supported: boolean;
  rejections: DomainRejection[];
  predictions: PropertyPrediction[];
}

export interface PredictionRequest {
  smiles: string;
  properties?: PropertyKey[];
  engines?: string[];
}

export interface PropertiesInfo {
  properties: string[];
  units: Record<string, string>;
  conditions: Record<string, string>;
}

export interface EngineInfo {
  name: string;
  version: string;
  supports: string[];
  ready?: boolean;
  ready_for?: string[];
  issues?: string[];
}

export interface EnginesInfo {
  engines: EngineInfo[];
  routing: Record<string, string[]>;
}

export interface VersionInfo {
  backend: string;
  rdkit: string;
  gnn: string;
  gnn_checkpoint_digest: string | null;
  gnn_dataset_manifest_digest: string | null;
  thermo: string;
  fastapi: string;
  pydantic: string;
}
