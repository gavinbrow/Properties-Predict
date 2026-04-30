import type {
  EnginesInfo,
  PredictionRequest,
  PredictionResponse,
  PropertiesInfo,
  VersionInfo,
} from "./types";

const BASE = import.meta.env.VITE_BACKEND_URL ?? "";

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
    public readonly detail?: unknown,
  ) {
    super(message);
  }
}

async function request<T>(
  path: string,
  init?: RequestInit,
  signal?: AbortSignal,
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    signal,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    // Read the body once as text — calling res.json() then falling back to
    // res.text() throws "body stream already read" because the JSON parse
    // consumes the stream even when it fails.
    const bodyText = await res.text();
    let detail: unknown = bodyText;
    if (bodyText) {
      try {
        detail = JSON.parse(bodyText);
      } catch {
        // Not JSON — keep the raw text as the detail.
      }
    }
    const msg =
      typeof detail === "object" && detail !== null && "detail" in detail
        ? String((detail as { detail: unknown }).detail)
        : bodyText || res.statusText;
    throw new ApiError(res.status, msg, detail);
  }
  return (await res.json()) as T;
}

export function predict(
  req: PredictionRequest,
  signal?: AbortSignal,
): Promise<PredictionResponse> {
  return request<PredictionResponse>(
    "/predict",
    { method: "POST", body: JSON.stringify(req) },
    signal,
  );
}

export function getProperties(signal?: AbortSignal): Promise<PropertiesInfo> {
  return request<PropertiesInfo>("/properties", undefined, signal);
}

export function getEngines(signal?: AbortSignal): Promise<EnginesInfo> {
  return request<EnginesInfo>("/engines", undefined, signal);
}

export function getVersion(signal?: AbortSignal): Promise<VersionInfo> {
  return request<VersionInfo>("/version", undefined, signal);
}
