import type { DomainRejection } from "../api/types";

export function RejectionList({ rejections }: { rejections: DomainRejection[] }) {
  if (rejections.length === 0) return null;
  return (
    <div className="rejection-panel">
      <h3>Input not supported</h3>
      <ul>
        {rejections.map((r, i) => (
          <li key={i}>
            <code>{r.code}</code> — {r.message}
          </li>
        ))}
      </ul>
    </div>
  );
}
