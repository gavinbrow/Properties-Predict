import type { Method } from "../api/types";
import { methodLabel } from "../api/format";

export function MethodBadge({ method }: { method: Method }) {
  return (
    <span className={`badge badge-method badge-method-${method}`}>
      {methodLabel(method)}
    </span>
  );
}
