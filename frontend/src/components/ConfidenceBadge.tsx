import type { ConfidenceTier } from "../api/types";
import { tierLabel } from "../api/format";

export function ConfidenceBadge({ tier }: { tier: ConfidenceTier }) {
  return (
    <span className={`badge badge-conf badge-conf-${tier}`}>
      {tierLabel(tier)}
    </span>
  );
}
