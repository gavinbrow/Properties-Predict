import type { CompoundInfo } from "../api/types";

export function CompoundHeader({ c }: { c: CompoundInfo }) {
  return (
    <div className="compound-header">
      {c.formula && (
        <div className="compound-row">
          <span className="compound-label">Formula</span>
          <code>{c.formula}</code>
        </div>
      )}
      {c.mw != null && (
        <div className="compound-row">
          <span className="compound-label">MW</span>
          <code>{c.mw.toFixed(2)} g/mol</code>
        </div>
      )}
      {c.inchikey && (
        <div className="compound-row">
          <span className="compound-label">InChIKey</span>
          <code>{c.inchikey}</code>
        </div>
      )}
      {c.canonical_smiles && (
        <div className="compound-row">
          <span className="compound-label">SMILES</span>
          <code>{c.canonical_smiles}</code>
        </div>
      )}
    </div>
  );
}
