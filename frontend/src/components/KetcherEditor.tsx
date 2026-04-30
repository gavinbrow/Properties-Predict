import { useRef } from "react";
import type { Ketcher } from "ketcher-core";
import { Editor } from "ketcher-react";
import { StandaloneStructServiceProvider } from "ketcher-standalone";
import "ketcher-react/dist/index.css";

type Props = {
  onReady: (ketcher: Ketcher) => void;
  onError?: (err: unknown) => void;
};

declare global {
  interface Window {
    ketcher?: Ketcher;
  }
}

// Ketcher requires a single, stable struct-service provider for its lifetime.
const structServiceProvider = new StandaloneStructServiceProvider();

export function KetcherEditor({ onReady, onError }: Props) {
  const onReadyRef = useRef(onReady);
  onReadyRef.current = onReady;

  return (
    <div className="ketcher-host">
      <Editor
        staticResourcesUrl=""
        structServiceProvider={structServiceProvider as never}
        errorHandler={(message: string) => onError?.(new Error(message))}
        onInit={(ketcher: Ketcher) => {
          window.ketcher = ketcher;
          onReadyRef.current(ketcher);
        }}
      />
    </div>
  );
}
