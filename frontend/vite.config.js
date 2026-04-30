import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
const BACKEND = process.env.VITE_BACKEND_URL ?? "http://127.0.0.1:8000";
export default defineConfig({
    plugins: [react()],
    // Ketcher pulls in Node builtins (util, assert) that reference `process`
    // and `global`; shim them for the browser.
    define: {
        "process.env": {},
        global: "globalThis",
    },
    server: {
        port: 5173,
        proxy: {
            "/predict": { target: BACKEND, changeOrigin: true },
            "/properties": { target: BACKEND, changeOrigin: true },
            "/engines": { target: BACKEND, changeOrigin: true },
            "/version": { target: BACKEND, changeOrigin: true },
            "/health": { target: BACKEND, changeOrigin: true },
        },
    },
});
