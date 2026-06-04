/** Shared health polling hook — one interval feeds nav + landing. */

import { useEffect, useState } from "react";
import { getHealth, type HealthResponse } from "./api";

export type HealthStatus = "loading" | "online" | "offline";

export function useHealth(intervalMs = 30000) {
    const [health, setHealth] = useState<HealthResponse | null>(null);
    const [status, setStatus] = useState<HealthStatus>("loading");

    useEffect(() => {
        let live = true;
        const check = async () => {
            try {
                const d = await getHealth();
                if (live) { setHealth(d); setStatus("online"); }
            } catch {
                if (live) setStatus("offline");
            }
        };
        check();
        const t = setInterval(check, intervalMs);
        return () => { live = false; clearInterval(t); };
    }, [intervalMs]);

    return { health, status };
}
