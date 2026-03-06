import { useState, useEffect } from "react";
import { getHealth, type HealthResponse } from "../lib/api";

export default function Sidebar({ open, onClose }: { open: boolean; onClose: () => void }) {
    const [health, setHealth] = useState<HealthResponse | null>(null);
    const [status, setStatus] = useState<"loading" | "online" | "offline">("loading");

    useEffect(() => {
        let live = true;
        const check = async () => {
            try { const d = await getHealth(); if (live) { setHealth(d); setStatus("online"); } }
            catch { if (live) setStatus("offline"); }
        };
        check();
        const t = setInterval(check, 30000);
        return () => { live = false; clearInterval(t); };
    }, []);

    const features = [
        ["🔍", "Hybrid Retrieval (FAISS + BM25)"],
        ["⚖️", "Cross-Encoder Reranking"],
        ["🌿", "Dialect Knowledge Graph"],
        ["🔄", "Self-Correcting Agent Loop"],
        ["✅", "Answer Verification"],
        ["🛡️", "Grounding Policy Enforcement"],
        ["🎤", "Voice Input (Whisper ASR)"],
        ["🔊", "Voice Output (TTS)"],
        ["📷", "Image Analysis (OCR + Heuristic)"],
    ];

    return (
        <>
            {open && <div className="fixed inset-0 bg-black/50 z-40 lg:hidden" onClick={onClose} />}
            <aside className={`w-72 shrink-0 bg-glass/50 backdrop-blur-2xl border-r border-glass-border flex flex-col p-5 overflow-y-auto z-50 transition-transform lg:translate-x-0 fixed lg:static inset-y-0 left-0 ${open ? "translate-x-0" : "-translate-x-full"}`}>
                {/* Brand */}
                <div className="flex items-center gap-3 mb-6">
                    <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-emerald-600 to-emerald-400 flex items-center justify-center text-lg shadow-[0_0_15px_rgba(16,185,129,0.2)]">🌾</div>
                    <div>
                        <h1 className="text-base font-bold bg-gradient-to-r from-emerald-300 to-emerald-100 bg-clip-text text-transparent">AgriBot</h1>
                        <p className="text-[0.6rem] text-emerald-700 uppercase tracking-wider font-medium">Agricultural Advisory</p>
                    </div>
                </div>

                {/* Health */}
                <div className="mb-5">
                    <div className="text-[0.6rem] uppercase tracking-wider text-emerald-700 font-semibold mb-2">System Status</div>
                    <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold ${status === "online" ? "bg-green-500/10 text-green-400 border border-green-500/20" : status === "offline" ? "bg-red-500/10 text-red-400 border border-red-500/20" : "bg-yellow-500/10 text-yellow-400 border border-yellow-500/20"}`}>
                        <span className="w-2 h-2 rounded-full animate-pulse-dot" style={{ background: status === "online" ? "#22c55e" : status === "offline" ? "#ef4444" : "#f59e0b" }} />
                        {status === "loading" ? "Connecting…" : status === "online" ? "Online" : "Offline"}
                    </div>
                    {health && (
                        <div className="text-[0.6rem] text-emerald-700 mt-1.5">
                            Mode: <span className="text-emerald-400">{health.grounding_mode}</span>
                        </div>
                    )}
                </div>

                {/* Stats */}
                {health && (
                    <div className="mb-5">
                        <div className="text-[0.6rem] uppercase tracking-wider text-emerald-700 font-semibold mb-2">Knowledge Base</div>
                        <div className="grid grid-cols-2 gap-1.5">
                            {[
                                [health.chunk_count, "Chunks"],
                                [health.kg_entities, "Entities"],
                                [health.kg_aliases, "Aliases"],
                                [health.kg_relations, "Relations"],
                            ].map(([v, l]) => (
                                <div key={l as string} className="bg-surface-3 border border-glass-border rounded-lg p-2.5 text-center hover:border-emerald-600 hover:shadow-[0_0_10px_rgba(16,185,129,0.1)] transition-all">
                                    <div className="text-lg font-bold text-emerald-400 tabular-nums">{v as number}</div>
                                    <div className="text-[0.6rem] text-emerald-700 mt-0.5">{l as string}</div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Features */}
                <div className="flex-1 mb-4">
                    <div className="text-[0.6rem] uppercase tracking-wider text-emerald-700 font-semibold mb-2">Capabilities</div>
                    <ul className="space-y-0.5">
                        {features.map(([icon, label]) => (
                            <li key={label} className="flex items-center gap-2 text-xs text-emerald-200/70 px-2.5 py-1.5 rounded-md hover:bg-emerald-900/20 transition-colors">
                                <span className="w-4 text-center">{icon}</span>{label}
                            </li>
                        ))}
                    </ul>
                </div>

                <div className="text-[0.6rem] text-emerald-800 text-center">Offline-first • RTX 3050 Optimized</div>
            </aside>
        </>
    );
}
