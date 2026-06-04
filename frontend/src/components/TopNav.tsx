/** TopNav — sticky top navigation bar (replaces the left sidebar). */

import { motion, AnimatePresence } from "framer-motion";
import type { HealthResponse } from "../lib/api";
import type { HealthStatus } from "../lib/useHealth";

export default function TopNav({
    health,
    status,
    hasMessages,
    onExport,
    onClear,
    onNew,
}: {
    health: HealthResponse | null;
    status: HealthStatus;
    hasMessages: boolean;
    onExport: () => void;
    onClear: () => void;
    onNew: () => void;
}) {
    const dot = status === "online" ? "#22c55e" : status === "offline" ? "#ef4444" : "#f59e0b";
    const statusLabel = status === "loading" ? "Connecting" : status === "online" ? "Online" : "Offline";

    return (
        <header className="sticky top-0 z-20 h-16 px-4 lg:px-8 flex items-center justify-between bg-card/80 backdrop-blur-xl border-b border-line">
            {/* Brand */}
            <button onClick={onNew} className="flex items-center gap-3 group">
                <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-brand-500 to-brand-700 flex items-center justify-center text-lg shadow-[0_4px_14px_rgba(34,197,94,0.35)] group-hover:scale-105 transition-transform">
                    🌾
                </div>
                <div className="text-left">
                    <div className="text-[15px] font-bold text-ink leading-none">AgriBot</div>
                    <div className="text-[10px] text-ink-3 uppercase tracking-wider mt-0.5">Agricultural Advisory</div>
                </div>
            </button>

            {/* Right cluster */}
            <div className="flex items-center gap-2 lg:gap-3">
                {/* KB stats — desktop only */}
                {health && (
                    <div className="hidden xl:flex items-center gap-3 mr-2 text-[11px] text-ink-3">
                        <span><b className="text-ink-2 tabular-nums">{health.chunk_count}</b> chunks</span>
                        <span className="w-px h-3 bg-line" />
                        <span><b className="text-ink-2 tabular-nums">{health.kg_entities}</b> entities</span>
                        <span className="w-px h-3 bg-line" />
                        <span><b className="text-ink-2 tabular-nums">{health.kg_relations}</b> relations</span>
                    </div>
                )}

                {/* Status pill */}
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-muted border border-line text-xs font-semibold text-ink-2">
                    <span className="w-2 h-2 rounded-full animate-pulse-dot" style={{ background: dot }} />
                    {statusLabel}
                </div>

                {/* Conversation actions */}
                <AnimatePresence>
                    {hasMessages && (
                        <motion.div
                            className="flex items-center gap-2"
                            initial={{ opacity: 0, x: 8 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: 8 }}
                        >
                            <button onClick={onExport} className="text-xs px-3 py-1.5 rounded-full border border-line text-ink-2 hover:bg-muted hover:text-ink transition-colors">Export</button>
                            <button onClick={onNew} className="text-xs px-3 py-1.5 rounded-full bg-brand-600 text-white font-medium hover:bg-brand-700 shadow-[0_2px_10px_rgba(34,197,94,0.3)] transition-colors">New chat</button>
                            <button onClick={onClear} className="text-xs w-8 h-8 rounded-full border border-line text-ink-3 hover:text-red-500 hover:border-red-200 hover:bg-red-50 transition-colors" title="Clear conversation">🗑</button>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </header>
    );
}
