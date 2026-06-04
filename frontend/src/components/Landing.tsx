/** Landing — the empty-state hero, distinct from the chat view. */

import { motion } from "framer-motion";
import type { HealthResponse } from "../lib/api";

const SUGGESTIONS: { icon: string; title: string; q: string }[] = [
    { icon: "🦠", title: "Diagnose a disease", q: "What causes rice blast disease and how do I manage it?" },
    { icon: "🐛", title: "Pest control", q: "ধানের পোকা দমনের উপায় কী?" },
    { icon: "🌱", title: "Fertilizer guidance", q: "What is a good fertilizer schedule for rice?" },
    { icon: "📷", title: "Identify from a photo", q: "How do I identify pest damage from a leaf photo?" },
];

const CAPABILITIES: { icon: string; label: string }[] = [
    { icon: "🔍", label: "Hybrid retrieval" },
    { icon: "🌿", label: "Dialect KG" },
    { icon: "🔬", label: "Disease CNN" },
    { icon: "🎤", label: "Voice in/out" },
    { icon: "🌐", label: "Bengali ↔ English" },
    { icon: "🛡️", label: "Safety refusal" },
];

const fade = {
    hidden: { opacity: 0, y: 14 },
    show: (i: number) => ({ opacity: 1, y: 0, transition: { duration: 0.4, delay: i * 0.06, ease: "easeOut" as const } }),
};

export default function Landing({
    health,
    onPick,
}: {
    health: HealthResponse | null;
    onPick: (q: string) => void;
}) {
    return (
        <div className="max-w-3xl mx-auto px-4 pt-10 pb-16 text-center">
            {/* Hero */}
            <motion.div custom={0} variants={fade} initial="hidden" animate="show">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-brand-50 border border-brand-100 text-brand-700 text-xs font-medium mb-6">
                    <span className="w-1.5 h-1.5 rounded-full bg-brand-500" /> Offline-first · runs on a 4&nbsp;GB laptop GPU
                </div>
            </motion.div>

            <motion.h1
                custom={1} variants={fade} initial="hidden" animate="show"
                className="text-4xl lg:text-5xl font-extrabold tracking-tight text-ink leading-[1.1]"
            >
                Crop advice you can <span className="text-brand-600">trust</span>,<br className="hidden sm:block" />
                in Bengali &amp; English.
            </motion.h1>

            <motion.p
                custom={2} variants={fade} initial="hidden" animate="show"
                className="mt-4 text-base text-ink-2 max-w-xl mx-auto"
            >
                Ask about diseases, pests, and fertilizers — or upload a leaf photo and let the
                vision model identify it. Every answer is grounded in trusted manuals with citations.
            </motion.p>

            {/* Suggestion cards */}
            <motion.div
                className="grid sm:grid-cols-2 gap-3 mt-10 text-left"
                initial="hidden" animate="show"
                variants={{ hidden: {}, show: { transition: { staggerChildren: 0.07, delayChildren: 0.25 } } }}
            >
                {SUGGESTIONS.map((s) => (
                    <motion.button
                        key={s.title}
                        variants={{ hidden: { opacity: 0, y: 12 }, show: { opacity: 1, y: 0 } }}
                        whileHover={{ y: -3 }}
                        whileTap={{ scale: 0.99 }}
                        onClick={() => onPick(s.q)}
                        className="group flex items-start gap-3 p-4 rounded-2xl bg-card border border-line shadow-[var(--shadow-card)] hover:border-brand-300 hover:shadow-[var(--shadow-pop)] transition-all"
                    >
                        <span className="text-2xl shrink-0">{s.icon}</span>
                        <span>
                            <span className="block text-sm font-semibold text-ink group-hover:text-brand-700 transition-colors">{s.title}</span>
                            <span className="block text-xs text-ink-3 mt-0.5 line-clamp-3">{s.q}</span>
                        </span>
                    </motion.button>
                ))}
            </motion.div>

            {/* Capability chips */}
            <motion.div
                custom={3} variants={fade} initial="hidden" animate="show"
                className="flex flex-wrap items-center justify-center gap-2 mt-10"
            >
                {CAPABILITIES.map((c) => (
                    <span key={c.label} className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-muted border border-line text-xs text-ink-2">
                        <span>{c.icon}</span> {c.label}
                    </span>
                ))}
            </motion.div>

            {/* KB stat strip */}
            {health && (
                <motion.div
                    custom={4} variants={fade} initial="hidden" animate="show"
                    className="grid grid-cols-4 gap-3 mt-10 max-w-md mx-auto"
                >
                    {[
                        [health.chunk_count, "Chunks"],
                        [health.kg_entities, "Entities"],
                        [health.kg_aliases, "Aliases"],
                        [health.kg_relations, "Relations"],
                    ].map(([v, l]) => (
                        <div key={l as string} className="rounded-xl bg-card border border-line py-3 shadow-[var(--shadow-card)]">
                            <div className="text-xl font-bold text-brand-600 tabular-nums">{v as number}</div>
                            <div className="text-[10px] text-ink-3 uppercase tracking-wide mt-0.5">{l as string}</div>
                        </div>
                    ))}
                </motion.div>
            )}
        </div>
    );
}
