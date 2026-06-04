import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { Message } from "../lib/api";
import { getTTSAudio } from "../lib/api";
import ImageFindings from "./ImageFindings";
import Markdown from "./Markdown";
import Lightbox from "./Lightbox";

type Tab = "en" | "bn";

export default function MessageBubble({ message: m, onFollowUp, onRetry }: { message: Message; onFollowUp?: (q: string) => void; onRetry?: (id: string) => void }) {
    const [tab, setTab] = useState<Tab>("en");
    const [showDiagnostics, setShowDiagnostics] = useState(false);
    const [isPlaying, setIsPlaying] = useState(false);
    const [zoom, setZoom] = useState<string | null>(null);
    const audioRef = useRef<HTMLAudioElement | null>(null);

    const isAssistant = m.role === "assistant";
    const content = tab === "en" ? m.content : (m.answer_bn || m.content);

    const toggleDiagnostic = () => setShowDiagnostics(p => !p);

    const handleTTS = async () => {
        if (isPlaying && audioRef.current) {
            audioRef.current.pause();
            setIsPlaying(false);
            return;
        }
        try {
            if (!audioRef.current || audioRef.current.dataset.lang !== tab) {
                const url = await getTTSAudio(content, tab);
                if (audioRef.current) URL.revokeObjectURL(audioRef.current.src);
                audioRef.current = new Audio(url);
                audioRef.current.dataset.lang = tab;
                audioRef.current.onended = () => setIsPlaying(false);
            }
            setIsPlaying(true);
            await audioRef.current.play();
        } catch (e) {
            console.error("TTS failed", e);
            const msg = e instanceof Error ? e.message : "TTS failed";
            window.alert(msg);
            setIsPlaying(false);
        }
    };

    useEffect(() => {
        return () => { if (audioRef.current) URL.revokeObjectURL(audioRef.current.src); };
    }, []);

    // --- USER message ---
    if (!isAssistant) {
        return (
            <motion.div
                className="flex justify-end"
                initial={{ opacity: 0, y: 10, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ duration: 0.25, ease: "easeOut" }}
                layout
            >
                <div className="max-w-[80%] flex flex-col items-end gap-2">
                    {m.image_url && (
                        <img
                            src={m.image_url}
                            alt="Uploaded"
                            onClick={() => setZoom(m.image_url!)}
                            className="max-h-44 rounded-2xl rounded-tr-sm border border-line object-cover shadow-[var(--shadow-card)] cursor-zoom-in hover:opacity-90 transition-opacity"
                        />
                    )}
                    <div className="bg-gradient-to-br from-brand-500 to-brand-700 text-white px-5 py-3 rounded-2xl rounded-tr-sm shadow-[0_4px_16px_rgba(34,197,94,0.28)]">
                        {m.input_mode === "voice" && <span className="mr-2 opacity-80">🎤</span>}
                        {m.input_mode === "image" && <span className="mr-2 opacity-80">📷</span>}
                        {m.content}
                    </div>
                </div>
                <Lightbox src={zoom} onClose={() => setZoom(null)} />
            </motion.div>
        );
    }

    // --- ERROR message ---
    if (m.content.startsWith("⚠️")) {
        return (
            <motion.div
                className="flex gap-4"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.25 }}
                layout
            >
                <div className="w-10 h-10 rounded-xl bg-red-50 border border-red-200 text-red-500 flex items-center justify-center text-lg shrink-0">⚠️</div>
                <div className="bg-red-50 border border-red-200 px-5 py-3.5 rounded-2xl rounded-tl-sm text-red-600 text-sm">{m.content.slice(2)}</div>
            </motion.div>
        );
    }

    const verifColor = m.is_verified ? "text-brand-700 bg-brand-50 border-brand-200" : "text-amber-700 bg-amber-50 border-amber-200";
    const groundColor = m.grounding_action === "pass" ? "text-brand-600" : m.grounding_action === "refuse" ? "text-red-500" : "text-amber-600";

    return (
        <motion.div
            className="flex gap-4 group"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, ease: "easeOut" }}
            layout
        >
            {/* Avatar */}
            <div className="relative shrink-0">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-500 to-brand-700 flex items-center justify-center text-lg shadow-[0_4px_14px_rgba(34,197,94,0.3)]">🌾</div>
                {m.input_mode === "image" && <div className="absolute -bottom-1 -right-1 w-4 h-4 rounded-full bg-brand-600 border-2 border-card flex items-center justify-center text-[8px]">📷</div>}
            </div>

            <div className="flex-1 min-w-0">
                <div className="bg-card border border-line rounded-2xl rounded-tl-sm shadow-[var(--shadow-card)] overflow-hidden">

                    {/* Tabs & Controls */}
                    {m.answer_bn && (
                        <div className="flex items-center justify-between border-b border-line px-4 py-2 bg-muted/60 text-xs">
                            <div className="flex gap-1 bg-muted rounded-full p-0.5 border border-line">
                                {(["en", "bn"] as Tab[]).map(t => (
                                    <button
                                        key={t}
                                        onClick={() => setTab(t)}
                                        className={`relative px-3 py-1 rounded-full font-medium transition-colors ${tab === t ? "text-white" : "text-ink-3 hover:text-ink-2"}`}
                                    >
                                        {tab === t && (
                                            <motion.span layoutId={`tab-${m.id}`} className="absolute inset-0 rounded-full bg-brand-600" transition={{ type: "spring", stiffness: 400, damping: 30 }} />
                                        )}
                                        <span className="relative z-10">{t === "en" ? "English" : "বাংলা"}</span>
                                    </button>
                                ))}
                            </div>
                            <div className="flex gap-2">
                                <button onClick={handleTTS} className={`w-7 h-7 rounded-full flex items-center justify-center transition-colors ${isPlaying ? "bg-brand-100 text-brand-700" : "text-ink-3 hover:bg-muted hover:text-brand-600"}`} title="Listen">
                                    {isPlaying ? "⏸" : "🔊"}
                                </button>
                                <button onClick={toggleDiagnostic} className={`flex items-center gap-1.5 px-2 py-1 rounded transition-colors ${showDiagnostics ? "bg-brand-100 text-brand-700" : "text-ink-3 hover:text-brand-600"}`}>
                                    <span className="text-[10px]">⚖️</span> Eval
                                </button>
                            </div>
                        </div>
                    )}

                    {/* Content — English rendered as markdown; Bengali (translated
                        prose) kept as plain wrapped text. */}
                    <motion.div
                        key={tab}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ duration: 0.2 }}
                        className="p-5"
                    >
                        {tab === "en"
                            ? <Markdown>{content}</Markdown>
                            : <div className="text-[15px] leading-relaxed whitespace-pre-wrap">{content}</div>}
                    </motion.div>

                    {/* Image findings (CNN keywords + predictions + thumbnail) */}
                    {m.image && <ImageFindings image={m.image} imageUrl={m.image_url} onZoom={setZoom} />}

                    {/* Citations */}
                    {m.citations && m.citations.length > 0 && (
                        <div className="px-5 pb-4">
                            <div className="flex flex-wrap gap-2 text-[11px]">
                                <span className="text-ink-3 my-auto mr-1">Sources:</span>
                                {m.citations.map((c, i) => (
                                    <span key={i} className="px-2 py-0.5 rounded-md border border-brand-200 bg-brand-50 text-brand-700 max-w-[200px] truncate" title={c}>{c}</span>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Diagnostics Drawer */}
                    <AnimatePresence>
                        {showDiagnostics && (
                            <motion.div
                                initial={{ height: 0, opacity: 0 }}
                                animate={{ height: "auto", opacity: 1 }}
                                exit={{ height: 0, opacity: 0 }}
                                transition={{ duration: 0.25, ease: "easeInOut" }}
                                className="overflow-hidden"
                            >
                                <div className="px-5 py-4 bg-muted/60 border-t border-line text-xs text-ink-2 space-y-3">
                                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                                        <div>
                                            <div className="text-[10px] text-ink-3 uppercase tracking-wide mb-1">Verification</div>
                                            <div className={`inline-block px-2 py-0.5 rounded border ${verifColor}`}>{m.is_verified ? "Verified" : "Failed / Unverified"}</div>
                                        </div>
                                        <div>
                                            <div className="text-[10px] text-ink-3 uppercase tracking-wide mb-1">Grounding Policy</div>
                                            <div className={`font-medium ${groundColor}`}>{m.grounding_action}</div>
                                        </div>
                                        <div>
                                            <div className="text-[10px] text-ink-3 uppercase tracking-wide mb-1">Evidence Grade</div>
                                            <div>{m.evidence_grade} (Retries: {m.retry_count})</div>
                                        </div>
                                        <div>
                                            <div className="text-[10px] text-ink-3 uppercase tracking-wide mb-1">Trace ID</div>
                                            <div className="font-mono text-[10px] truncate" title={m.diagnostics?.trace_id}>{m.diagnostics?.trace_id?.split('-')[0]}...</div>
                                        </div>
                                    </div>

                                    {m.verification_reason && (
                                        <div>
                                            <div className="text-[10px] text-ink-3 uppercase tracking-wide mb-1">LLM Verification Reason</div>
                                            <div className="bg-card border border-line rounded p-2 text-[11px] leading-snug">{m.verification_reason}</div>
                                        </div>
                                    )}

                                    {m.diagnostics?.timings_ms && Object.keys(m.diagnostics.timings_ms).length > 0 && (
                                        <div>
                                            <div className="text-[10px] text-ink-3 uppercase tracking-wide mb-2">Node Latency (ms)</div>
                                            <div className="flex flex-wrap gap-2 text-[10px]">
                                                {Object.entries(m.diagnostics.timings_ms).map(([k, v]) => (
                                                    <span key={k} className={`px-1.5 py-0.5 rounded ${k === 'total' ? 'bg-brand-100 text-brand-700 font-bold' : 'bg-card border border-line text-ink-2'}`}>
                                                        {k}: {Number(v)}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {m.kg_entities && m.kg_entities.length > 0 && (
                                        <div>
                                            <div className="text-[10px] text-ink-3 uppercase tracking-wide mb-1">Linked Entities ({m.kg_entities.length})</div>
                                            <div className="flex flex-wrap gap-1">
                                                {m.kg_entities.slice(0, 5).map((e, i) => (
                                                    <span key={i} className="text-[10px] px-1.5 py-0.5 bg-brand-50 text-brand-700 rounded">{e.en}</span>
                                                ))}
                                                {m.kg_entities.length > 5 && <span className="text-[10px] px-1.5 py-0.5 text-ink-3">+{m.kg_entities.length - 5} more</span>}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>

                {/* Action row: retry + follow-up suggestions */}
                {(onRetry || (m.follow_up_suggestions && m.follow_up_suggestions.length > 0 && onFollowUp)) && (
                    <div className="flex flex-wrap items-center gap-2 mt-3 ml-2">
                        {onRetry && (
                            <motion.button
                                whileHover={{ scale: 1.04 }}
                                whileTap={{ scale: 0.97 }}
                                onClick={() => onRetry(m.id)}
                                className="flex items-center gap-1 text-[11px] px-3 py-1 rounded-full border border-line text-ink-2 hover:bg-muted hover:text-brand-700 transition-colors"
                                title="Regenerate this answer"
                            >
                                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M23 4v6h-6" /><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" /></svg>
                                Retry
                            </motion.button>
                        )}
                        {m.follow_up_suggestions && m.follow_up_suggestions.length > 0 && onFollowUp && (
                            <>
                                <span className="text-[11px] text-ink-3 my-auto">Suggestions:</span>
                                {m.follow_up_suggestions.map(s => (
                                    <motion.button
                                        key={s}
                                        whileHover={{ scale: 1.04 }}
                                        whileTap={{ scale: 0.97 }}
                                        onClick={() => onFollowUp(s)}
                                        className="text-[11px] px-3 py-1 rounded-full border border-line text-ink-2 hover:bg-brand-50 hover:text-brand-700 hover:border-brand-200 transition-colors"
                                    >
                                        {s}
                                    </motion.button>
                                ))}
                            </>
                        )}
                    </div>
                )}
            </div>
            <Lightbox src={zoom} onClose={() => setZoom(null)} />
        </motion.div>
    );
}
