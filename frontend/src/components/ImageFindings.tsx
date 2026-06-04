/**
 * ImageFindings — visualises what the vision pipeline detected for an
 * uploaded crop image: the user's thumbnail, top-K disease predictions with
 * animated confidence bars, and the keyword/symptom chips the CNN + heuristic
 * analyzer extracted (these are what get fused into the RAG query).
 *
 * Data comes from the backend ImageBlock: possible_conditions, plus the richer
 * analysis_summary (keywords, symptom_hints, ocr_text, quality_flags).
 */

import { motion, type Variants } from "framer-motion";
import type { ImageBlock } from "../lib/api";

const PIPELINE_LABEL: Record<string, string> = {
    classifier_assisted: "CNN classifier",
    ocr_baseline: "OCR + heuristics",
    ocr_fallback: "OCR fallback",
};

function confidenceTone(c: number): { bar: string; text: string } {
    if (c >= 0.7) return { bar: "from-brand-500 to-brand-400", text: "text-brand-700" };
    if (c >= 0.4) return { bar: "from-amber-500 to-amber-400", text: "text-amber-600" };
    return { bar: "from-rose-500 to-rose-400", text: "text-rose-500" };
}

export default function ImageFindings({
    image,
    imageUrl,
    onZoom,
}: {
    image: ImageBlock;
    imageUrl?: string;
    onZoom?: (src: string) => void;
}) {
    const summary = image.analysis_summary || {};
    const conditions = image.possible_conditions || summary.possible_conditions || [];
    const keywords = summary.keywords || [];
    const symptoms = summary.symptom_hints || [];
    const qualityFlags = summary.quality_flags || [];
    const ocrText = (summary.ocr_text || "").trim();
    const pipelineLabel = PIPELINE_LABEL[image.pipeline_used] || image.pipeline_used;

    // Merge keywords + symptoms into one de-duplicated chip set (the "what the
    // model found" view the user asked for), tagging the source for colour.
    const chips: { text: string; kind: "symptom" | "keyword" }[] = [];
    const seen = new Set<string>();
    for (const s of symptoms) {
        const k = s.toLowerCase();
        if (!seen.has(k)) { seen.add(k); chips.push({ text: s, kind: "symptom" }); }
    }
    for (const w of keywords) {
        const k = w.toLowerCase();
        if (!seen.has(k)) { seen.add(k); chips.push({ text: w, kind: "keyword" }); }
    }

    const container: Variants = {
        hidden: {},
        show: { transition: { staggerChildren: 0.06, delayChildren: 0.05 } },
    };
    const item: Variants = {
        hidden: { opacity: 0, y: 8 },
        show: { opacity: 1, y: 0, transition: { duration: 0.3, ease: "easeOut" as const } },
    };

    return (
        <motion.div
            variants={container}
            initial="hidden"
            animate="show"
            className="mx-5 mb-4 rounded-xl border border-line bg-muted/40 overflow-hidden"
        >
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-2.5 border-b border-line bg-card/60">
                <div className="flex items-center gap-2">
                    <span className="text-sm">🔬</span>
                    <span className="text-xs font-semibold text-ink">Image analysis</span>
                </div>
                <span className="text-[10px] px-2 py-0.5 rounded-full border border-brand-200 text-brand-700 bg-brand-50">
                    {pipelineLabel}
                </span>
            </div>

            <div className="p-4 flex gap-4">
                {/* Thumbnail */}
                {imageUrl && (
                    <motion.div variants={item} className="shrink-0">
                        <img
                            src={imageUrl}
                            alt="Uploaded crop"
                            onClick={onZoom ? () => onZoom(imageUrl) : undefined}
                            className={`w-28 h-28 object-cover rounded-lg border border-line shadow-[var(--shadow-card)] ${onZoom ? "cursor-zoom-in hover:opacity-90 transition-opacity" : ""}`}
                        />
                        {qualityFlags.length > 0 && (
                            <div className="mt-1.5 text-[9px] text-amber-600 text-center">
                                ⚠ {qualityFlags.join(", ")}
                            </div>
                        )}
                    </motion.div>
                )}

                <div className="flex-1 min-w-0 space-y-3">
                    {/* Predictions with confidence bars */}
                    {conditions.length > 0 && (
                        <motion.div variants={item}>
                            <div className="text-[10px] text-ink-3 uppercase tracking-wide mb-1.5">
                                Possible conditions
                            </div>
                            <div className="space-y-1.5">
                                {conditions.slice(0, 4).map((c, i) => {
                                    const pct = Math.round((c.confidence || 0) * 100);
                                    const tone = confidenceTone(c.confidence || 0);
                                    return (
                                        <div key={`${c.label}-${i}`} className="flex items-center gap-2">
                                            <span className="text-xs text-ink w-32 truncate" title={c.label}>
                                                {c.label}
                                            </span>
                                            <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden border border-line">
                                                <motion.div
                                                    className={`h-full rounded-full bg-gradient-to-r ${tone.bar}`}
                                                    initial={{ width: 0 }}
                                                    animate={{ width: `${pct}%` }}
                                                    transition={{ duration: 0.7, delay: 0.15 + i * 0.08, ease: "easeOut" }}
                                                />
                                            </div>
                                            <span className={`text-[11px] tabular-nums w-9 text-right font-semibold ${tone.text}`}>
                                                {pct}%
                                            </span>
                                        </div>
                                    );
                                })}
                            </div>
                        </motion.div>
                    )}

                    {/* Detected keywords / symptoms */}
                    {chips.length > 0 && (
                        <motion.div variants={item}>
                            <div className="text-[10px] text-ink-3 uppercase tracking-wide mb-1.5">
                                Detected features
                            </div>
                            <div className="flex flex-wrap gap-1.5">
                                {chips.map((c, i) => (
                                    <motion.span
                                        key={`${c.text}-${i}`}
                                        variants={item}
                                        className={`text-[11px] px-2 py-0.5 rounded-full border ${
                                            c.kind === "symptom"
                                                ? "border-amber-200 bg-amber-50 text-amber-700"
                                                : "border-brand-200 bg-brand-50 text-brand-700"
                                        }`}
                                    >
                                        {c.text}
                                    </motion.span>
                                ))}
                            </div>
                        </motion.div>
                    )}

                    {/* OCR text, if any */}
                    {ocrText && (
                        <motion.div variants={item}>
                            <div className="text-[10px] text-ink-3 uppercase tracking-wide mb-1">
                                Text in image (OCR)
                            </div>
                            <div className="text-[11px] text-ink-2 bg-card border border-line rounded p-2 leading-snug line-clamp-3">
                                {ocrText.slice(0, 240)}
                            </div>
                        </motion.div>
                    )}

                    {/* Empty state */}
                    {conditions.length === 0 && chips.length === 0 && !ocrText && (
                        <motion.div variants={item} className="text-xs text-ink-3 italic">
                            No distinct features were extracted from this image. Try a closer,
                            well-lit photo of the affected area.
                        </motion.div>
                    )}
                </div>
            </div>

            {/* Limitations footer */}
            {image.limitations && image.limitations.length > 0 && (
                <div className="px-4 py-2 border-t border-line bg-card/40 text-[10px] text-ink-3">
                    {image.limitations[0]}
                </div>
            )}
        </motion.div>
    );
}
