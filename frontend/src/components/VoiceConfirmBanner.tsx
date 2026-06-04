/**
 * VoiceConfirmBanner — inline confirmation UI for low-confidence voice transcription.
 *
 * Shown when the voice endpoint returns needs_confirmation=true.
 * Users can edit the transcript and confirm, or cancel and re-record.
 */

import { useState } from "react";
import { motion } from "framer-motion";

interface Props {
    transcript: string;
    confidence: number;
    language: string;
    warnings: string[];
    suggestedActions: string[];
    traceId: string;
    onConfirm: (editedText: string, traceId: string) => void;
    onCancel: () => void;
}

export default function VoiceConfirmBanner({
    transcript,
    confidence,
    language,
    warnings,
    suggestedActions,
    traceId,
    onConfirm,
    onCancel,
}: Props) {
    const [editedText, setEditedText] = useState(transcript);

    const pct = Math.round(confidence * 100);
    const tone =
        confidence >= 0.6
            ? "text-brand-700 bg-brand-50 border-brand-200"
            : confidence >= 0.4
              ? "text-amber-300 bg-amber-500/15 border-amber-500/30"
              : "text-rose-300 bg-rose-500/15 border-rose-500/30";

    return (
        <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.3, ease: "easeOut" }}
            className="rounded-2xl border border-amber-200 bg-amber-50 p-4 shadow-[var(--shadow-card)]"
        >
            {/* Header */}
            <div className="flex items-center gap-2 mb-3">
                <span className="text-lg">🎤</span>
                <strong className="flex-1 text-sm text-amber-900">Voice transcription — please review</strong>
                <span className={`text-[11px] font-semibold px-2.5 py-0.5 rounded-full border ${tone}`}>
                    {pct}% confidence
                </span>
            </div>

            {/* Language + warnings */}
            <div className="text-xs text-amber-800 mb-2">
                Detected language:{" "}
                <strong className="text-amber-900">
                    {language === "bn" ? "Bengali 🇧🇩" : language === "en" ? "English" : language}
                </strong>
                {warnings.length > 0 && (
                    <span className="ml-2 text-amber-700">⚠ {warnings.join(", ")}</span>
                )}
            </div>

            {/* Editable transcript */}
            <textarea
                id="voice-confirm-transcript"
                value={editedText}
                onChange={(e) => setEditedText(e.target.value)}
                placeholder="Edit the transcribed text if needed..."
                className="w-full min-h-[60px] p-3 rounded-xl border border-amber-200 bg-card text-ink text-sm resize-y outline-none focus:border-brand-400 transition-colors mb-3"
            />

            {/* Suggested actions */}
            {suggestedActions.length > 0 && (
                <div className="text-xs text-amber-800 mb-3">💡 {suggestedActions.join(" · ")}</div>
            )}

            {/* Buttons */}
            <div className="flex gap-2 justify-end">
                <button
                    id="voice-confirm-cancel"
                    onClick={onCancel}
                    className="px-4 py-2 rounded-xl border border-line bg-card text-ink-2 text-[13px] hover:bg-muted transition-colors"
                >
                    ✕ Cancel
                </button>
                <motion.button
                    id="voice-confirm-submit"
                    whileHover={{ scale: editedText.trim() ? 1.04 : 1 }}
                    whileTap={{ scale: editedText.trim() ? 0.96 : 1 }}
                    onClick={() => onConfirm(editedText, traceId)}
                    disabled={!editedText.trim()}
                    className="px-4 py-2 rounded-xl bg-brand-600 text-white font-semibold text-[13px] shadow-[0_2px_12px_rgba(34,197,94,0.35)] hover:bg-brand-700 transition-colors disabled:opacity-50 disabled:shadow-none"
                >
                    ✓ Confirm &amp; Send
                </motion.button>
            </div>
        </motion.div>
    );
}
