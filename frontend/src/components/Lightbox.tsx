/**
 * Lightbox — full-screen image viewer. Rendered at the app root via a portal
 * so it overlays everything. Closes on backdrop click or Escape.
 */

import { useEffect } from "react";
import { createPortal } from "react-dom";
import { motion, AnimatePresence } from "framer-motion";

export default function Lightbox({ src, onClose }: { src: string | null; onClose: () => void }) {
    useEffect(() => {
        if (!src) return;
        const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, [src, onClose]);

    return createPortal(
        <AnimatePresence>
            {src && (
                <motion.div
                    className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm p-6 cursor-zoom-out"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    onClick={onClose}
                >
                    <motion.img
                        src={src}
                        alt="Enlarged"
                        className="max-w-full max-h-full rounded-xl shadow-2xl object-contain"
                        initial={{ scale: 0.9, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        exit={{ scale: 0.9, opacity: 0 }}
                        transition={{ type: "spring", stiffness: 300, damping: 30 }}
                        onClick={(e) => e.stopPropagation()}
                    />
                    <button
                        onClick={onClose}
                        className="absolute top-5 right-5 w-10 h-10 rounded-full bg-white/15 border border-white/20 text-white text-lg flex items-center justify-center hover:bg-white/25 transition-colors"
                        aria-label="Close"
                    >
                        ✕
                    </button>
                </motion.div>
            )}
        </AnimatePresence>,
        document.body
    );
}
