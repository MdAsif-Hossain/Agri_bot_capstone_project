import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

export default function Composer({ onText, onVoice, onImage, disabled }: {
    onText: (t: string) => void;
    onVoice: (b: Blob) => void;
    onImage: (f: File, q: string) => void;
    disabled: boolean;
}) {
    const [text, setText] = useState("");
    const [isRecording, setIsRecording] = useState(false);
    const [image, setImage] = useState<File | null>(null);
    const [imgUrl, setImgUrl] = useState<string | null>(null);
    const imageRef = useRef<HTMLInputElement>(null);
    const mediaStreamRef = useRef<MediaStream | null>(null);
    const audioCtxRef = useRef<AudioContext | null>(null);
    const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
    const processorRef = useRef<ScriptProcessorNode | null>(null);
    const gainRef = useRef<GainNode | null>(null);
    const pcmChunksRef = useRef<Float32Array[]>([]);

    const mergeFloat32 = (chunks: Float32Array[]): Float32Array => {
        const total = chunks.reduce((sum, c) => sum + c.length, 0);
        const out = new Float32Array(total);
        let offset = 0;
        for (const c of chunks) {
            out.set(c, offset);
            offset += c.length;
        }
        return out;
    };

    const resampleLinear = (input: Float32Array, inputRate: number, targetRate: number): Float32Array => {
        if (inputRate === targetRate) return input;
        const ratio = inputRate / targetRate;
        const outLength = Math.max(1, Math.round(input.length / ratio));
        const out = new Float32Array(outLength);
        for (let i = 0; i < outLength; i++) {
            const pos = i * ratio;
            const left = Math.floor(pos);
            const right = Math.min(left + 1, input.length - 1);
            const frac = pos - left;
            out[i] = input[left] * (1 - frac) + input[right] * frac;
        }
        return out;
    };

    const trimSilence = (input: Float32Array, threshold = 0.01): Float32Array => {
        let start = 0;
        let end = input.length - 1;

        while (start < input.length && Math.abs(input[start]) < threshold) start += 1;
        while (end > start && Math.abs(input[end]) < threshold) end -= 1;

        if (start >= end) return input;
        return input.slice(start, end + 1);
    };

    const normalizePeak = (input: Float32Array, targetPeak = 0.9): Float32Array => {
        let peak = 0;
        for (let i = 0; i < input.length; i++) {
            const a = Math.abs(input[i]);
            if (a > peak) peak = a;
        }
        if (peak <= 1e-6) return input;
        const gain = Math.min(8.0, targetPeak / peak);
        if (gain <= 1.02) return input;
        const out = new Float32Array(input.length);
        for (let i = 0; i < input.length; i++) {
            out[i] = Math.max(-1, Math.min(1, input[i] * gain));
        }
        return out;
    };

    const encodeWav = (samples: Float32Array, sampleRate: number): ArrayBuffer => {
        const numChannels = 1;
        const bitsPerSample = 16;
        const blockAlign = numChannels * (bitsPerSample / 8);
        const byteRate = sampleRate * blockAlign;
        const dataSize = samples.length * 2;
        const buffer = new ArrayBuffer(44 + dataSize);
        const view = new DataView(buffer);

        const writeString = (offset: number, value: string) => {
            for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i));
        };

        writeString(0, "RIFF");
        view.setUint32(4, 36 + dataSize, true);
        writeString(8, "WAVE");
        writeString(12, "fmt ");
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, byteRate, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitsPerSample, true);
        writeString(36, "data");
        view.setUint32(40, dataSize, true);

        let offset = 44;
        for (let i = 0; i < samples.length; i++) {
            const s = Math.max(-1, Math.min(1, samples[i]));
            const int16 = s < 0 ? s * 0x8000 : s * 0x7fff;
            view.setInt16(offset, int16, true);
            offset += 2;
        }

        return buffer;
    };

    useEffect(() => {
        if (image) { const u = URL.createObjectURL(image); setImgUrl(u); return () => URL.revokeObjectURL(u); }
        setImgUrl(null);
    }, [image]);

    const handleSubmit = () => {
        if (disabled) return;
        if (image) { onImage(image, text); setImage(null); setText(""); }
        else if (text.trim()) { onText(text.trim()); setText(""); }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
    };

    const toggleVoice = async () => {
        if (isRecording) {
            try {
                const sourceRate = audioCtxRef.current?.sampleRate || 48000;

                processorRef.current?.disconnect();
                sourceRef.current?.disconnect();
                gainRef.current?.disconnect();

                if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
                    await audioCtxRef.current.close();
                }
                mediaStreamRef.current?.getTracks().forEach(t => t.stop());

                const merged = mergeFloat32(pcmChunksRef.current);
                const targetRate = 16000;
                const resampled = resampleLinear(merged, sourceRate, targetRate);
                const trimmed = trimSilence(resampled, 0.01);
                const prepared = normalizePeak(trimmed.length > 4000 ? trimmed : resampled, 0.9);
                const wav = encodeWav(prepared, targetRate);
                const blob = new Blob([wav], { type: "audio/wav" });
                onVoice(blob);
            } catch (e) {
                console.error("Voice finalize failed", e);
                alert("Could not process recorded audio.");
            } finally {
                setIsRecording(false);
                pcmChunksRef.current = [];
                audioCtxRef.current = null;
                mediaStreamRef.current = null;
                sourceRef.current = null;
                processorRef.current = null;
                gainRef.current = null;
            }
            return;
        }
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false,
                },
            });

            const audioCtx = new AudioContext();
            await audioCtx.resume();
            const source = audioCtx.createMediaStreamSource(stream);
            const processor = audioCtx.createScriptProcessor(4096, 1, 1);
            const silent = audioCtx.createGain();
            silent.gain.value = 0;

            pcmChunksRef.current = [];

            processor.onaudioprocess = (event) => {
                const input = event.inputBuffer.getChannelData(0);
                pcmChunksRef.current.push(new Float32Array(input));
            };

            source.connect(processor);
            processor.connect(silent);
            silent.connect(audioCtx.destination);

            mediaStreamRef.current = stream;
            audioCtxRef.current = audioCtx;
            sourceRef.current = source;
            processorRef.current = processor;
            gainRef.current = silent;

            setIsRecording(true);
        } catch (e) { console.error("Mic access denied", e); alert("Microphone access denied."); }
    };

    return (
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-canvas via-canvas to-transparent pt-10 pb-6 px-4 lg:px-8">
            <div className="max-w-3xl mx-auto bg-card border border-line rounded-2xl shadow-[var(--shadow-pop)] p-2 transition-all focus-within:border-brand-400 focus-within:shadow-[0_8px_30px_rgba(34,197,94,0.15)] relative">

                {/* Image Preview Area */}
                <AnimatePresence>
                    {imgUrl && (
                        <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: "auto" }}
                            exit={{ opacity: 0, height: 0 }}
                            transition={{ duration: 0.25, ease: "easeOut" }}
                            className="overflow-hidden"
                        >
                            <div className="p-3 border-b border-line flex items-start gap-4">
                                <div className="relative group">
                                    <img src={imgUrl} alt="Upload" className="h-20 w-20 object-cover rounded-lg border border-line shadow-[var(--shadow-card)]" />
                                    <button onClick={() => setImage(null)} className="absolute -top-2 -right-2 w-6 h-6 bg-card border border-line text-red-500 rounded-full text-xs flex items-center justify-center hover:bg-red-50 transition-colors shadow-sm">✕</button>
                                </div>
                                <div className="text-xs text-ink-2 pt-2 flex-1">
                                    <span className="text-brand-700 font-medium">📷 Ready to analyze.</span> Add a question about this image, or just press send — the vision model will identify the crop condition automatically.
                                </div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                <div className="flex items-end gap-2 p-1">
                    {/* Add Image Button */}
                    <button onClick={() => imageRef.current?.click()} disabled={disabled || isRecording} className="p-3 shrink-0 text-ink-3 hover:text-brand-600 hover:bg-muted rounded-xl transition-colors disabled:opacity-50">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
                        <input type="file" ref={imageRef} hidden accept="image/*" onChange={e => { if (e.target.files?.[0]) setImage(e.target.files[0]); e.target.value = ''; }} />
                    </button>

                    {/* Input/Recording Area */}
                    <div className="flex-1 min-w-0 relative flex items-center">
                        {isRecording ? (
                            <div className="w-full flex items-center justify-center h-10 gap-1.5 text-red-500 animate-pulse-rec bg-red-50 rounded-xl border border-red-100">
                                <span className="w-2 h-2 rounded-full bg-red-500 flex shrink-0" />
                                <span className="text-sm font-medium">Recording voice... click mic to send</span>
                                <div className="flex items-center gap-0.5 ml-4 h-4">
                                    {[...Array(5)].map((_, i) => <div key={i} className="w-1 bg-red-400 rounded-full animate-wave" style={{ animationDelay: `${i * 0.15}s` }} />)}
                                </div>
                            </div>
                        ) : (
                            <textarea
                                value={text}
                                onChange={e => setText(e.target.value)}
                                onKeyDown={handleKeyDown}
                                disabled={disabled}
                                placeholder={image ? "Add details..." : "Ask AgriBot any crop question (English বা বাংলা)..."}
                                className="w-full bg-transparent text-ink placeholder-ink-3 resize-none outline-none py-3 px-2 min-h-[44px] max-h-[160px] text-[15px] leading-relaxed block overflow-hidden"
                                style={{ height: '44px' }}
                                rows={1}
                                onInput={e => {
                                    const t = e.target as HTMLTextAreaElement;
                                    t.style.height = '44px';
                                    t.style.height = Math.min(t.scrollHeight, 160) + 'px';
                                }}
                            />
                        )}
                    </div>

                    {/* Actions */}
                    <div className="flex items-center gap-1 shrink-0 pb-1">
                        {(!text.trim() && !image) ? (
                            <motion.button whileTap={{ scale: 0.9 }} onClick={toggleVoice} disabled={disabled} className={`p-2.5 rounded-xl transition-colors ${isRecording ? "bg-red-500 text-white shadow-[0_0_15px_rgba(239,68,68,0.4)]" : "bg-muted text-ink-2 hover:bg-brand-50 hover:text-brand-600"} disabled:opacity-50`}>
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={isRecording ? "animate-pulse" : ""}><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="22"></line></svg>
                            </motion.button>
                        ) : (
                            <motion.button whileHover={{ scale: 1.06 }} whileTap={{ scale: 0.92 }} onClick={handleSubmit} disabled={disabled} className="p-2.5 rounded-xl bg-brand-600 text-white hover:bg-brand-700 shadow-[0_2px_12px_rgba(34,197,94,0.35)] transition-colors disabled:opacity-50 disabled:shadow-none">
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
                            </motion.button>
                        )}
                    </div>
                </div>

                <div className="absolute -bottom-5 left-0 right-0 text-center text-[10px] text-ink-3">
                    AgriBot can make mistakes. Verify critical advice.
                </div>
            </div>
        </div>
    );
}
