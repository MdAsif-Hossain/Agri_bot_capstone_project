import { useState, useRef, useCallback, useEffect } from "react";
import { AnimatePresence, motion } from "framer-motion";
import TopNav from "./components/TopNav";
import Landing from "./components/Landing";
import MessageBubble from "./components/MessageBubble";
import VoiceConfirmBanner from "./components/VoiceConfirmBanner";
import Composer from "./components/Composer";
import { useHealth } from "./lib/useHealth";
import { sendChat, sendVoice, sendVoiceConfirm, sendImage, exportCaseReport } from "./lib/api";
import type { Message, ChatResponse, VoiceBlock } from "./lib/api";

let _id = 0;
const uid = () => `msg-${Date.now()}-${++_id}`;

const STORAGE_KEY = "agribot.chat.v1";

/** Load persisted history. Blob image URLs don't survive a reload, so we drop
 *  them on load to avoid broken <img> references. */
function loadHistory(): Message[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as Message[];
    if (!Array.isArray(parsed)) return [];
    return parsed.map(m => (m.image_url ? { ...m, image_url: undefined } : m));
  } catch { return []; }
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>(loadHistory);
  const [isLoading, setIsLoading] = useState(false);
  const [pendingVoiceConfirm, setPendingVoiceConfirm] = useState<{ voice: VoiceBlock; traceId: string } | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const { health, status } = useHealth();

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, isLoading]);

  // Persist history. Blob URLs are stripped (not meaningful across reloads).
  useEffect(() => {
    try {
      const serialisable = messages.map(m => (m.image_url ? { ...m, image_url: undefined } : m));
      localStorage.setItem(STORAGE_KEY, JSON.stringify(serialisable));
    } catch { /* quota or serialisation error — non-fatal */ }
  }, [messages]);

  const clearChat = useCallback(() => {
    setMessages([]);
    try { localStorage.removeItem(STORAGE_KEY); } catch { /* ignore */ }
  }, []);

  const addResponse = useCallback((r: ChatResponse, mode: "text" | "voice" | "voice_confirmed" | "image", userMsgId: string, imageUrl?: string) => {
    setMessages(prev => {
      const newMessages = [...prev];
      if (r.parsed_query) {
        const uIdx = newMessages.findIndex(m => m.id === userMsgId);
        if (uIdx >= 0) {
          const prefix = mode === "voice" || mode === "voice_confirmed" ? "🎤 " : mode === "image" ? "📷 " : "";
          newMessages[uIdx] = { ...newMessages[uIdx], content: `${prefix}${r.parsed_query}` };
        }
      }
      return [...newMessages, {
        id: uid(), role: "assistant", content: r.answer, answer_bn: r.answer_bn,
        input_mode: mode, citations: r.citations, kg_entities: r.kg_entities,
        evidence_grade: r.evidence_grade, is_verified: r.is_verified,
        verification_reason: r.verification_reason, retry_count: r.retry_count,
        diagnostics: r.diagnostics, voice: r.voice, image: r.image,
        grounding_action: r.grounding_action, follow_up_suggestions: r.follow_up_suggestions,
        image_url: imageUrl,
        timestamp: Date.now(),
      }];
    });
  }, []);

  const addError = useCallback((msg: string) => {
    setMessages(prev => [...prev, { id: uid(), role: "assistant", content: `⚠️ ${msg}`, timestamp: Date.now() }]);
  }, []);

  const handleText = useCallback(async (q: string) => {
    const uId = uid();
    setMessages(prev => [...prev, { id: uId, role: "user", content: q, input_mode: "text", timestamp: Date.now() }]);
    setIsLoading(true);
    try { addResponse(await sendChat(q), "text", uId); } catch (e: any) { addError(e.message); } finally { setIsLoading(false); }
  }, [addResponse, addError]);

  const handleVoice = useCallback(async (blob: Blob) => {
    const uId = uid();
    setMessages(prev => [...prev, { id: uId, role: "user", content: "🎤 Transcribing voice...", input_mode: "voice", timestamp: Date.now() }]);
    setIsLoading(true);
    try {
      const r = await sendVoice(blob);
      // Check if voice needs confirmation
      if (r.voice?.needs_confirmation) {
        setPendingVoiceConfirm({
          voice: r.voice,
          traceId: r.diagnostics?.trace_id || "",
        });
        // Update user message to show transcript
        setMessages(prev => {
          const msgs = [...prev];
          const idx = msgs.findIndex(m => m.id === uId);
          if (idx >= 0) msgs[idx] = { ...msgs[idx], content: `🎤 "${r.voice!.transcript_suspected || r.voice!.transcript}" (needs confirmation)` };
          return msgs;
        });
      } else {
        addResponse(r, "voice", uId);
      }
    } catch (e: any) { addError(e.message); } finally { setIsLoading(false); }
  }, [addResponse, addError]);

  const handleVoiceConfirm = useCallback(async (editedText: string, traceId: string) => {
    setPendingVoiceConfirm(null);
    const uId = uid();
    setMessages(prev => [...prev, { id: uId, role: "user", content: `🎤✓ ${editedText}`, input_mode: "voice_confirmed", timestamp: Date.now() }]);
    setIsLoading(true);
    try { addResponse(await sendVoiceConfirm(editedText, traceId), "voice_confirmed", uId); } catch (e: any) { addError(e.message); } finally { setIsLoading(false); }
  }, [addResponse, addError]);

  const handleVoiceCancel = useCallback(() => {
    setPendingVoiceConfirm(null);
  }, []);

  const handleImage = useCallback(async (file: File, q: string) => {
    const uId = uid();
    // Persist a thumbnail URL so the image survives in the conversation log
    // (the Composer's own preview URL is revoked on send).
    const thumbUrl = URL.createObjectURL(file);
    setMessages(prev => [...prev, { id: uId, role: "user", content: q ? `📷 ${q}` : `📷 Analyzing: ${file.name}`, input_mode: "image", image_url: thumbUrl, timestamp: Date.now() }]);
    setIsLoading(true);
    try { addResponse(await sendImage(file, q), "image", uId, thumbUrl); } catch (e: any) { addError(e.message); } finally { setIsLoading(false); }
  }, [addResponse, addError]);

  /** Regenerate an assistant answer: drop it, re-send the preceding user text.
   *  Image/voice turns can't be replayed (the binary is gone), so retry is
   *  offered only for text-derived answers. */
  const handleRetry = useCallback(async (assistantId: string) => {
    if (isLoading) return;
    const idx = messages.findIndex(m => m.id === assistantId);
    if (idx <= 0) return;
    let userMsg: Message | undefined;
    for (let i = idx - 1; i >= 0; i--) {
      if (messages[i].role === "user") { userMsg = messages[i]; break; }
    }
    if (!userMsg) return;
    // Strip the modality prefix emoji we prepend to user content for display.
    const query = userMsg.content.replace(/^[🎤📷✓\s]+/u, "").trim();
    if (!query) return;

    setMessages(prev => prev.filter(m => m.id !== assistantId));
    const uId = userMsg.id;
    setIsLoading(true);
    try { addResponse(await sendChat(query), "text", uId); }
    catch (e: any) { addError(e.message); }
    finally { setIsLoading(false); }
  }, [messages, isLoading, addResponse, addError]);

  const hasMessages = messages.length > 0;

  return (
    <div className="h-screen flex flex-col overflow-hidden bg-app">
      <TopNav
        health={health}
        status={status}
        hasMessages={hasMessages}
        onExport={() => exportCaseReport(messages)}
        onClear={clearChat}
        onNew={clearChat}
      />

      <main className="flex-1 min-h-0 relative">
        <div className="h-full overflow-y-auto px-4 lg:px-8 pb-40 pt-4 scroll-smooth">
          {!hasMessages ? (
            <Landing health={health} onPick={handleText} />
          ) : (
            <div className="max-w-3xl mx-auto flex flex-col gap-6 pt-4">
              <AnimatePresence initial={false}>
                {messages.map(m => (
                  <MessageBubble key={m.id} message={m} onFollowUp={handleText} onRetry={handleRetry} />
                ))}
              </AnimatePresence>

              <AnimatePresence>
                {isLoading && (
                  <motion.div
                    className="flex gap-3"
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                  >
                    <div className="w-9 h-9 rounded-xl bg-card border border-line flex items-center justify-center text-sm shrink-0 shadow-[var(--shadow-card)]">🌾</div>
                    <div className="flex items-center gap-2 px-4 py-3">
                      {[0, 1, 2].map(i => <div key={i} className="w-2 h-2 rounded-full bg-brand-500 animate-typing" style={{ animationDelay: `${i * 0.2}s` }} />)}
                      <span className="text-xs text-ink-3 ml-2">Searching &amp; reasoning…</span>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              <AnimatePresence>
                {pendingVoiceConfirm && (
                  <VoiceConfirmBanner
                    transcript={pendingVoiceConfirm.voice.transcript_suspected || pendingVoiceConfirm.voice.transcript}
                    confidence={pendingVoiceConfirm.voice.asr_confidence}
                    language={pendingVoiceConfirm.voice.asr_language}
                    warnings={pendingVoiceConfirm.voice.asr_warnings}
                    suggestedActions={pendingVoiceConfirm.voice.suggested_actions}
                    traceId={pendingVoiceConfirm.traceId}
                    onConfirm={handleVoiceConfirm}
                    onCancel={handleVoiceCancel}
                  />
                )}
              </AnimatePresence>

              <div ref={endRef} />
            </div>
          )}
        </div>

        <Composer onText={handleText} onVoice={handleVoice} onImage={handleImage} disabled={isLoading} />
      </main>
    </div>
  );
}
