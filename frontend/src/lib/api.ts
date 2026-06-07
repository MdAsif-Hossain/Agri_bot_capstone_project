/**
 * AgriBot API Client — typed fetch for /v1/* endpoints.
 *
 * Response types mirror the nested ChatResponseV1 schema.
 */

// --- Nested block types ---

export interface DiagnosticsBlock {
    trace_id: string;
    timings_ms: Record<string, number>;
    mode_flags: string[];
    warnings: string[];
}

export interface VoiceBlock {
    transcript: string;
    asr_language: string;
    asr_confidence: number;
    asr_warnings: string[];
    needs_confirmation: boolean;
    transcript_suspected: string;
    suggested_actions: string[];
}

export interface PossibleCondition {
    label: string;
    confidence: number;
}

/** Full structured result the backend nests under image.analysis_summary. */
export interface AnalysisSummary {
    pipeline_used?: string;
    ocr_text?: string;
    symptom_hints?: string[];
    quality_flags?: string[];
    limitations?: string[];
    keywords?: string[];
    possible_conditions?: PossibleCondition[];
}

export interface ImageBlock {
    pipeline_used: "ocr_baseline" | "classifier_assisted" | "ocr_fallback";
    analysis_summary: AnalysisSummary;
    limitations: string[];
    possible_conditions: PossibleCondition[];
}

// --- Main response ---

export interface ChatResponse {
    answer: string;
    answer_bn: string;
    citations: string[];
    kg_entities: KGEntity[];
    evidence_grade: string;
    is_verified: boolean;
    verification_reason: string;
    retry_count: number;
    input_mode: string;
    grounding_action: string;
    follow_up_suggestions: string[];
    parsed_query?: string;
    // Nested blocks
    diagnostics: DiagnosticsBlock;
    voice?: VoiceBlock | null;
    image?: ImageBlock | null;
}

export interface KGEntity {
    bn: string;
    en: string;
    type: string;
    id?: number;
    canonical_bn?: string;
    canonical_en?: string;
    entity_type?: string;
    aliases?: string[];
}

export interface HealthResponse {
    status: string;
    chunk_count: number;
    kg_entities: number;
    kg_aliases: number;
    kg_relations: number;
    manifest: Record<string, unknown> | null;
    enabled_modules: Record<string, boolean>;
    grounding_mode: string;
}

export interface Message {
    id: string;
    role: "user" | "assistant";
    content: string;
    answer_bn?: string;
    input_mode?: "text" | "voice" | "voice_confirmed" | "image";
    citations?: string[];
    kg_entities?: KGEntity[];
    evidence_grade?: string;
    is_verified?: boolean;
    verification_reason?: string;
    retry_count?: number;
    grounding_action?: string;
    follow_up_suggestions?: string[];
    timestamp: number;
    // Wall-clock time the system took to answer, in milliseconds (client-side
    // round-trip: request sent -> response received). Shown in the UI.
    elapsed_ms?: number;
    // Local object URL of the user's uploaded image (client-side only).
    image_url?: string;
    // Nested diagnostics
    diagnostics?: DiagnosticsBlock;
    voice?: VoiceBlock | null;
    image?: ImageBlock | null;
}

const API = import.meta.env.VITE_API_URL || "";

async function json<T>(path: string, init?: RequestInit): Promise<T> {
    const r = await fetch(`${API}${path}`, { ...init, headers: { "Content-Type": "application/json", ...init?.headers } });
    if (!r.ok) { const e = await r.json().catch(() => ({ detail: r.statusText })); throw new Error(e.detail || `API ${r.status}`); }
    return r.json();
}

export const sendChat = (query: string, inputMode = "text", traceId = "") =>
    json<ChatResponse>("/v1/chat", {
        method: "POST",
        body: JSON.stringify({ query, input_mode: inputMode, trace_id: traceId }),
    });

/** Send confirmed/edited voice transcript as text query. */
export const sendVoiceConfirm = (text: string, traceId: string) =>
    sendChat(text, "voice_confirmed", traceId);

export async function sendVoice(blob: Blob): Promise<ChatResponse> {
    const ext = blob.type.includes("webm")
        ? "webm"
        : blob.type.includes("ogg")
            ? "ogg"
            : blob.type.includes("mpeg") || blob.type.includes("mp3")
                ? "mp3"
                : blob.type.includes("wav")
                    ? "wav"
                    : "bin";
    const f = new FormData();
    f.append("audio", blob, `recording.${ext}`);
    const r = await fetch(`${API}/v1/chat/voice`, { method: "POST", body: f });
    if (!r.ok) throw new Error((await r.json().catch(() => ({ detail: "" }))).detail || `API ${r.status}`);
    return r.json();
}

export async function sendImage(file: File, query?: string): Promise<ChatResponse> {
    const f = new FormData(); f.append("image", file); if (query) f.append("query", query);
    const r = await fetch(`${API}/v1/chat/image`, { method: "POST", body: f });
    if (!r.ok) throw new Error((await r.json().catch(() => ({ detail: "" }))).detail || `API ${r.status}`);
    return r.json();
}

export const getHealth = () => json<HealthResponse>("/v1/health");

export async function getTTSAudio(text: string, language: "en" | "bn"): Promise<string> {
    const r = await fetch(`${API}/v1/tts`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, language }) });
    if (!r.ok) {
        const err = await r.json().catch(() => ({} as { detail?: string }));
        throw new Error(err.detail || `TTS error (${r.status})`);
    }
    return URL.createObjectURL(await r.blob());
}

export function exportCaseReport(messages: Message[]): void {
    const report = messages.filter(m => m.role === "assistant").map(m => ({
        query: messages.find(u => u.timestamp < m.timestamp && u.role === "user")?.content || "",
        answer: m.content,
        answer_bn: m.answer_bn || "",
        citations: m.citations || [],
        kg_entities: m.kg_entities || [],
        evidence_grade: m.evidence_grade || "",
        is_verified: m.is_verified || false,
        diagnostics: m.diagnostics || {},
        grounding_action: m.grounding_action || "",
    }));
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = `agribot_report_${Date.now()}.json`; a.click();
    URL.revokeObjectURL(url);
}
