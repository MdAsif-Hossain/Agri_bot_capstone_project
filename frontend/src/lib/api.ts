/**
 * AgriBot API Client — typed fetch for /v1/* endpoints.
 */

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
    trace_id: string;
    timings_ms: Record<string, number>;
    grounding_action: string;
    follow_up_suggestions: string[];
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
    input_mode?: "text" | "voice" | "image";
    citations?: string[];
    kg_entities?: KGEntity[];
    evidence_grade?: string;
    is_verified?: boolean;
    verification_reason?: string;
    retry_count?: number;
    trace_id?: string;
    timings_ms?: Record<string, number>;
    grounding_action?: string;
    follow_up_suggestions?: string[];
    timestamp: number;
}

const API = import.meta.env.VITE_API_URL || "";

async function json<T>(path: string, init?: RequestInit): Promise<T> {
    const r = await fetch(`${API}${path}`, { ...init, headers: { "Content-Type": "application/json", ...init?.headers } });
    if (!r.ok) { const e = await r.json().catch(() => ({ detail: r.statusText })); throw new Error(e.detail || `API ${r.status}`); }
    return r.json();
}

export const sendChat = (query: string) => json<ChatResponse>("/v1/chat", { method: "POST", body: JSON.stringify({ query }) });

export async function sendVoice(blob: Blob): Promise<ChatResponse> {
    const f = new FormData(); f.append("audio", blob, "recording.wav");
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
    if (!r.ok) throw new Error("TTS error");
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
        trace_id: m.trace_id || "",
        timings_ms: m.timings_ms || {},
        grounding_action: m.grounding_action || "",
    }));
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = `agribot_report_${Date.now()}.json`; a.click();
    URL.revokeObjectURL(url);
}
