"""Fig. 1 (journal version) — system architecture as five stacked layers.

Legibility rule for an IEEE double-column figure (7.16 in wide): the widest
row must stay under ~70 characters, so labels are short and the agent's
internal loop is left to Fig. 3. Every component named here exists in the
repository. GPU-resident node is orange, CPU nodes green, data stores grey.

Run:  python scripts/fig1_architecture_v2.py   (needs the `dot` binary)
Writes fig1.pdf / fig1.png in the current directory.
"""
import graphviz

CPU = "#e3f1e3"
GPU = "#fbe0cf"
DATA = "#ececec"
UI = "#dfe9f7"
API = "#fdf1cf"
AGENT = "#e8dff5"
FRAME = "#7a8aa6"
A = "→"    # arrow
D = "·"    # middle dot


def build() -> graphviz.Digraph:
    g = graphviz.Digraph("agribot_arch", format="pdf", engine="dot")
    g.attr(rankdir="TB", bgcolor="white", nodesep="0.35", ranksep="0.45",
           fontname="Helvetica", fontsize="12", newrank="true", splines="spline")
    g.attr("node", shape="box", style="rounded,filled", fontname="Helvetica",
           fontsize="11", color="#444444", penwidth="1.1", margin="0.12,0.07")
    g.attr("edge", fontname="Helvetica", fontsize="9", color="#444444",
           arrowsize="0.7", penwidth="1.0")

    def layer(name, label, nodes):
        with g.subgraph(name=f"cluster_{name}") as c:
            c.attr(label=label, style="dashed", color=FRAME, fontsize="12",
                   labeljust="l", rank="same")
            for nid, text, fill, extra in nodes:
                c.node(nid, text, fillcolor=fill, **extra)

    layer("client", "1  Client (browser)", [
        ("user", f"Farmer / extension officer\ntext {D} voice {D} leaf photo", UI, {}),
    ])
    layer("api", "2  FastAPI gateway", [
        ("api", "REST endpoints\n/v1/chat, /chat/voice,\n/chat/image, /tts", API, {}),
        ("mw", "rate limit, API key,\nsemaphores (LLM 1, STT 2,\nimage 2), telemetry", API, {}),
    ])
    layer("pre", "3  Per-request preprocessors (CPU)", [
        ("stt", f"Speech-to-text cascade\nfaster-whisper {A} Vosk\n{A} BanglaSpeech2Text\n+ confidence gate", CPU, {}),
        ("vision", f"Image pipeline\nvegetation gate {A}\nMobileNetV3-L + CBAM\n+ Tesseract OCR", CPU, {}),
        ("bn2en", f"Input translation\nBanglaT5 BN{A}EN", CPU, {}),
    ])
    layer("agent", "4  LangGraph agent (Fig. 3)", [
        ("agent", f"Bounded agentic loop\nnormalize {A} kg_link {A} retrieve {A} rerank {A} grade\n"
                  f"{A} [rewrite, at most 2] {A} generate {A} translate {A} verify {A} enforce_policy",
         AGENT, {"penwidth": "1.6"}),
    ])
    layer("models", "Models called by the agent", [
        ("llm", "Generator (GPU)\nQwen2.5-3B Q4_0\nllama.cpp, 36 layers", GPU, {"penwidth": "1.6"}),
        ("retr", f"Hybrid retrieval\nFAISS + BM25 {A} RRF\n{A} cross-encoder", CPU, {}),
        ("kg", "KG entity linker\n28 entities,\n84 aliases", CPU, {}),
        ("en2bn", f"Output translation\nAgriBanglaT5\nEN{A}BN", CPU, {}),
        ("tts", "Text-to-speech\npyttsx3\n(system voice)", CPU, {}),
    ])
    layer("data", "5  Local data stores (built offline by ingest.py)", [
        ("idx", "FAISS + BM25 indexes\n914 chunks, 10 PDFs", DATA, {"shape": "cylinder"}),
        ("kgdb", "knowledge_graph.db\n(SQLite)", DATA, {"shape": "cylinder"}),
        ("tel", "telemetry.db", DATA, {"shape": "cylinder"}),
        ("ingest", f"PDF {A} Marker / OCR\n{A} chunk {A} MiniLM embed", DATA, {"shape": "note"}),
    ])

    # Flows
    g.edge("user", "api", label="HTTP JSON / multipart")
    g.edge("api", "mw", style="dashed", arrowhead="none")
    g.edge("api", "stt", label="audio")
    g.edge("api", "vision", label="image")
    g.edge("api", "bn2en", label="Bengali text")
    g.edge("stt", "bn2en", label="transcript")
    g.edge("vision", "agent", label="disease label + OCR text")
    g.edge("bn2en", "agent", label="English query")
    g.edge("api", "agent", label="English text", constraint="false")
    for m in ("llm", "retr", "kg", "en2bn"):
        g.edge("agent", m, dir="both")
    g.edge("agent", "api", label="EN + BN answer, citations,\nverdict, refusal / disclaimer",
           color="#b03030", fontcolor="#b03030", constraint="false")
    g.edge("api", "tts", label="/v1/tts", style="dashed", constraint="false")
    g.edge("retr", "idx", style="dotted", arrowhead="none")
    g.edge("kg", "kgdb", style="dotted", arrowhead="none")
    g.edge("mw", "tel", style="dotted", arrowhead="none", constraint="false")
    g.edge("ingest", "idx", label="builds", constraint="false")
    # keep the model row in a sensible order
    with g.subgraph() as s:
        s.attr(rank="same")
        s.edge("llm", "retr", style="invis")
        s.edge("retr", "kg", style="invis")
        s.edge("kg", "en2bn", style="invis")
        s.edge("en2bn", "tts", style="invis")
    return g


if __name__ == "__main__":
    g = build()
    g.render("fig1", cleanup=True, format="pdf")
    g.render("fig1", cleanup=True, format="png")
    print("wrote fig1.pdf / fig1.png")
