# Colab cell: regenerate fig1 (architecture) + fig2 (workflow) with the
# corrected model name (Qwen2.5-3B Q4_0). Writes fig1/fig2 .png + .pdf and
# downloads them. Paste this whole thing into one Colab cell and run.

import subprocess
import sys
subprocess.run("apt-get -qq install -y graphviz", shell=True)
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "graphviz"])

from graphviz import Digraph

PALETTE = {"bg_cpu": "#e6f2e6", "bg_gpu": "#fde9d9"}


def fig1_architecture(out="fig1"):
    g = Digraph("arch", format="pdf")
    g.attr(rankdir="TB", bgcolor="white", fontname="Times", fontsize="10",
           nodesep="0.25", ranksep="0.45", compound="true")
    g.attr("node", shape="box", style="filled,rounded", fontname="Times", fontsize="9")
    g.attr("edge", fontname="Times", fontsize="8", arrowsize="0.6")

    with g.subgraph(name="cluster_client") as c:
        c.attr(label="Client layer", style="dashed", color="gray50", fontsize="9")
        c.node("mobile", "Mobile app\n(Capacitor)", fillcolor="#dde8f5")
        c.node("web", "Web UI\n(Vite + React)", fillcolor="#dde8f5")

    with g.subgraph(name="cluster_api") as c:
        c.attr(label="API layer (FastAPI)", style="dashed", color="gray50", fontsize="9")
        c.node("middleware", "Middlewares:\nrate-limit · API-key · telemetry", fillcolor="#fde7d3")
        c.node("api", "REST endpoints\n/v1/chat · /chat/voice · /chat/image · /tts", fillcolor="#fde7d3")

    with g.subgraph(name="cluster_pre") as c:
        c.attr(label="Multimodal preprocessors (per-request, CPU)", style="dashed", color="gray50", fontsize="9")
        c.node("stt", "STT pipeline\nfaster-whisper →\nBanglaSpeech2Text →\nVosk (fallback)", fillcolor=PALETTE["bg_cpu"])
        c.node("imgproc", "Image processor\nPIL/NumPy heuristics\n+ Tesseract OCR", fillcolor=PALETTE["bg_cpu"])
        c.node("classifier", "MobileNetV3+CBAM\n(ONNX)", fillcolor=PALETTE["bg_cpu"])

    g.node("agent", "LangGraph agent (Fig. 2)\nnormalize → kg_link → retrieve →\nrerank → grade ↻ → generate →\ntranslate → verify → enforce_policy",
           fillcolor="#f6e6f5", shape="box3d")

    with g.subgraph(name="cluster_services") as c:
        c.attr(label="Inference services (per-node calls from the agent)", style="dashed", color="gray50", fontsize="9")
        c.node("llm", "Qwen2.5-3B (Q4_0 GGUF)\n— GPU —", fillcolor=PALETTE["bg_gpu"])
        c.node("retrv", "Hybrid retrieval\nFAISS dense + BM25 sparse\n+ RRF fusion + Cross-encoder", fillcolor=PALETTE["bg_cpu"])
        c.node("kg", "KG entity linker\n(SQLite)", fillcolor=PALETTE["bg_cpu"])
        c.node("trans", "BanglaT5 EN↔BN\n(2 models, CPU)", fillcolor=PALETTE["bg_cpu"])

    g.node("tts", "TTS\npyttsx3 (system voice)", fillcolor=PALETTE["bg_cpu"])

    with g.subgraph(name="cluster_data") as c:
        c.attr(label="Persistent data (offline)", style="dashed", color="gray50", fontsize="9")
        c.node("pdfs", "Agricultural\nPDFs", shape="folder", fillcolor="#ececec")
        c.node("indexes", "FAISS + BM25 +\nchunks.json", shape="cylinder", fillcolor="#ececec")
        c.node("kgdb", "knowledge_graph.db", shape="cylinder", fillcolor="#ececec")
        c.node("telem", "telemetry.db", shape="cylinder", fillcolor="#ececec")

    g.edge("mobile", "api", label="HTTPS / JSON")
    g.edge("web", "api", label="HTTPS / JSON")
    g.edge("middleware", "api", style="dashed", arrowhead="none", label="all routes", fontsize="7", color="dimgray")
    g.edge("api", "stt", label="voice path", fontsize="7")
    g.edge("api", "imgproc", label="image path", fontsize="7")
    g.edge("imgproc", "classifier", style="dashed", label="optional", fontsize="7")
    g.edge("stt", "agent", label="transcribed text", fontsize="7", color="dimgray")
    g.edge("imgproc", "agent", label="combined query +\nimage analysis block", fontsize="7", color="dimgray")
    g.edge("api", "agent", label="text path", fontsize="7")
    g.edge("api", "tts", label="/v1/tts", style="dashed", fontsize="7")
    g.edge("agent", "llm", label="grade / rewrite / generate / verify", fontsize="7")
    g.edge("agent", "retrv", label="retrieve + rerank", fontsize="7")
    g.edge("agent", "kg", label="entity link + expand", fontsize="7")
    g.edge("agent", "trans", label="normalize +\ntranslate answer", fontsize="7")
    g.edge("retrv", "indexes", style="dotted")
    g.edge("kg", "kgdb", style="dotted")
    g.edge("indexes", "pdfs", style="dotted", label="built by ingest.py", fontsize="7")
    g.edge("middleware", "telem", style="dotted", label="every request", fontsize="7")

    g.render(out, cleanup=True, format="pdf")
    g.render(out, cleanup=True, format="png")
    print("wrote", out + ".png /.pdf")


def fig2_workflow(out="fig2"):
    g = Digraph("workflow", format="pdf")
    g.attr(rankdir="LR", bgcolor="white", fontname="Times", fontsize="10")
    g.attr("node", shape="box", style="filled,rounded", fontname="Times", fontsize="9", fillcolor="#dde8f5")
    g.attr("edge", fontname="Times", fontsize="8", arrowsize="0.6")

    g.node("start", "Input", shape="ellipse", fillcolor="#cccccc")
    g.node("normalize", "normalize")
    g.node("kg_link", "kg_link")
    g.node("retrieve", "retrieve\n(FAISS + BM25)")
    g.node("rerank", "rerank\n(cross-encoder)")
    g.node("grade", "grade\n(LLM)", shape="diamond", fillcolor="#fff2cc")
    g.node("rewrite", "rewrite\n(LLM)", fillcolor="#f5dada")
    g.node("generate", "generate\n(Qwen2.5-3B, GPU)")
    g.node("translate", "translate\n(BanglaT5)")
    g.node("verify", "verify\n(LLM)", fillcolor="#e0f0e0")
    g.node("enforce", "enforce_policy\n(safety)", fillcolor="#e0f0e0")
    g.node("end", "Output", shape="ellipse", fillcolor="#cccccc")

    g.edge("start", "normalize")
    g.edge("normalize", "kg_link")
    g.edge("kg_link", "retrieve")
    g.edge("retrieve", "rerank")
    g.edge("rerank", "grade")
    g.edge("grade", "rewrite", label="INSUFFICIENT\n(retry < 2)")
    g.edge("rewrite", "retrieve", style="dashed", constraint="false")
    g.edge("grade", "generate", label="SUFFICIENT\nor max retries")
    g.edge("generate", "translate")
    g.edge("translate", "verify")
    g.edge("verify", "enforce")
    g.edge("enforce", "end")

    g.render(out, cleanup=True, format="pdf")
    g.render(out, cleanup=True, format="png")
    print("wrote", out + ".png /.pdf")


fig1_architecture("fig1")
fig2_workflow("fig2")

# Download (Colab). Remove this block if running locally.
try:
    from google.colab import files
    for f in ("fig1.png", "fig1.pdf", "fig2.png", "fig2.pdf"):
        files.download(f)
except Exception:
    pass
