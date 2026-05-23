"""Generate all paper figures from a single source.

Produces 7 figures as vector PDFs (IEEE-ready) into docs/figures/.

Tools used:
  - matplotlib for all quantitative charts (Figs 3, 4, 5)
  - graphviz for all diagrams (Figs 1, 2, 6)
  - PIL for the screenshot triptych (Fig 7)

To run locally:
    pip install matplotlib numpy graphviz Pillow
    # System graphviz: choco install graphviz (Windows) | brew install graphviz (Mac) | apt install graphviz (Linux)
    python scripts/make_figures.py

To run in Colab:
    !apt install -y graphviz
    !pip install graphviz matplotlib numpy Pillow
    # Then paste each FIGURE_* section as a cell

Regenerate one figure: comment out the others at the bottom of this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# SETUP — IEEE-quality defaults applied to every matplotlib figure
# =============================================================================
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
        "grid.linewidth": 0.4,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,  # TrueType — fully editable, no font embedding warnings
        "ps.fonttype": 42,
    }
)

# Color-blind-safe palette (sequential blue + accent orange)
PALETTE = {
    "primary": "#1f4e79",
    "accent": "#e07a1f",
    "muted": "#7d7d7d",
    "good": "#2c7a2c",
    "warn": "#b54a4a",
    "bg_cpu": "#e6f2e6",
    "bg_gpu": "#fde9d9",
}

# IEEE column widths
COL_WIDTH_IN = 3.5  # single column
WIDE_WIDTH_IN = 7.16  # spans both columns

# Project paths
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR = ROOT / "evaluation" / "results"


def _save(fig, name: str) -> Path:
    out_pdf = OUT_DIR / f"{name}.pdf"
    out_png = OUT_DIR / f"{name}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"  ✓ {out_pdf.relative_to(ROOT)}")
    return out_pdf


# =============================================================================
# FIGURE 1 — System architecture (graphviz)
# =============================================================================
def fig1_architecture():
    from graphviz import Digraph

    g = Digraph("arch", format="pdf")
    g.attr(rankdir="TB", bgcolor="white", fontname="Times", fontsize="10",
           nodesep="0.25", ranksep="0.45", compound="true")
    g.attr("node", shape="box", style="filled,rounded", fontname="Times", fontsize="9")
    g.attr("edge", fontname="Times", fontsize="8", arrowsize="0.6")

    # --- Client layer ---
    with g.subgraph(name="cluster_client") as c:
        c.attr(label="Client layer", style="dashed", color="gray50", fontsize="9")
        c.node("mobile", "Mobile app\n(Capacitor)", fillcolor="#dde8f5")
        c.node("web", "Web UI\n(Vite + React)", fillcolor="#dde8f5")

    # --- API layer (middlewares + endpoints) ---
    with g.subgraph(name="cluster_api") as c:
        c.attr(label="API layer (FastAPI)", style="dashed", color="gray50", fontsize="9")
        c.node("middleware", "Middlewares:\nrate-limit · API-key · telemetry", fillcolor="#fde7d3")
        c.node("api", "REST endpoints\n/v1/chat · /chat/voice · /chat/image · /tts",
               fillcolor="#fde7d3")

    # --- Multimodal preprocessors (run BEFORE the agent on voice/image paths) ---
    with g.subgraph(name="cluster_pre") as c:
        c.attr(label="Multimodal preprocessors (per-request, CPU)",
               style="dashed", color="gray50", fontsize="9")
        c.node("stt", "STT pipeline\nfaster-whisper →\nBanglaSpeech2Text →\nVosk (fallback)",
               fillcolor=PALETTE["bg_cpu"])
        c.node("imgproc", "Image processor\nPIL/NumPy heuristics\n+ Tesseract OCR",
               fillcolor=PALETTE["bg_cpu"])
        c.node("classifier", "MobileNetV3+CBAM\n(ONNX)", fillcolor=PALETTE["bg_cpu"])

    # --- Agent layer (the orchestrator) ---
    g.node("agent", "LangGraph agent (Fig. 2)\nnormalize → kg_link → retrieve →\nrerank → grade ↻ → generate →\ntranslate → verify → enforce_policy",
           fillcolor="#f6e6f5", shape="box3d")

    # --- Inference services (called BY the agent's nodes) ---
    with g.subgraph(name="cluster_services") as c:
        c.attr(label="Inference services (per-node calls from the agent)",
               style="dashed", color="gray50", fontsize="9")
        c.node("llm", "Qwen2.5-1.5B (Q8 GGUF)\n— GPU —", fillcolor=PALETTE["bg_gpu"])
        c.node("retrv", "Hybrid retrieval\nFAISS dense + BM25 sparse\n+ RRF fusion + Cross-encoder",
               fillcolor=PALETTE["bg_cpu"])
        c.node("kg", "KG entity linker\n(SQLite)", fillcolor=PALETTE["bg_cpu"])
        c.node("trans", "BanglaT5 EN↔BN\n(2 models, CPU)", fillcolor=PALETTE["bg_cpu"])

    # --- Output service (called directly by /v1/tts, not the agent) ---
    g.node("tts", "TTS\npyttsx3 (system voice)", fillcolor=PALETTE["bg_cpu"])

    # --- Persistent data ---
    with g.subgraph(name="cluster_data") as c:
        c.attr(label="Persistent data (offline)", style="dashed", color="gray50", fontsize="9")
        c.node("pdfs", "Agricultural\nPDFs", shape="folder", fillcolor="#ececec")
        c.node("indexes", "FAISS + BM25 +\nchunks.json", shape="cylinder", fillcolor="#ececec")
        c.node("kgdb", "knowledge_graph.db", shape="cylinder", fillcolor="#ececec")
        c.node("telem", "telemetry.db", shape="cylinder", fillcolor="#ececec")

    # --- Edges: client → api ---
    g.edge("mobile", "api", label="HTTPS / JSON")
    g.edge("web", "api", label="HTTPS / JSON")
    g.edge("middleware", "api", style="dashed", arrowhead="none",
           label="all routes", fontsize="7", color="gray40")

    # --- Edges: api → preprocessors → agent (per modality) ---
    g.edge("api", "stt", label="voice path", fontsize="7")
    g.edge("api", "imgproc", label="image path", fontsize="7")
    g.edge("imgproc", "classifier", style="dashed", label="optional", fontsize="7")
    g.edge("stt", "agent", label="transcribed text", fontsize="7", color="gray40")
    g.edge("imgproc", "agent", label="combined query +\nimage analysis block",
           fontsize="7", color="gray40")
    g.edge("api", "agent", label="text path", fontsize="7")

    # --- Edges: api → tts (separate output endpoint) ---
    g.edge("api", "tts", label="/v1/tts", style="dashed", fontsize="7")

    # --- Edges: agent → services ---
    g.edge("agent", "llm", label="grade / rewrite / generate / verify", fontsize="7")
    g.edge("agent", "retrv", label="retrieve + rerank", fontsize="7")
    g.edge("agent", "kg", label="entity link + expand", fontsize="7")
    g.edge("agent", "trans", label="normalize +\ntranslate answer", fontsize="7")

    # --- Edges: services → data ---
    g.edge("retrv", "indexes", style="dotted")
    g.edge("kg", "kgdb", style="dotted")
    g.edge("indexes", "pdfs", style="dotted", label="built by ingest.py", fontsize="7")
    g.edge("middleware", "telem", style="dotted", label="every request", fontsize="7")

    out = OUT_DIR / "fig1_architecture"
    g.render(filename=out.name, directory=out.parent, cleanup=True, format="pdf")
    g.render(filename=out.name, directory=out.parent, cleanup=True, format="png")
    print(f"  ✓ {out.relative_to(ROOT)}.pdf")


# =============================================================================
# FIGURE 2 — LangGraph agentic workflow (graphviz)
# =============================================================================
def fig2_workflow():
    from graphviz import Digraph

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
    g.node("generate", "generate\n(LLM)")
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

    out = OUT_DIR / "fig2_workflow"
    g.render(filename=out.name, directory=out.parent, cleanup=True, format="pdf")
    g.render(filename=out.name, directory=out.parent, cleanup=True, format="png")
    print(f"  ✓ {out.relative_to(ROOT)}.pdf")


# =============================================================================
# FIGURE 3 — Retrieval Recall@k bar chart (reads eval JSON)
# =============================================================================
def fig3_retrieval():
    path = EVAL_DIR / "retrieval_latest.json"
    if not path.exists():
        # Placeholder data so the figure exists during drafting
        print(f"  ⚠ {path.relative_to(ROOT)} not found — using PLACEHOLDER values")
        data = {"summary": {"recall_at_k": {"1": 0.0, "3": 0.0, "5": 0.0, "10": 0.0}, "mrr": 0.0}}
    else:
        data = json.loads(path.read_text(encoding="utf-8"))

    rk = data["summary"]["recall_at_k"]
    # JSON keys are always strings; the eval script writes int keys that get
    # serialized to "1", "3", ... here. Cast safely.
    ks = sorted(int(k) for k in rk)
    vals = [rk[str(k)] for k in ks]
    mrr = data["summary"].get("mrr", 0.0)
    avg_lat = data["summary"].get("avg_retrieval_latency_ms", None)
    n_queries = data["summary"].get("queries_evaluated", None)

    fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, 2.4))
    bars = ax.bar([f"k={k}" for k in ks], vals, color=PALETTE["primary"], edgecolor="black", linewidth=0.4)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("Recall@k")
    ax.set_ylim(0, max(1.05, max(vals) * 1.15 if vals else 1.0))

    # Stats badge: MRR + N + avg latency
    badge_lines = [f"MRR = {mrr:.3f}"]
    if n_queries is not None:
        badge_lines.append(f"N = {n_queries}")
    if avg_lat is not None:
        badge_lines.append(f"avg = {avg_lat:.0f} ms")
    ax.text(0.98, 0.95, "\n".join(badge_lines), transform=ax.transAxes, ha="right", va="top",
            fontsize=7, bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                   edgecolor="gray", linewidth=0.4))
    ax.set_axisbelow(True)
    _save(fig, "fig3_retrieval")


# =============================================================================
# FIGURE 4 — Per-stage latency stacked horizontal bar (reads eval JSON)
# =============================================================================
def fig4_latency():
    path = EVAL_DIR / "latency_latest.json"
    if not path.exists():
        print(f"  ⚠ {path.relative_to(ROOT)} not found — using PLACEHOLDER values")
        # Representative full-pipeline shape — replaced by real numbers after
        # `python evaluation/eval_latency.py` runs against the live backend.
        stages_ms = {
            "normalize": 5,
            "kg_link": 8,
            "retrieve": 14,
            "rerank": 240,
            "grade": 800,
            "rewrite": 1200,
            "generate": 15700,
            "translate": 6200,
            "verify": 1100,
        }
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        stages_ms = data["summary"]["stage_means_ms"]

    # Order stages by their position in the pipeline
    pipeline_order = ["normalize", "kg_link", "retrieve", "rerank", "grade", "rewrite",
                      "generate", "translate", "verify", "image_analysis"]
    ordered = [(k, stages_ms.get(k, 0)) for k in pipeline_order if k in stages_ms and stages_ms[k] > 0]
    # Append any unknown stages (future-proof against new agent nodes)
    extra = sorted(
        (k, v) for k, v in stages_ms.items()
        if k not in pipeline_order and k != "total" and isinstance(v, (int, float)) and v > 0
    )
    if extra:
        print(f"  ℹ Including non-standard stage(s): {[k for k, _ in extra]}")
    ordered.extend(extra)
    names = [k for k, _ in ordered]
    vals = [v for _, v in ordered]
    total = sum(vals)

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(names)))

    fig, ax = plt.subplots(figsize=(WIDE_WIDTH_IN, 1.4))
    left = 0.0
    for name, v, c in zip(names, vals, colors):
        ax.barh(0, v, left=left, color=c, edgecolor="white", linewidth=0.6, label=name)
        pct = v / total * 100 if total else 0
        if pct >= 4:
            ax.text(left + v / 2, 0, f"{name}\n{pct:.0f}%", ha="center", va="center", fontsize=7, color="white")
        left += v
    ax.set_xlim(0, total)
    ax.set_yticks([])
    ax.set_xlabel(f"Cumulative latency (ms) — total = {total / 1000:.1f} s")
    ax.grid(False)
    ax.spines[["top", "left", "right"]].set_visible(False)
    _save(fig, "fig4_latency")


# =============================================================================
# FIGURE 5 — CBAM CNN confusion matrix (you may also export from notebook)
# =============================================================================
def fig5_confusion():
    """Reads docs/cnn_confusion.json if present (6×6 ints), else uses placeholder."""
    path = ROOT / "docs" / "cnn_confusion.json"
    labels = ["Bact. Leaf\nBlight", "Brown\nSpot", "Healthy", "Leaf\nBlast", "Leaf\nScald", "Sheath\nBlight"]
    if path.exists():
        cm = np.array(json.loads(path.read_text(encoding="utf-8")), dtype=float)
    else:
        print(f"  ⚠ {path.relative_to(ROOT)} not found — using PLACEHOLDER matrix")
        cm = np.eye(6) * 90 + np.random.RandomState(0).randint(0, 4, (6, 6))

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid div/0 if a class had no test samples
    cm_norm = cm / row_sums
    fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, COL_WIDTH_IN * 0.95))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm_norm[i, j]
            ax.text(j, i, f"{v * 100:.0f}%", ha="center", va="center", fontsize=6,
                    color="white" if v > 0.5 else "black")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized fraction (per-class recall)", fontsize=7)
    cbar.ax.tick_params(labelsize=7)
    _save(fig, "fig5_confusion")


# =============================================================================
# FIGURE 6 — Knowledge graph dialect expansion example (graphviz)
# =============================================================================
def fig6_kg_example():
    """KG dialect-expansion example.
    Aliases shown are real entries from agribot/knowledge_graph/seed_data.py.
    Neighbor entities (symptom/chemical) are illustrative — edit to match
    your final KG content if those specific relations aren't seeded yet.
    """
    from graphviz import Digraph

    g = Digraph("kg", format="pdf")
    g.attr(rankdir="TB", bgcolor="white", fontname="Times", fontsize="10")
    g.attr("node", shape="box", style="filled,rounded", fontname="Times", fontsize="9")
    g.attr("edge", fontname="Times", fontsize="8", arrowsize="0.6")

    # Dialect / colloquial alias (entry point) — REAL seeded alias
    g.node(
        "alias_a",
        "গাছফড়িং\n(colloquial)",
        fillcolor="#fff2cc",
        shape="note",
    )
    g.node(
        "alias_b",
        "brown planthopper\n(English)",
        fillcolor="#fff2cc",
        shape="note",
    )

    # Canonical entity — real seed_data.py entry (bph)
    g.node(
        "canon",
        "বাদামি গাছফড়িং\nBrown Planthopper\n(canonical · type=pest)",
        fillcolor="#dde8f5",
    )

    # Symptom + treatment neighbors (illustrative)
    g.node("sym1", "Hopper burn\n(symptom)", fillcolor="#f6e6f5")
    g.node("sym2", "Yellowing of leaves\n(symptom)", fillcolor="#f6e6f5")
    g.node("treat1", "Imidacloprid\n(chemical)", fillcolor="#fde7d3")
    g.node("treat2", "Resistant variety: BR3\n(cultivar)", fillcolor="#fde7d3")

    g.edge("alias_a", "canon", label="alias_of")
    g.edge("alias_b", "canon", label="alias_of")
    g.edge("canon", "sym1", label="causes")
    g.edge("canon", "sym2", label="causes")
    g.edge("canon", "treat1", label="controlled_by")
    g.edge("canon", "treat2", label="controlled_by")

    out = OUT_DIR / "fig6_kg_example"
    g.render(filename=out.name, directory=out.parent, cleanup=True, format="pdf")
    g.render(filename=out.name, directory=out.parent, cleanup=True, format="png")
    print(f"  ✓ {out.relative_to(ROOT)}.pdf")


# =============================================================================
# FIGURE 7 — Frontend screenshot triptych (PIL)
# =============================================================================
def fig7_screenshots():
    """Stitches three UI screenshots side-by-side with thin captions.
    Expects screenshots at docs/figures/ui_text.png, ui_voice.png, ui_image.png.
    """
    from PIL import Image, ImageDraw, ImageFont

    paths = [
        OUT_DIR / "ui_text.png",
        OUT_DIR / "ui_voice.png",
        OUT_DIR / "ui_image.png",
    ]
    captions = ["(a) Text query", "(b) Voice input", "(c) Image upload + classifier"]
    missing = [p for p in paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"  ⚠ Screenshot missing: {p.relative_to(ROOT)} — skipping fig7")
        return

    imgs = [Image.open(p).convert("RGB") for p in paths]
    target_h = 600
    resized = [im.resize((int(im.width * target_h / im.height), target_h), Image.LANCZOS) for im in imgs]
    gap = 16
    caption_h = 30
    total_w = sum(im.width for im in resized) + gap * (len(resized) - 1)
    canvas = Image.new("RGB", (total_w, target_h + caption_h), "white")
    draw = ImageDraw.Draw(canvas)
    # Try a portable cascade: Windows → macOS → Linux/Colab → PIL default
    font = None
    for candidate in (
        "times.ttf",                                       # Windows
        "/Library/Fonts/Times New Roman.ttf",              # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",  # Debian/Ubuntu/Colab
        "/usr/share/fonts/dejavu/DejaVuSerif.ttf",         # Fedora/Arch
        "DejaVuSerif.ttf",
    ):
        try:
            font = ImageFont.truetype(candidate, 16)
            break
        except (IOError, OSError):
            continue
    if font is None:
        font = ImageFont.load_default()

    x = 0
    for im, cap in zip(resized, captions):
        canvas.paste(im, (x, 0))
        tw = draw.textlength(cap, font=font)
        draw.text((x + (im.width - tw) / 2, target_h + 6), cap, fill="black", font=font)
        x += im.width + gap

    out = OUT_DIR / "fig7_screenshots.png"
    canvas.save(out, dpi=(300, 300))
    print(f"  ✓ {out.relative_to(ROOT)}")


# =============================================================================
# MAIN — comment out any figure you don't want to regenerate
# =============================================================================
if __name__ == "__main__":
    print("Generating paper figures into:", OUT_DIR.relative_to(ROOT))
    fig1_architecture()
    fig2_workflow()
    fig3_retrieval()
    fig4_latency()
    fig5_confusion()
    fig6_kg_example()
    fig7_screenshots()
    print("Done.")
