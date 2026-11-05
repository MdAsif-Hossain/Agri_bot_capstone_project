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

# Project paths.
# Use __file__ when run as a script; fall back to CWD in notebooks/Colab,
# where __file__ is undefined.
try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    ROOT = Path.cwd()
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
# MATPLOTLIB-NATIVE DIAGRAM HELPERS (no graphviz/dot.exe required)
# =============================================================================
def _draw_box(ax, x, y, w, h, label, fill="#dde8f5", edge="black", fontsize=8,
              text_color="black", lw=0.8):
    """Draw a rounded rectangle with a centered label."""
    from matplotlib.patches import FancyBboxPatch

    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=lw, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, color=text_color, wrap=True)


def _draw_arrow(ax, x1, y1, x2, y2, label="", color="dimgray", lw=0.8, ls="-",
                fontsize=7, label_offset=(0, 0.02)):
    """Arrow from (x1,y1) to (x2,y2) with optional midpoint label."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                linestyle=ls, shrinkA=2, shrinkB=2))
    if label:
        mx, my = (x1 + x2) / 2 + label_offset[0], (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=fontsize,
                color=color)


def _setup_diagram_axes(ax, xlim=(0, 10), ylim=(0, 10)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.grid(False)


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
        c.node("llm", "Qwen2.5-3B (Q4_0 GGUF)\n— GPU —", fillcolor=PALETTE["bg_gpu"])
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
    g.edge("web", "api", label="HTTPS / JSON")
    g.edge("middleware", "api", style="dashed", arrowhead="none",
           label="all routes", fontsize="7", color="dimgray")

    # --- Edges: api → preprocessors → agent (per modality) ---
    g.edge("api", "stt", label="voice path", fontsize="7")
    g.edge("api", "imgproc", label="image path", fontsize="7")
    g.edge("imgproc", "classifier", style="dashed", label="optional", fontsize="7")
    g.edge("stt", "agent", label="transcribed text", fontsize="7", color="dimgray")
    g.edge("imgproc", "agent", label="combined query +\nimage analysis block",
           fontsize="7", color="dimgray")
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
    ax.set_ylim(0, max(1.28, max(vals) * 1.15 if vals else 1.0))

    # Stats badge: MRR + N + avg latency
    badge_lines = [f"MRR = {mrr:.3f}"]
    if n_queries is not None:
        badge_lines.append(f"N = {n_queries}")
    if avg_lat is not None:
        badge_lines.append(f"avg = {avg_lat:.0f} ms")
    # One line, centred in the headroom above the bars, so the badge never
    # covers the value label of the tallest bar (it did at k=10).
    ax.text(0.5, 0.96, "   |   ".join(badge_lines), transform=ax.transAxes, ha="center", va="top",
            fontsize=7, bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                   edgecolor="gray", linewidth=0.4))
    ax.set_axisbelow(True)
    _save(fig, "fig3_retrieval")


# =============================================================================
# FIGURE 4 — Per-stage latency stacked horizontal bar (reads eval JSON)
# =============================================================================
def fig4_latency():
    """Two stacked horizontal bars on a shared axis: CPU-only LLM inference
    (top) vs. GPU-offloaded LLM (bottom). Reads the two measured JSONs; the
    GPU bar makes the measured end-to-end speedup explicit instead of a
    projection. If the GPU JSON is absent, draws the CPU bar alone."""
    pipeline_order = ["normalize", "kg_link", "retrieve", "rerank", "grade",
                      "rewrite", "generate", "translate", "verify", "image_analysis"]

    def _load(path, fallback_stages, fallback_wall):
        if path.exists():
            blob = json.loads(path.read_text(encoding="utf-8"))
            d = blob["summary"]
            # Amortise over ALL queries, matching the paper's per-stage table.
            # summary["stage_means_ms"] averages each stage over only the
            # queries on which it ran, so `rewrite` (4 of 37) is inflated
            # ~9x and the shares do not sum to the stage total.
            pq = blob.get("per_query")
            if pq:
                stages = {k: sum(q["stage_means_ms"].get(k) or 0 for q in pq) / len(pq)
                          for k in d["stage_means_ms"]}
            else:
                stages = dict(d["stage_means_ms"])
            stages.pop("total", None)
            w = d["wall_ms"]
            return stages, (w["mean"], w["p50"], w["p95"])
        print(f"  ⚠ {path.name} not found — using embedded measured values")
        return dict(fallback_stages), fallback_wall

    cpu_stages, (cpu_mean, cpu_p50, cpu_p95) = _load(
        EVAL_DIR / "latency_latest.json",
        {"normalize": 0.1, "kg_link": 3.6, "retrieve": 16.8, "rerank": 429.3,
         "grade": 7970.5, "rewrite": 2449.5, "generate": 12641.1,
         "translate": 4121.1, "verify": 8488.5},
        (55433.7, 53005.2, 83578.0))

    gpu_path = EVAL_DIR / "latency_gpu_latest.json"
    gpu_stages, (gpu_mean, gpu_p50, gpu_p95) = _load(
        gpu_path,
        {"normalize": 0.1, "kg_link": 3.0, "retrieve": 12.8, "rerank": 192.2,
         "grade": 328.6, "rewrite": 252.0, "generate": 1576.0,
         "translate": 2801.4, "verify": 532.5},
        (7045.1, 6199.9, 13413.6))

    # Stable stage ordering + consistent colour per stage across both bars.
    all_stages = [s for s in pipeline_order
                  if cpu_stages.get(s, 0) > 0 or gpu_stages.get(s, 0) > 0]
    stage_color = dict(zip(all_stages,
                           plt.cm.viridis(np.linspace(0.15, 0.85, len(all_stages)))))
    overhead_color = (0.72, 0.72, 0.72, 1.0)

    def _segments(stages, mean_wall):
        segs = [(s, stages.get(s, 0.0)) for s in all_stages if stages.get(s, 0.0) > 0]
        overhead = max(0.0, mean_wall - sum(v for _, v in segs))
        segs.append(("overhead", overhead))
        return segs

    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(WIDE_WIDTH_IN, 1.55))
    rows = [("CPU", cpu_stages, cpu_mean, cpu_p50, cpu_p95, 1.0),
            ("GPU", gpu_stages, gpu_mean, gpu_p50, gpu_p95, 0.0)]
    bar_ends = []
    for label, stages, mean_wall, p50, p95, ypos in rows:
        left = 0.0
        segs = _segments(stages, mean_wall)
        # Percentages use the stage total, excluding the overhead segment,
        # so they match the paper's per-stage table exactly. The overhead
        # segment is still drawn (it is ~2% and falls below the label
        # threshold), which is why the bar is slightly wider than the sum.
        total = sum(v for n, v in segs if n != "overhead")
        for name, v in segs:
            c = overhead_color if name == "overhead" else stage_color[name]
            ax.barh(ypos, v, left=left, height=0.55, color=c,
                    edgecolor="white", linewidth=0.6)
            pct = v / total * 100 if total else 0
            # Only the wide CPU bar carries inline labels; the GPU bar is too
            # short, so its stages are read off the shared legend + colour.
            if label == "CPU" and pct >= 6:
                ax.text(left + v / 2, ypos, f"{name}\n{pct:.0f}%", ha="center",
                        va="center", fontsize=6.5,
                        color="black" if name == "overhead" else "white")
            left += v
        bar_ends.append(left)
        ax.text(left + cpu_mean * 0.015, ypos,
                f"{label}: p50 {p50/1000:.1f}\\,s, p95 {p95/1000:.1f}\\,s".replace("\\,", " "),
                va="center", ha="left", fontsize=8, fontweight="bold")

    speedup = cpu_p50 / gpu_p50 if gpu_p50 else 0
    ax.set_xlim(0, max(bar_ends) * 1.20)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels(["CPU-only\nLLM", "GPU-offload\nLLM"], fontsize=8)
    ax.set_xlabel("End-to-end per-stage latency (ms), mean over 37 queries "
                  "x 3 runs, amortised over all queries.", fontsize=8)

    # Every stage that appears in either bar goes in the legend (the old
    # [:7] cap silently dropped `translate` and `verify`).
    legend_stages = [s for s in all_stages if cpu_stages.get(s, 0) > 0]
    # "overhead" is only worth a legend entry when a bar actually has some.
    if any(max(0.0, w - sum(st.values())) > 0
           for st, w in ((cpu_stages, cpu_mean), (gpu_stages, gpu_mean))):
        legend_stages.append("overhead")
    handles = [Patch(facecolor=(overhead_color if s == "overhead" else stage_color[s]),
                     edgecolor="white", label=s) for s in legend_stages]
    ax.legend(handles=handles, ncol=min(len(legend_stages), 5), fontsize=6.5,
              loc="upper center", bbox_to_anchor=(0.5, -0.32),
              frameon=False, handlelength=1.0, columnspacing=1.0)
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
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
    """KG dialect-expansion example — Rice Blast canonical.

    All aliases AND neighbor relations are real entries from
    agribot/knowledge_graph/seed_data.py (verified):
      Aliases (lines 74-78):       ব্লাস্ট রোগ, blast, rice blast, পাতা পোড়া
      Neighbor relations:
        - leaf_spot symptom_of blast (line 198)
        - blast affects rice         (line 205)
        - tricyclazole treatment_for blast (line 217)
    """
    from graphviz import Digraph

    g = Digraph("kg", format="pdf")
    g.attr(rankdir="TB", bgcolor="white", fontname="Times", fontsize="10")
    g.attr("node", shape="box", style="filled,rounded", fontname="Times", fontsize="9")
    g.attr("edge", fontname="Times", fontsize="8", arrowsize="0.6")

    # Surface forms (real seeded aliases)
    g.node("alias_a", "পাতা পোড়া\n(colloquial, Bn)",
           fillcolor="#fff2cc", shape="note")
    g.node("alias_b", "ব্লাস্ট রোগ\n(standard, Bn)",
           fillcolor="#fff2cc", shape="note")
    g.node("alias_c", "rice blast\n(English)",
           fillcolor="#fff2cc", shape="note")

    # Canonical entity
    g.node("canon",
           "Rice Blast\nব্লাস্ট রোগ\n(canonical · type=disease)",
           fillcolor="#dde8f5")

    # Real 1-hop neighbors
    g.node("sym",
           "Leaf Spots\nপাতায় দাগ\n(symptom)",
           fillcolor="#f6e6f5")
    g.node("crop",
           "Rice\nধান\n(crop)",
           fillcolor="#c8e6c9")
    g.node("treat",
           "Tricyclazole\nট্রাইসাইক্লাজল\n(chemical)",
           fillcolor="#fde7d3")

    # Alias edges
    g.edge("alias_a", "canon", label="alias_of")
    g.edge("alias_b", "canon", label="alias_of")
    g.edge("alias_c", "canon", label="alias_of")

    # Real relation edges (direction matches seed_data.py)
    g.edge("sym",   "canon", label="symptom_of")     # leaf_spot symptom_of blast
    g.edge("canon", "crop",  label="affects")        # blast affects rice
    g.edge("treat", "canon", label="treatment_for")  # tricyclazole treatment_for blast

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
# MATPLOTLIB-ONLY DIAGRAM ALTERNATIVES (use these if graphviz/dot is unavailable)
# Same visual content as fig1/fig2/fig6 but rendered with pure matplotlib —
# no external `dot` binary needed.
# =============================================================================

def fig1_mpl():
    """Architecture diagram, pure matplotlib (no graphviz)."""
    fig, ax = plt.subplots(figsize=(WIDE_WIDTH_IN, 5.5))
    _setup_diagram_axes(ax, xlim=(0, 14), ylim=(0, 10))

    # --- Row 1: Client ---
    _draw_box(ax, 5.9, 8.7, 2.2, 0.8, "Web UI\n(Vite + React)", fill="#dde8f5")
    ax.text(0.2, 9.1, "Client", fontsize=8, color="gray", style="italic")

    # --- Row 2: API ---
    _draw_box(ax, 3.0, 7.0, 8.0, 0.8,
              "FastAPI · /v1/chat · /chat/voice · /chat/image · /tts\n"
              "middlewares: rate-limit · API-key · telemetry",
              fill="#fde7d3", fontsize=7)
    ax.text(0.2, 7.4, "API", fontsize=8, color="gray", style="italic")

    # --- Row 3: Preprocessors (CPU) ---
    _draw_box(ax, 0.5, 5.2, 2.6, 1.1,
              "STT pipeline\nfaster-whisper →\nBanglaSpeech2Text →\nVosk (fallback)",
              fill=PALETTE["bg_cpu"], fontsize=6.5)
    _draw_box(ax, 3.3, 5.2, 2.6, 1.1,
              "Image processor\nPIL/NumPy heuristics\n+ Tesseract OCR\n+ MobileNetV3+CBAM (ONNX)",
              fill=PALETTE["bg_cpu"], fontsize=6.5)
    ax.text(0.2, 5.7, "Multimodal\npreprocessors", fontsize=7, color="gray", style="italic")

    # --- Row 4: Agent ---
    _draw_box(ax, 3.0, 3.4, 8.0, 1.1,
              "LangGraph agent  (Fig. 2)\nnormalize → kg_link → retrieve → rerank → grade ↻ →\n"
              "generate → translate → verify → enforce_policy",
              fill="#f6e6f5", fontsize=7)

    # --- Row 5: Services ---
    _draw_box(ax, 0.3, 1.6, 2.1, 1.1, "Qwen2.5-1.5B\n(Q8_0 GGUF)\n— GPU —",
              fill=PALETTE["bg_gpu"], fontsize=7)
    _draw_box(ax, 2.6, 1.6, 2.6, 1.1, "Hybrid retrieval\nFAISS + BM25\n+ RRF + Reranker",
              fill=PALETTE["bg_cpu"], fontsize=7)
    _draw_box(ax, 5.4, 1.6, 2.1, 1.1, "KG entity\nlinker\n(SQLite)",
              fill=PALETTE["bg_cpu"], fontsize=7)
    _draw_box(ax, 7.7, 1.6, 2.1, 1.1, "BanglaT5\nEN↔BN\n(2 models)",
              fill=PALETTE["bg_cpu"], fontsize=7)
    _draw_box(ax, 10.0, 1.6, 2.1, 1.1, "TTS\npyttsx3\n(system voice)",
              fill=PALETTE["bg_cpu"], fontsize=7)
    ax.text(0.2, 2.1, "Services", fontsize=8, color="gray", style="italic")

    # --- Row 6: Data ---
    _draw_box(ax, 1.5, 0.1, 2.0, 0.7, "PDFs", fill="#ececec", fontsize=7)
    _draw_box(ax, 3.8, 0.1, 2.6, 0.7, "FAISS+BM25+chunks", fill="#ececec", fontsize=7)
    _draw_box(ax, 6.7, 0.1, 2.4, 0.7, "knowledge_graph.db", fill="#ececec", fontsize=7)
    _draw_box(ax, 9.4, 0.1, 2.0, 0.7, "telemetry.db", fill="#ececec", fontsize=7)

    # --- Arrows ---
    _draw_arrow(ax, 5.6, 8.65, 5.6, 7.85, "HTTPS", fontsize=6)
    _draw_arrow(ax, 8.4, 8.65, 8.4, 7.85, "HTTPS", fontsize=6)
    _draw_arrow(ax, 4.0, 6.95, 1.8, 6.35, "voice", fontsize=6, color="dimgray")
    _draw_arrow(ax, 6.0, 6.95, 4.6, 6.35, "image", fontsize=6, color="dimgray")
    _draw_arrow(ax, 8.0, 6.95, 8.0, 4.55, "text path", fontsize=6, color="dimgray")
    _draw_arrow(ax, 1.8, 5.15, 5.0, 4.55, "transcribed", fontsize=6, color="dimgray")
    _draw_arrow(ax, 4.6, 5.15, 6.0, 4.55, "combined query", fontsize=6, color="dimgray")
    _draw_arrow(ax, 11.0, 7.0, 11.0, 2.75, "/v1/tts", fontsize=6, color="dimgray", ls="--")
    # Agent → services
    for x in [1.4, 3.9, 6.4, 8.7]:
        _draw_arrow(ax, 5.5 + (x - 5.5) * 0.2, 3.35, x, 2.75, color="dimgray")
    # Services → data (dotted)
    _draw_arrow(ax, 3.9, 1.55, 4.5, 0.85, color="darkgray", ls=":")
    _draw_arrow(ax, 6.4, 1.55, 7.9, 0.85, color="darkgray", ls=":")

    _save(fig, "fig1_architecture")


def fig2_mpl():
    """LangGraph workflow, pure matplotlib (no graphviz)."""
    fig, ax = plt.subplots(figsize=(WIDE_WIDTH_IN, 2.6))
    _setup_diagram_axes(ax, xlim=(0, 18), ylim=(0, 4))

    # Linear chain
    chain = [
        ("Input",     0.2, 2.0, 1.3, 0.7, "#cccccc", "ellipse"),
        ("normalize", 1.7, 2.0, 1.3, 0.7, "#dde8f5", "box"),
        ("kg_link",   3.2, 2.0, 1.3, 0.7, "#dde8f5", "box"),
        ("retrieve\n(FAISS+BM25)", 4.7, 1.9, 1.7, 0.9, "#dde8f5", "box"),
        ("rerank\n(cross-enc)",    6.6, 1.9, 1.7, 0.9, "#dde8f5", "box"),
        ("grade\n(LLM)", 8.5, 1.9, 1.4, 0.9, "#fff2cc", "diamond"),
        ("generate\n(LLM)",   10.1, 1.9, 1.4, 0.9, "#dde8f5", "box"),
        ("translate\n(BanglaT5)", 11.7, 1.9, 1.5, 0.9, "#dde8f5", "box"),
        ("verify\n(LLM)",    13.4, 1.9, 1.3, 0.9, "#e0f0e0", "box"),
        ("enforce_policy\n(safety)", 14.9, 1.9, 1.9, 0.9, "#e0f0e0", "box"),
        ("Output",  17.0, 2.0, 0.9, 0.7, "#cccccc", "ellipse"),
    ]

    from matplotlib.patches import Ellipse, FancyBboxPatch
    centers = {}
    for name, x, y, w, h, color, shape in chain:
        if shape == "ellipse":
            ax.add_patch(Ellipse((x + w / 2, y + h / 2), w, h, facecolor=color,
                                  edgecolor="black", linewidth=0.8))
            ax.text(x + w / 2, y + h / 2, name, ha="center", va="center", fontsize=7)
        elif shape == "diamond":
            cx, cy = x + w / 2, y + h / 2
            ax.add_patch(plt.Polygon(
                [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)],
                facecolor=color, edgecolor="black", linewidth=0.8))
            ax.text(cx, cy, name, ha="center", va="center", fontsize=7)
        else:
            ax.add_patch(FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                facecolor=color, edgecolor="black", linewidth=0.8))
            ax.text(x + w / 2, y + h / 2, name, ha="center", va="center", fontsize=7)
        centers[name.split("\n")[0]] = (x + w / 2, y + h / 2, x, x + w)

    # Forward arrows
    order = ["Input", "normalize", "kg_link", "retrieve", "rerank", "grade"]
    for a, b in zip(order, order[1:]):
        ax.annotate("", xy=(centers[b][2], centers[b][1]),
                    xytext=(centers[a][3], centers[a][1]),
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

    # grade → generate (SUFFICIENT)
    ax.annotate("", xy=(centers["generate"][2], centers["generate"][1]),
                xytext=(centers["grade"][3], centers["grade"][1]),
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
    ax.text((centers["grade"][3] + centers["generate"][2]) / 2, 2.7,
            "SUFFICIENT\nor max retries", ha="center", fontsize=6, color="dimgray")

    # generate → translate → verify → enforce → Output
    for a, b in [("generate", "translate"), ("translate", "verify"),
                 ("verify", "enforce_policy"), ("enforce_policy", "Output")]:
        ax.annotate("", xy=(centers[b][2], centers[b][1]),
                    xytext=(centers[a][3], centers[a][1]),
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

    # Retry loop: grade → rewrite (down arrow) → retrieve (curved back)
    rew_x, rew_y = 7.5, 0.6
    ax.add_patch(FancyBboxPatch((rew_x - 0.7, rew_y - 0.35), 1.4, 0.7,
                                 boxstyle="round,pad=0.02,rounding_size=0.04",
                                 facecolor="#f5dada", edgecolor="black", linewidth=0.8))
    ax.text(rew_x, rew_y, "rewrite\n(LLM)", ha="center", va="center", fontsize=7)
    ax.annotate("", xy=(rew_x + 0.7, rew_y + 0.2), xytext=(centers["grade"][0], 1.85),
                arrowprops=dict(arrowstyle="->", color="#b54a4a", lw=0.8,
                                connectionstyle="arc3,rad=0.2"))
    ax.text((centers["grade"][0] + rew_x) / 2 + 0.4, 1.2,
            "INSUFFICIENT\n(retry < 2)", ha="center", fontsize=6, color="#b54a4a")
    ax.annotate("", xy=(centers["retrieve"][0], 1.85), xytext=(rew_x - 0.7, rew_y + 0.2),
                arrowprops=dict(arrowstyle="->", color="#b54a4a", lw=0.8, linestyle="--",
                                connectionstyle="arc3,rad=0.2"))

    _save(fig, "fig2_workflow")


def fig6_mpl():
    """KG dialect-expansion example, pure matplotlib (no graphviz)."""
    fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, 3.0))
    _setup_diagram_axes(ax, xlim=(0, 10), ylim=(0, 8))

    # Two dialect aliases (top)
    _draw_box(ax, 0.4, 6.5, 3.2, 1.0,
              "গাছফড়িং\n(colloquial alias)", fill="#fff2cc", fontsize=7)
    _draw_box(ax, 6.4, 6.5, 3.2, 1.0,
              "brown planthopper\n(English alias)", fill="#fff2cc", fontsize=7)

    # Canonical center
    _draw_box(ax, 2.8, 3.6, 4.4, 1.4,
              "বাদামি গাছফড়িং\nBrown Planthopper\n(canonical · type=pest)",
              fill="#dde8f5", fontsize=7.5)

    # Neighbors (bottom)
    _draw_box(ax, 0.1, 0.5, 2.3, 1.0, "Hopper burn\n(symptom)", fill="#f6e6f5", fontsize=6.5)
    _draw_box(ax, 2.6, 0.5, 2.4, 1.0, "Yellowing of leaves\n(symptom)", fill="#f6e6f5", fontsize=6.5)
    _draw_box(ax, 5.2, 0.5, 2.3, 1.0, "Imidacloprid\n(chemical)", fill="#fde7d3", fontsize=6.5)
    _draw_box(ax, 7.7, 0.5, 2.2, 1.0, "BR3 resistant\n(cultivar)", fill="#fde7d3", fontsize=6.5)

    # Arrows from aliases to canonical (with relation label)
    _draw_arrow(ax, 2.0, 6.45, 4.0, 5.05, "alias_of", fontsize=6)
    _draw_arrow(ax, 8.0, 6.45, 6.0, 5.05, "alias_of", fontsize=6)
    # Canonical to neighbors
    _draw_arrow(ax, 4.0, 3.55, 1.3, 1.55, "causes", fontsize=6)
    _draw_arrow(ax, 4.6, 3.55, 3.8, 1.55, "causes", fontsize=6)
    _draw_arrow(ax, 5.4, 3.55, 6.3, 1.55, "controlled_by", fontsize=6)
    _draw_arrow(ax, 6.0, 3.55, 8.8, 1.55, "controlled_by", fontsize=6)

    _save(fig, "fig6_kg_example")


# =============================================================================
# MAIN — comment out any figure you don't want to regenerate
# =============================================================================
if __name__ == "__main__":
    print("Generating paper figures into:", OUT_DIR.relative_to(ROOT))

    import shutil
    has_dot = shutil.which("dot") is not None

    if has_dot:
        fig1_architecture()
        fig2_workflow()
        fig6_kg_example()
    else:
        print("  ℹ Graphviz `dot` not on PATH — using matplotlib-only versions of Figs 1/2/6")
        fig1_mpl()
        fig2_mpl()
        fig6_mpl()

    fig3_retrieval()
    fig4_latency()
    fig5_confusion()
    fig7_screenshots()
    print("Done.")
