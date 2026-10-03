"""Fig. 1 (journal) - AgriBot system architecture as five layers.

Drawn deterministically with matplotlib at the final IEEE double-column
width (7.16 in) so every label is a known point size; nothing is scaled at
include time.  Layers are full-width bands, components sit on a fixed
column grid, connectors are orthogonal.  Every component named here
exists in the repository; in particular BN->EN input translation is drawn
where the code runs it, inside the agent (agribot/agent/nodes.py,
make_normalize_node), not as a separate preprocessor.

Run:  python scripts/fig1_architecture.py
Writes research/paper/figures/fig1.pdf and docs/figures/fig1.{pdf,png}.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUTS = [ROOT / "research" / "paper" / "figures" / "fig1",
        ROOT / "docs" / "figures" / "fig1"]

plt.rcParams.update({
    "font.family": "Arial",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

W, H = 7.16, 5.50                    # inches; y grows downward
COL = "#2B2B2B"                      # text / request arrows
RESP = "#A83232"                     # response path
DATA_LN = "#6E6E6E"                  # data access (dotted)
BAND_FILL, BAND_EDGE = "#F5F6F8", "#C7CAD1"

STYLE = {                            # fill, edge
    "client":  ("#E4ECF7", "#4F6FA3"),
    "gateway": ("#FFF3D6", "#B8862B"),
    "cpu":     ("#E6F2E6", "#3F8A4C"),
    "gpu":     ("#FCE3D2", "#C25E1A"),
    "agent":   ("#ECE6F7", "#6E4FA0"),
    "data":    ("#ECECEC", "#6E6E6E"),
    "proc":    ("#FFFFFF", "#6E6E6E"),
}

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.set_aspect("equal")
ax.axis("off")


# ----------------------------------------------------------------- helpers
def band(y0, y1, label):
    ax.add_patch(Rectangle((0.06, y0), W - 0.12, y1 - y0, facecolor=BAND_FILL,
                           edgecolor=BAND_EDGE, linewidth=0.6, zorder=0))
    ax.text(0.20, (y0 + y1) / 2, label, rotation=90, ha="center", va="center",
            fontsize=6.8, fontweight="bold", color="#4A4E57", zorder=1)


def box(x0, y0, x1, y1, title, body, kind, dashed=False, title_size=7.4, body_size=6.6):
    fill, edge = STYLE[kind]
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=0.05",
                                facecolor=fill, edgecolor=edge, linewidth=0.9,
                                linestyle="--" if dashed else "-", zorder=2))
    cx = (x0 + x1) / 2
    ax.text(cx, y0 + 0.11, title, ha="center", va="center", fontsize=title_size,
            fontweight="bold", color=COL, zorder=3)
    if body:
        ax.text(cx, y0 + 0.22, body, ha="center", va="top", fontsize=body_size,
                color=COL, linespacing=1.28, zorder=3)


def arrow(pts, color=COL, ls="-", lw=0.8, head=True, both=False):
    """Orthogonal polyline through pts; arrowhead on the last segment."""
    xs, ys = zip(*pts)
    if len(pts) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=4)
    style = "<|-|>" if both else ("-|>" if head else "-")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle=style, mutation_scale=7,
                                 color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0,
                                 zorder=4))


def elabel(x, y, s, ha="left", color=COL, size=6.2):
    ax.text(x, y, s, ha=ha, va="center", fontsize=size, color=color,
            linespacing=1.2, zorder=5,
            bbox=dict(boxstyle="square,pad=0.12", facecolor="white", edgecolor="none"))


# ------------------------------------------------------------------ bands
band(0.06, 0.70, "1  Client")
band(0.78, 1.46, "2  Gateway")
band(1.54, 2.56, "3  Preprocessors")
band(2.64, 4.38, "4  Agent + models")
band(4.46, 5.06, "5  Local data")

X0, X1 = 0.46, 7.02                  # content span
# preprocessor boxes sit left; the right-hand column is the text/response bus
pre_w = 2.30
pre_x = [X0, X0 + pre_w + 0.30]
# five model columns across the full span
n_mod, mod_g = 5, 0.09
mod_w = (X1 - X0 - (n_mod - 1) * mod_g) / n_mod
mod_x = [X0 + i * (mod_w + mod_g) for i in range(n_mod)]
mod_c = [x + mod_w / 2 for x in mod_x]

# ----------------------------------------------------------------- layer 1
cx_client = 1.40
box(0.46, 0.15, 2.34, 0.61, "Browser client (React)",
    "text  ·  voice recording  ·  leaf photo", "client")

# ----------------------------------------------------------------- layer 2
box(X0, 0.86, X1, 1.38, "FastAPI gateway",
    "REST  /v1/chat  ·  /v1/chat/voice  ·  /v1/chat/image  ·  /v1/tts (pyttsx3 system voice)\n"
    "rate limit  ·  optional API key  ·  semaphores: LLM 1, STT 2, image 2  ·  local telemetry",
    "gateway")

# ----------------------------------------------------------------- layer 3
py0, py1 = 1.70, 2.44
box(pre_x[0], py0, pre_x[0] + pre_w, py1, "Image pipeline  (CPU)",
    "vegetation gate (green pixels ≥ 12 %) →\n"
    "MobileNetV3-Large + CBAM (ONNX) →\n"
    "Tesseract OCR; colour-based symptom hints", "cpu")
box(pre_x[1], py0, pre_x[1] + pre_w, py1, "Speech-to-text cascade  (CPU)",
    "faster-whisper medium → Vosk → BanglaSpeech2Text;\n"
    "composite quality score; confidence caps;\n"
    "gate < 0.60 returns the transcript for confirmation", "cpu")

# ----------------------------------------------------------------- layer 4
ay0, ay1 = 2.78, 3.40
box(X0, ay0, X1, ay1, "LangGraph agent — bounded loop  (Fig. 3, Algorithm 1)",
    "normalize (BN→EN if Bengali) → kg_link → retrieve → rerank → grade → generate "
    "→ translate → verify → enforce_policy\n"
    "grade sends weak evidence to rewrite (at most twice) and back to retrieve",
    "agent", body_size=6.8)

my0, my1 = 3.60, 4.28
box(mod_x[0], my0, mod_x[0] + mod_w, my1, "Generator  (GPU)",
    "Qwen2.5-3B-Instruct, Q4_0\nllama.cpp, 36 layers on GPU\ncalled by grade, rewrite,\ngenerate and verify",
    "gpu", body_size=6.2, title_size=7.0)
box(mod_x[1], my0, mod_x[1] + mod_w, my1, "Input translator  (CPU)",
    "BanglaT5 BN→EN (stock)\ncalled by normalize when\nthe query contains\nBengali-block code points",
    "cpu", body_size=6.2, title_size=7.0)
box(mod_x[2], my0, mod_x[2] + mod_w, my1, "Hybrid retriever  (CPU)",
    "FAISS (MiniLM-L6) + BM25\n→ weighted RRF\n→ cross-encoder\n(MS-MARCO MiniLM-L-12)",
    "cpu", body_size=6.2, title_size=7.0)
box(mod_x[3], my0, mod_x[3] + mod_w, my1, "KG entity linker  (CPU)",
    "SQLite lookup, no model:\n28 entities, 84 aliases,\n20 relations; appends\n1-hop neighbours",
    "cpu", body_size=6.2, title_size=7.0)
box(mod_x[4], my0, mod_x[4] + mod_w, my1, "Output translator  (CPU)",
    "AgriBanglaT5 (BanglaT5\n+ LoRA, merged)\nEN→BN, line by line",
    "cpu", body_size=6.2, title_size=7.0)

# ----------------------------------------------------------------- layer 5
dy0, dy1 = 4.56, 4.98
ING_R = mod_x[2] - 0.24              # leave a visible gap for the "builds" arrow
box(mod_x[0], dy0, ING_R, dy1, "Offline ingestion (ingest.py)",
    "PDF → Marker / OCR → 600-char chunks (120 overlap)\n→ MiniLM embeddings → FAISS + BM25",
    "proc", dashed=True, body_size=6.3)
box(mod_x[2], dy0, mod_x[2] + mod_w, dy1, "FAISS + BM25 index",
    "+ chunk store: 914 chunks\nfrom 10 PDFs (204 pages)", "data", body_size=6.2)
box(mod_x[3], dy0, mod_x[3] + mod_w, dy1, "knowledge_graph.db",
    "SQLite, seeded offline", "data", body_size=6.2)
box(mod_x[4], dy0, mod_x[4] + mod_w, dy1, "telemetry.db",
    "SQLite, written by the\ngateway; stays on device", "data", body_size=6.2)

# ------------------------------------------------------------- connectors
# client <-> gateway
arrow([(cx_client, 0.61), (cx_client, 0.86)], both=True)
elabel(cx_client + 0.08, 0.735, "HTTP (JSON or multipart)")

# gateway -> preprocessors
pre_c = [x + pre_w / 2 for x in pre_x]
for x, lab in zip(pre_c, ["image", "audio"]):
    arrow([(x, 1.38), (x, py0)])
    elabel(x + 0.07, 1.54, lab)

# preprocessors -> agent
arrow([(pre_c[0], py1), (pre_c[0], ay0)])
elabel(pre_c[0] + 0.07, 2.61, "disease label + OCR text")
arrow([(pre_c[1], py1), (pre_c[1], ay0)])
elabel(pre_c[1] + 0.07, 2.61, "transcript")

# gateway -> agent directly: typed text in either language
xt = 5.92
arrow([(xt, 1.38), (xt, ay0)])
elabel(xt, 1.54, "text (EN or BN)", ha="center")

# response path: agent -> gateway (red), straight up the bus
xr = 6.38
arrow([(xr, ay0), (xr, 1.38)], color=RESP)
elabel(xr + 0.07, 2.02, "EN + BN answer,\ncitations, verdict,\npolicy action", color=RESP, size=5.9)

# agent <-> models
for x in mod_c:
    arrow([(x, ay1), (x, my0)], both=True)

# models <-> data (dotted) and ingestion -> index
for i in (2, 3):
    arrow([(mod_c[i], my1), (mod_c[i], dy0)], color=DATA_LN, ls=":", head=False)
arrow([(ING_R, 4.77), (mod_x[2], 4.77)], color=DATA_LN)

# ----------------------------------------------------------------- legend
renderer = fig.canvas.get_renderer()


def text_w(t):
    return t.get_window_extent(renderer).width / fig.dpi


ly = 5.20
x = 0.14
for kind, lab, dashed in [("gpu", "GPU-resident model", False),
                          ("cpu", "CPU-pinned model or lookup", False),
                          ("agent", "orchestration", False),
                          ("data", "data on disk", False),
                          ("proc", "offline process", True)]:
    fill, edge = STYLE[kind]
    ax.add_patch(Rectangle((x, ly - 0.065), 0.17, 0.13, facecolor=fill, edgecolor=edge,
                           lw=0.8, linestyle="--" if dashed else "-"))
    t = ax.text(x + 0.22, ly, lab, va="center", fontsize=6.3, color=COL)
    x += 0.22 + text_w(t) + 0.16
ly = 5.38
x = 0.14
for color, ls, lab in [(COL, "-", "request path"), (RESP, "-", "response path"),
                       (DATA_LN, ":", "index / database access")]:
    ax.plot([x, x + 0.26], [ly, ly], color=color, lw=0.8, ls=ls)
    t = ax.text(x + 0.31, ly, lab, va="center", fontsize=6.3, color=COL)
    x += 0.31 + text_w(t) + 0.16

for out in OUTS:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    if "docs" in str(out):
        fig.savefig(out.with_suffix(".png"), dpi=300)
    print("wrote", out.with_suffix(".pdf"))
