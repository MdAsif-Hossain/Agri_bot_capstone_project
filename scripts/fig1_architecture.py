"""Fig. 1 (journal) - AgriBot system architecture as five layers.

Drawn deterministically with matplotlib at the final IEEE double-column
width (7.16 in) so every label is a known point size; nothing is scaled at
include time.  Layers are full-width bands, components sit on a fixed
column grid, connectors are orthogonal.  Every component named here
exists in the repository (see research/paper/README.md for provenance).

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

W, H = 7.16, 5.30                    # inches; y grows downward
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
    if body:
        ax.text(cx, y0 + 0.11, title, ha="center", va="center", fontsize=title_size,
                fontweight="bold", color=COL, zorder=3)
        ax.text(cx, y0 + 0.22, body, ha="center", va="top", fontsize=body_size,
                color=COL, linespacing=1.28, zorder=3)
    else:
        ax.text(cx, (y0 + y1) / 2, title, ha="center", va="center", fontsize=title_size,
                fontweight="bold", color=COL, zorder=3)


def arrow(pts, color=COL, ls="-", lw=0.8, head=True, both=False):
    """Orthogonal polyline through pts; arrowhead on the last segment."""
    xs, ys = zip(*pts)
    if len(pts) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=4)
    style = "<|-|>" if both else ("-|>" if head else "-")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle=style, mutation_scale=7,
                                 color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0,
                                 zorder=4))


def elabel(x, y, s, ha="left", color=COL, size=6.2, rot=0):
    ax.text(x, y, s, ha=ha, va="center", fontsize=size, color=color, rotation=rot,
            linespacing=1.2, zorder=5,
            bbox=dict(boxstyle="square,pad=0.12", facecolor="white", edgecolor="none"))


# ------------------------------------------------------------------ bands
B1 = (0.06, 0.70)
B2 = (0.78, 1.46)
B3 = (1.54, 2.56)
B4 = (2.64, 4.30)
B5 = (4.38, 5.02)
band(*B1, "1  Client")
band(*B2, "2  Gateway")
band(*B3, "3  Preprocessors")
band(*B4, "4  Agent + models")
band(*B5, "5  Local data")

# column grid for the three preprocessors and the four models
X0, XR = 0.46, 5.79                  # content span (right column is the bus)
pre_w = 1.56
pre_x = [X0, X0 + pre_w + 0.14, X0 + 2 * pre_w + 0.14 + 0.42]
mod_w, mod_g = 1.28, 0.07
mod_x = [X0 + i * (mod_w + mod_g) for i in range(4)]

# ----------------------------------------------------------------- layer 1
cx_client = 1.40
box(0.46, 0.15, 2.34, 0.61, "Browser client (React)",
    "text  ·  voice recording  ·  leaf photo", "client")

# ----------------------------------------------------------------- layer 2
box(0.46, 0.86, 6.10, 1.38, "FastAPI gateway",
    "REST  /v1/chat  ·  /v1/chat/voice  ·  /v1/chat/image  ·  /v1/tts (pyttsx3 system voice)\n"
    "rate limit  ·  optional API key  ·  semaphores: LLM 1, STT 2, image 2  ·  local telemetry",
    "gateway")

# ----------------------------------------------------------------- layer 3
py0, py1 = 1.70, 2.44
box(pre_x[0], py0, pre_x[0] + pre_w, py1, "Image pipeline  (CPU)",
    "vegetation gate (green px ≥ 12%) →\n"
    "MobileNetV3-L + CBAM, ONNX →\n"
    "Tesseract OCR, colour hints", "cpu")
box(pre_x[1], py0, pre_x[1] + pre_w, py1, "Speech-to-text cascade  (CPU)",
    "faster-whisper medium → Vosk →\n"
    "BanglaSpeech2Text; composite\n"
    "quality score, confidence gate", "cpu")
box(pre_x[2], py0, pre_x[2] + pre_w, py1, "Input translation  (CPU)",
    "BanglaT5 BN→EN (stock)\n"
    "applied when the query is in\n"
    "the Bengali Unicode block", "cpu")

# ----------------------------------------------------------------- layer 4
ay0, ay1 = 2.78, 3.40
box(X0, ay0, XR, ay1, "LangGraph agent — bounded loop  (Fig. 3, Algorithm 1)",
    "normalize → kg_link → retrieve → rerank → grade → generate "
    "→ translate → verify → enforce_policy\n"
    "grade sends weak evidence to rewrite (at most twice) and back to retrieve",
    "agent", body_size=6.8)

my0, my1 = 3.62, 4.20
box(mod_x[0], my0, mod_x[0] + mod_w, my1, "Generator  (GPU)",
    "Qwen2.5-3B-Instruct, Q4_0\nllama.cpp, 36 layers on GPU\ngrade, rewrite, generate, verify",
    "gpu", body_size=6.2)
box(mod_x[1], my0, mod_x[1] + mod_w, my1, "Hybrid retriever  (CPU)",
    "FAISS (MiniLM-L6) + BM25\nweighted RRF → cross-encoder\n(MS-MARCO MiniLM-L-12)", "cpu")
box(mod_x[2], my0, mod_x[2] + mod_w, my1, "KG entity linker  (CPU)",
    "28 entities, 84 aliases,\n20 relations; appends\n1-hop neighbours", "cpu")
box(mod_x[3], my0, mod_x[3] + mod_w, my1, "Output translator  (CPU)",
    "AgriBanglaT5 (BanglaT5 +\nLoRA, merged), EN→BN", "cpu")

# ----------------------------------------------------------------- layer 5
dy0, dy1 = 4.48, 4.94
ING_R = mod_x[1] - 0.22              # leave a visible gap for the "builds" arrow
box(mod_x[0], dy0, ING_R, dy1, "Offline ingestion",
    "PDF → Marker / OCR →\n600-char chunks → embed", "proc", dashed=True, body_size=6.3)
box(mod_x[1], dy0, mod_x[1] + mod_w, dy1, "FAISS + BM25 indexes",
    "+ chunk store: 914 chunks\nfrom 10 PDFs (204 pages)", "data")
box(mod_x[2], dy0, mod_x[2] + mod_w, dy1, "knowledge_graph.db",
    "SQLite, seeded offline", "data")
box(5.90, dy0, 7.02, dy1, "telemetry.db",
    "SQLite, written by the\ngateway; stays on device", "data")

# ------------------------------------------------------------- connectors
# client <-> gateway
arrow([(cx_client, 0.61), (cx_client, 0.86)], both=True)
elabel(cx_client + 0.08, 0.735, "HTTP  JSON / multipart")

# gateway -> preprocessors (request path)
for x, lab in zip([p + pre_w / 2 for p in pre_x], ["image", "audio", "Bengali text"]):
    arrow([(x, 1.38), (x, py0)])
    elabel(x + 0.07, 1.54, lab)

# gateway -> agent directly (English text)
xe = 5.92
arrow([(xe, 1.38), (xe, ay0)])
elabel(xe, 1.54, "English text", ha="center")

# STT -> input translation (transcript)
arrow([(pre_x[1] + pre_w, 2.07), (pre_x[2], 2.07)])
elabel(pre_x[1] + pre_w + 0.08, 1.93, "transcript", size=6.0)

# preprocessors -> agent
xi = pre_x[0] + pre_w / 2
arrow([(xi, py1), (xi, ay0)])
elabel(xi + 0.07, 2.61, "disease label + OCR text")
xt = pre_x[2] + pre_w / 2
arrow([(xt, py1), (xt, ay0)])
elabel(xt + 0.07, 2.61, "English query")

# agent <-> models
for x in [m + mod_w / 2 for m in mod_x]:
    arrow([(x, ay1), (x, my0)], both=True)

# models <-> data (dotted)
for i in (1, 2):
    x = mod_x[i] + mod_w / 2
    arrow([(x, my1), (x, dy0)], color=DATA_LN, ls=":", head=False)
arrow([(ING_R, 4.71), (mod_x[1], 4.71)], color=DATA_LN)

# response path: agent -> gateway (red), drawn on the right-hand bus
xr = 6.30
arrow([(XR, 3.09), (xr, 3.09), (xr, 1.12), (6.10, 1.12)], color=RESP)
elabel(xr + 0.07, 2.20, "EN + BN answer,\ncitations, verdict,\npolicy action", color=RESP, size=5.9)

# ----------------------------------------------------------------- legend
ly = 5.16
renderer = fig.canvas.get_renderer()


def text_w(t):
    return t.get_window_extent(renderer).width / fig.dpi


x = 0.14
for kind, lab in [("gpu", "GPU-resident model"), ("cpu", "CPU-pinned model"),
                  ("agent", "orchestration"), ("data", "data on disk")]:
    fill, edge = STYLE[kind]
    ax.add_patch(Rectangle((x, ly - 0.065), 0.17, 0.13, facecolor=fill, edgecolor=edge, lw=0.8))
    t = ax.text(x + 0.22, ly, lab, va="center", fontsize=6.3, color=COL)
    x += 0.22 + text_w(t) + 0.16
for color, ls, lab in [(COL, "-", "request"), (RESP, "-", "response"), (DATA_LN, ":", "data access")]:
    ax.plot([x, x + 0.26], [ly, ly], color=color, lw=0.8, ls=ls)
    t = ax.text(x + 0.31, ly, lab, va="center", fontsize=6.3, color=COL)
    x += 0.31 + text_w(t) + 0.16

for out in OUTS:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    if "docs" in str(out):
        fig.savefig(out.with_suffix(".png"), dpi=300)
    print("wrote", out.with_suffix(".pdf"))
