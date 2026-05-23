# AgriBot Deployment Runbook

> Audience: field officers and lab/IT staff installing AgriBot on a single
> offline laptop or LAN-accessible workstation. Assumes a fresh Windows 10/11
> or Ubuntu 22.04 install with no prior Python/Node tooling.

---

## 1. What you are deploying

A self-contained offline system composed of:

| Component | Role | Resource budget |
|---|---|---|
| FastAPI backend (`api.py`) | Text/voice/image API + agent orchestration | ~5 GB RAM, 2-4 GB VRAM (LLM) |
| Vite/React frontend | Browser UI for operators | Negligible |
| Local LLM (Qwen2.5-1.5B GGUF) | Answer generation | 1.9 GB on disk |
| FAISS + BM25 indexes | Hybrid retrieval | ~10 MB per 1000 chunks |
| SQLite KG (`knowledge_graph.db`) | Dialect → canonical entity mapping | < 1 MB |
| BanglaT5 EN↔BN models | Bilingual translation | ~2 GB on disk |
| MobileNetV3+CBAM ONNX | Rice disease classifier | 17 MB |
| MS-MARCO MiniLM-L-12-v2 | Cross-encoder reranker | 34 MB |
| BanglaSpeech2Text + Whisper | Bengali/English ASR | ~1.5 GB on disk |

The deployment is **fully offline after first install** — no outbound
internet calls at query time.

---

## 2. Hardware requirements

| Tier | CPU | RAM | Storage | GPU |
|---|---|---|---|---|
| Minimum | 4 cores, AVX2 | 8 GB | 25 GB free | None (CPU LLM works, slower) |
| Recommended | 6+ cores, AVX2 | 16 GB | 40 GB free | 4 GB+ VRAM CUDA-capable |
| Field deployment | Intel i5 12th gen+ | 16 GB | 50 GB free | RTX 3050 4 GB+ |

> **AVX-512 note:** Intel Alder Lake (12th gen) and newer disable AVX-512.
> The CPU build of `llama-cpp-python` must be the AVX2 variant — the default
> "cpu" wheel uses AVX-512 and will crash with `STATUS_ILLEGAL_INSTRUCTION`
> (Windows error `0xC000001D`).

---

## 3. One-time install

### 3.1 Install Python 3.12 (off the system drive when possible)

Windows (via winget):
```powershell
winget install Python.Python.3.12 --silent --location "F:\Python312"
```

Linux:
```bash
sudo apt install -y python3.12 python3.12-venv python3.12-dev
```

### 3.2 Create the project venv

```powershell
# Windows
F:\Python312\python.exe -m venv F:\Projects\Agri_bot\venv
.\venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel
```

```bash
# Linux
python3.12 -m venv ~/agri_bot/venv
source ~/agri_bot/venv/bin/activate
python -m pip install -U pip setuptools wheel
```

### 3.3 Redirect caches off the system drive (Windows)

Set these as **persistent user environment variables** so all future shells
pick them up automatically:

```powershell
[System.Environment]::SetEnvironmentVariable("PIP_CACHE_DIR",            "F:\Projects\Agri_bot\.cache\pip", "User")
[System.Environment]::SetEnvironmentVariable("HF_HOME",                  "F:\Projects\Agri_bot\.cache\huggingface", "User")
[System.Environment]::SetEnvironmentVariable("HF_HUB_CACHE",             "F:\Projects\Agri_bot\.cache\huggingface\hub", "User")
[System.Environment]::SetEnvironmentVariable("TRANSFORMERS_CACHE",       "F:\Projects\Agri_bot\.cache\huggingface\transformers", "User")
[System.Environment]::SetEnvironmentVariable("SENTENCE_TRANSFORMERS_HOME","F:\Projects\Agri_bot\.cache\sentence-transformers", "User")
[System.Environment]::SetEnvironmentVariable("TORCH_HOME",               "F:\Projects\Agri_bot\.cache\torch", "User")
```

### 3.4 Install Python dependencies

```powershell
pip install -r requirements.txt
# AVX2-only CPU build of llama-cpp-python (Alder Lake and newer):
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu_avx2 --prefer-binary --force-reinstall --no-deps
```

If you have CUDA Toolkit 12.x installed and want GPU acceleration:
```powershell
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122 --prefer-binary --force-reinstall --no-deps
```

### 3.5 Install Node.js (frontend only)

```powershell
winget install OpenJS.NodeJS.LTS
cd frontend
npm install
```

### 3.6 Download model artifacts

Place these in `models/`:

| File | Source | Size |
|---|---|---|
| `qwen2.5-1.5b-instruct-q8_0.gguf` | HuggingFace `Qwen/Qwen2.5-1.5B-Instruct-GGUF` (Q8_0) | 1.9 GB |
| `vision/rice_disease_mobilenetv3_cbam.onnx` | Trained CBAM CNN (in `cnn/outputs/`) | 414 KB |
| `vision/rice_disease_mobilenetv3_cbam.onnx.data` | External weights, same folder as `.onnx` | 16 MB |
| `vision/rice_disease_mobilenetv3_cbam.txt` | 6 class labels, one per line | 87 B |
| `reranker/ms-marco-MiniLM-L-12-v2/` | HF download (auto on first run) | 34 MB |
| `banglaspeech2text/BanglaSpeech2Text/` | Per BanglaSpeech2Text package | ~500 MB |

BanglaT5 models download automatically to `HF_HOME` on first backend boot
(~2 GB combined).

### 3.7 Place agricultural PDFs

Drop the source corpus into `data/pdfs/`. Any new PDFs added later will be
picked up by re-running `python ingest.py`.

### 3.8 Build indexes

```powershell
python ingest.py
```

This extracts text from PDFs, chunks them (default 600 chars, 120 overlap),
builds the FAISS + BM25 indexes, and seeds the knowledge graph. Expect:

- **5–10 minutes** for ~20 PDFs with pypdf-only extraction
- **30–60 minutes** for the first run with Marker enabled (downloads ~5 GB of
  Surya layout/OCR models). Set `AGRIBOT_DISABLE_MARKER=1` to skip Marker
  entirely if you cannot afford the download.

---

## 4. Configuration

All knobs live in `config.py` and can be overridden via environment variables
prefixed `AGRIBOT_` (e.g. `AGRIBOT_LLM_N_GPU_LAYERS=20`). Common overrides:

| Variable | Default | Use case |
|---|---|---|
| `AGRIBOT_LLM_N_GPU_LAYERS` | 15 | Lower if VRAM tight; 0 = CPU only |
| `AGRIBOT_GROUNDING_MODE` | strict | "lenient" to suppress refusal disclaimers |
| `AGRIBOT_RATE_LIMIT_PER_MINUTE` | 30 | Per-client request cap |
| `AGRIBOT_RATE_LIMIT_BURST` | 10 | Token-bucket overflow |
| `AGRIBOT_API_KEY` | "" | Set to a non-empty string to enforce X-API-Key |
| `AGRIBOT_IMAGE_CLASSIFIER_ENABLED` | true | Disable to skip CBAM CNN |
| `AGRIBOT_REQUEST_TIMEOUT_S` | 120 | Per-request hard timeout |

Place a `.env` file in the project root for persistent overrides — see
`.env.example` for the full list.

---

## 5. Running the system

Start the backend (terminal 1):
```powershell
python api.py
# Serves http://localhost:8000
# OpenAPI docs at http://localhost:8000/docs
```

Start the frontend (terminal 2):
```powershell
cd frontend
npm run dev
# Vite serves http://localhost:5173 (or :5174 if 5173 is busy)
```

Browse to the URL printed by Vite. The frontend talks to the backend at
`http://localhost:8000` by default. To expose to a LAN, run Vite with
`--host 0.0.0.0` and pass the LAN backend URL via env.

For a production-style deployment, run uvicorn behind a reverse proxy:
```powershell
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
```
> Use **a single worker** — the LLM is process-local (GPU memory). Scaling is
> done by adding nodes, not workers.

---

## 6. Verifying the install

```powershell
curl http://localhost:8000/v1/health
```
Expected: `status: ok`, all module flags true except `vlm` (optional VLM).

End-to-end text query:
```powershell
$body = '{"query": "What are good agricultural practices for fertilizer application?"}'
$body | Out-File -Encoding UTF8 -NoNewline .query.json
curl -s -X POST http://localhost:8000/v1/chat -H "Content-Type: application/json" --data-binary "@.query.json"
```
Expected: non-empty `answer`, `citations` array, `is_verified: true`.

---

## 7. Operations

### 7.1 Telemetry

The backend writes every request to `data/telemetry.db` (SQLite). Inspect via
the built-in endpoint:
```
GET /v1/telemetry?hours=24&top=10
```
Returns request count, p50/p95 latency, refusal rate, and top queries.

### 7.2 Updating the knowledge base

Drop new PDFs into `data/pdfs/`, then:
```powershell
python ingest.py
```
Restart the backend to load the rebuilt indexes. No code changes needed.

### 7.3 Rotating the LLM

Drop a new GGUF into `models/` and update `AGRIBOT_MODEL_PATH`. Quantization
levels Q4_K_M (smaller, slightly degraded) through Q8_0 (default) are
supported. Avoid F16 weights on 4 GB GPUs.

### 7.4 Backup / recovery

To clone an install to a second laptop, copy:
```
config.py
.env
agribot/
api.py, ingest.py, requirements.txt, pyproject.toml
models/        (~3 GB)
data/pdfs/     (your corpus)
data/indexes/  (or re-run ingest.py)
data/knowledge_graph.db
```
Skip `venv/`, `.cache/`, `data/telemetry.db`, `__pycache__/`.

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `OSError: WinError -1073741795` on llama_cpp import | AVX-512 build on AVX2 CPU | Reinstall `llama-cpp-python` from `…/whl/cpu_avx2` |
| Backend hangs at "Loading LLM" | GPU layers too high, VRAM OOM | Lower `AGRIBOT_LLM_N_GPU_LAYERS` or set to 0 |
| "Marker extraction failed … artifact_dict" | marker-pdf >= 1.x without models loaded | Already handled; falls back to pypdf. To enable Marker, ensure HF cache is writable |
| All answers refuse with "I don't know" | Indexes empty or missing | Re-run `python ingest.py` |
| Bengali text returned as mojibake garbage | Legacy non-Unicode fonts in source PDFs | Enable Marker (uses Surya OCR) or install Tesseract + Bengali language pack |
| Rate-limit 429 returned to a legitimate user | Bucket too small or shared NAT | Raise `AGRIBOT_RATE_LIMIT_PER_MINUTE` or set `AGRIBOT_RATE_LIMIT_ENABLED=false` |
| BanglaT5 download stalls behind a corporate proxy | HF_HUB_OFFLINE not set | Pre-download on another machine, copy `HF_HOME/hub` over |

---

## 9. Security checklist before field deployment

- [ ] Set a non-empty `AGRIBOT_API_KEY` and store it in the device's secure
      enclave (Windows Credential Manager / GNOME Keyring).
- [ ] Set `AGRIBOT_GROUNDING_MODE=strict` and `AGRIBOT_ON_VERIFY_FAIL=refuse`
      for safety-critical advisory contexts.
- [ ] Confirm `data/telemetry.db` retention policy meets the deployment's
      privacy rules — by default it stores the query text verbatim (truncated
      to 280 chars).
- [ ] Bind the API to `127.0.0.1` rather than `0.0.0.0` unless the LAN is
      trusted.
- [ ] Verify the frontend's `VITE_API_BASE_URL` points to the intended
      backend before building for distribution.
