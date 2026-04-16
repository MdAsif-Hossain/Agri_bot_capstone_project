# AgriBot: Multimodal Pipelines & Architecture Deep Dive

This document provides a comprehensive breakdown of the core execution pipelines within AgriBot. It maps out the flow of data, the local AI models and transformers utilized, the integrated fallback mechanisms, and the crucial design rationales that make this architecture the best solution for an offline, hardware-constrained (e.g., RTX 3050 4GB VRAM) edge deployment in rural agriculture.

---

## 1. The Offline Ingestion Pipeline (`agribot/ingestion/`)
**Purpose:** Transform highly complex, multi-column agricultural PDF manuals into a clean, searchable hybrid vector database.

### Flow
`Raw PDFs` → `Marker (Layout-Aware Extraction)` → (Fallback: `Tesseract OCR` / `PyPDF`) → `Noise Filtering` → `Chunker` → `Index Builder` → `FAISS (Dense) & BM25 (Sparse) stores`

### Models & Tools Used
*   **Primary Extractor:** `Marker`
*   **Fallback Extractors:** `Tesseract OCR` (for scanned images) / `PyPDF` (for pure text fallbacks)
*   **Embedding Transformer:** `sentence-transformers/all-MiniLM-L6-v2` (A lightweight, CPU-efficient embedding model).

### Why this is the best solution:
Agricultural documents are notoriously messy, featuring complex tables, pesticide dosage charts, and scanned images. Traditional text extractors mangle this data. **Marker** accurately recovers reading order and structures tables into Markdown, which LLMs can easily comprehend. 
**Fallbacks:** If a page is entirely scanned, the pipeline automatically detects the lack of text and falls back to **Tesseract OCR** (eng+ben).
**Noise Filtering:** Heuristics calculate line frequencies to strip out repeated headers/footers that would otherwise pollute the vector search embeddings.

---

## 2. The Voice & Audio Pipeline (`agribot/voice/`)
**Purpose:** Handle real-time speech interactions (Speech-to-Text and Text-to-Speech) entirely offline.

### Flow
*   **Input:** `Raw Audio` → `VAD (Voice Activity Detection) / FFmpeg` → `faster-whisper` → (Fallback: `BanglaSpeech2Text` / `Vosk`) → `Bengali Transcript`
*   **Output:** `Bengali Text` → `OS Native TTS` → `Spoken Audio`

### Models & Tools Used
*   **Primary STT:** `faster-whisper` (medium model).
*   **Fallback 1:** `bangla-speech-processing/BanglaASR` (BanglaSpeech2Text).
*   **Fallback 2:** `Vosk` (Bengali acoustic models).
*   **TTS:** Native System Voices (SAPI5 on Windows).

### Why this is the best solution:
Farmers often speak in distinct dialects with heavy background noise (tractors, weather). Processing this in the cloud is impossible without internet. We utilize `faster-whisper` bound strictly to the **CPU**, reserving the GPU for the ultimate LLM generation. 
**Fallbacks:** Whisper occasionally hallucinates or struggles with deep rural dialects. We integrate `BanglaSpeech2Text` and `Vosk` as deterministic fallbacks if the primary confidence score falls below our threshold (`ASR_CONFIDENCE_THRESHOLD`), ensuring a robust safety net without breaking the interaction loop.

---

## 3. The CPU-Bound Vision Pipeline (`agribot/vision/`)
**Purpose:** Analyze crop disease photos without crashing low-VRAM edge GPUs.

### Flow
`Crop Image` → `Tesseract OCR (Text Extraction)` + `NumPy Color Heuristics (Symptoms)` → `MobileNetV3 ONNX Classifier` → `Composite Text/Keyword Description`

### Models & Tools Used
*   **Classifier:** `MobileNetV3` + CBAM Attention Module (exported to ONNX).
*   **OCR:** `Tesseract` (to read fertilizer packets or labels in the photo).
*   **Heuristics:** `OpenCV/Pillow/NumPy` (Extracting yellowing ratios, necrosis percentages).

### Why this is the best solution:
**The Problem:** Standard Vision-Language Models (VLMs) like LLaVA require massive VRAM (8GB+). Loading a VLM alongside our answering LLM on a 4GB RTX 3050 guarantees an Out-Of-Memory (OOM) crash.
**The Solution:** We offload vision to the CPU utilizing a lightweight Convolutional Neural Network (`MobileNetV3` via `ONNXRuntime`). We pair it with deterministic math (HSV color matching) to find "yellowing" or "brown spots." Instead of generating a full paragraph, this pipeline extracts structured keywords (e.g., `Confidence 92%: Rice Blast + Symptoms: Yellowing`), handing them dynamically to the Agent pipeline to retrieve context. 

---

## 4. The Hybrid Retrieval Pipeline (`agribot/retrieval/`)
**Purpose:** Accurately locate evidence facts within the ingested database.

### Flow
`Expanded Query` → `Parallel Search` (`FAISS Dense` + `BM25 Sparse`) → `Combine & Deduplicate` → `Cross-Encoder Reranker` → `Top-K Verified Evidence Chunks`

### Models & Tools Used
*   **Dense Search:** `FAISS` (Facebook AI Similarity Search).
*   **Sparse Search:** `BM25` (Keyword frequency matching).
*   **Reranker:** `ms-marco-MiniLM-L-12-v2` (Cross-Encoder).

### Why this is the best solution:
Relying solely on dense embeddings (FAISS) often misses exact chemical names or specific numerical dosages. Relying purely on BM25 misses semantic meaning (e.g., "bugs on leaves" vs "pest infestation"). **Hybrid retrieval** resolves this by doing both simultaneously. However, hybrid scoring is often misaligned, so we pipe the top results into a lightweight **Cross-Encoder Reranker**. The reranker actually "reads" the query and the chunk together to provide a final, highly accurate relevancy score.

---

## 5. The Agentic LLM Main Pipeline (`agribot/agent/` & `agribot/llm/`)
**Purpose:** The central orchestrator that manages conversations, language differences, and prevents LLM hallucination through self-correction.

### Flow (The LangGraph State Machine)
1. `Query` → **`Normalize Node`** (Translates BN to EN via `BanglaT5` if necessary).
2. → **`KG Link Node`** (Scans the SQLite Dialect Knowledge Graph for colloquial aliases and attaches scientific canonical terms to expand the query).
3. → **`Retrieve & Rerank Node`** (Calls the Retrieval Pipeline).
4. → **`Grade Node`** (LLM reads the evidence and grades it as `SUFFICIENT` or `INSUFFICIENT`).
5. → *(Fallback Loop)* If `INSUFFICIENT` → **`Rewrite Node`** (Formulates a better search query) → *Back to Retrieval (Max 2 Retries)*.
6. → **`Generate Node`** (LLM drafts the answer strictly citing the evidence).
7. → **`Translate Node`** (Translates EN answer back to BN via `BanglaT5`).
8. → **`Verify Node`** (Secondary LLM sanity check to ensure no un-evidenced claims slipped through).

### Models & Tools Used
*   **LLM Core:** `Qwen2.5 1.5B Instruct` (GGUF format, deployed via `llama-cpp-python`).
*   **Translation Mapper:** `csebuetnlp/banglat5_nmt_en_bn` & `bn_en`.
*   **Graph Framework:** `LangGraph`.
*   **Knowledge Graph (KG):** SQLite mapping (`Alias` → `Canonical Entity` → `Relation`).

### Why this is the best solution:
*   **Qwen 1.5B GGUF:** Extremely fast, capable reasoning model that neatly fits into partial GPU offloading while leaving room for the OS.
*   **Translators over Native BN LLMs:** Highly capable LLMs that can natively reason in fluent Bengali are too massive for edge hardware. Using a 1.5B English-reasoning core and bracketing it with dedicated `BanglaT5` CPU CPU-transformers yields higher reasoning quality at a fraction of the hardware cost.
*   **LangGraph Self-Correction vs Single Prompt:** A standard RAG just dumps search results into a prompt. If the search was bad, the LLM hallucinates an answer. By forcing the LLM to explicitly "Grade" the evidence first, we create a **fail-safe**. If the evidence is lacking, the Agentic Loop pauses, attempts to "Rewrite" the search string, and tries again. If it still fails, it triggers the Safety Policy to honestly refuse the question rather than delivering harmful agricultural advice.
*   **Dialect KG Expansion:** The term "মড়ক" might be used regionally for a disease, where manuals use the scientific name. The SQLite Knowledge Graph safely patches the farmer's colloquial input to the canonical textbook term *before* the search happens, drastically boosting the recall accuracy of the retrieval pipeline.
