# AgriBot: End-to-End System Workflow & Architecture

This document visualizes the complete data and execution flow of the **AgriBot** system, explicitly showing how inputs from various client platforms are routed through the FastAPI gateway, pre-processed by dedicated CPU models, orchestrated by the Agentic LangGraph loop, and evaluated by the strictly isolated local LLM.

## Workflow Diagram

```mermaid
graph TD
    %% Styling
    classDef client fill:#e1f5fe,stroke:#3b82f6,stroke-width:2px,color:#000;
    classDef api fill:#fdf4ff,stroke:#8b5cf6,stroke-width:2px,color:#000;
    classDef cpu fill:#f0fdf4,stroke:#10b981,stroke-width:2px,color:#000;
    classDef gpu fill:#fff1f2,stroke:#ec4899,stroke-width:2px,color:#000;
    classDef db fill:#fefce8,stroke:#eab308,stroke-width:2px,color:#000;
    classDef agent fill:#f8fafc,stroke:#64748b,stroke-width:2px,color:#000;

    %% 1. Clients
    subgraph Clients ["📱 Multi-Platform Client Ecosystem"]
        direction TB
        Web["💻 Web Kiosk App"]:::client
        Mobile["📱 Mobile App (Farmers)"]:::client
        Desktop["🖥️ Desktop App (Researchers)"]:::client
    end

    %% 2. Inputs
    subgraph Modalities ["📥 Input Modalities"]
        direction LR
        TextInput[/"📝 Text Input"/]
        VoiceInput[/"🎤 Voice Audio"/]
        ImageInput[/"📷 Crop Image"/]
    end

    Clients --> TextInput & VoiceInput & ImageInput

    %% 3. API Gateway
    subgraph Gateway ["🛡️ FastAPI Microservices Gateway"]
        API["api.py Central Router"]:::api
        Locks((Semaphore<br>Locks)):::api
        API -.- Locks
    end

    TextInput -->|/v1/chat| API
    VoiceInput -->|/v1/chat/voice| API
    ImageInput -->|/v1/chat/image| API

    %% 4. Pre-processing (CPU-Bound Models)
    subgraph CPU_Processing ["⚙️ Hardware-Isolated Pre-Processing (Strictly CPU)"]
        direction TB
        STT["🎙️ Speech-to-Text (STT)<br>Primary: whisper_base_bn_sifat<br>Fallback: Vosk Local"]:::cpu
        Vision["👁️ Image Classification<br>Primary: MobileNetV3 + CBAM CNN<br>Fallback: OCR + Color Heuristics"]:::cpu
        TextMerger["Unified Query String<br>(User Text OR Keywords OR Resumed Voice)"]:::agent
    end

    API -->|Audio Data| STT
    API -->|Image Data| Vision
    API -->|Raw Text| TextMerger

    STT -->|Low Confidence? < 0.6<br>Route to UI for Confirm| Clients
    STT -->|Success: Transcript| TextMerger
    Vision -->|Success: Top-K Keywords| TextMerger

    %% 5. LangGraph Agentic Loop
    subgraph LangGraph ["🧠 Agentic RAG Loop (LangGraph Orchestrator)"]
        direction TB
        
        Normalize["1. Normalize<br>(CPU: BanglaT5 translation)"]:::agent
        KGLink["2. KG Link<br>(Dialect to Canvas)"]:::agent
        Retrieve["3. Hybrid Retrieve<br>(CPU: Vector + BM25)"]:::agent
        Rerank["4. Rerank<br>(CPU: ms-marco Cross-Encoder)"]:::agent
        
        Grade{"5. Grade Evidence<br>(GPU: Qwen2.5 LLM)"}:::gpu
        
        Rewrite["6a. Rewrite Query<br>(GPU: Qwen2.5 LLM)"]:::gpu
        Generate["6b. Generate Answer /<br>Ask Follow-up (GPU: Qwen2.5)"]:::gpu
        
        Translate["7. Re-Translate output<br>(CPU: BanglaT5)"]:::agent
        Verify["8. Hallucination Check<br>(GPU: Qwen2.5 LLM)"]:::gpu
        Policy["9. Enforce Policy<br>(Add disclaimer/Refusal)"]:::agent
    end

    TextMerger --> Normalize
    Normalize --> KGLink
    KGLink --> Retrieve
    Retrieve --> Rerank
    Rerank --> Grade
    
    Grade -->|"INSUFFICIENT<br>(Failed & Retries < 2)"| Rewrite
    Rewrite -->|Loop back| Retrieve
    
    Grade -->|"SUFFICIENT<br>OR (Retries == 2)"| Generate
    Generate --> Translate
    Translate --> Verify
    Verify --> Policy

    %% Databases
    subgraph DataStore ["💾 Offline Storage"]
        direction TB
        KGDB[("SQLite Database<br>(Knowledge Graph & Aliases)")]:::db
        VectorDB[("FAISS Index &<br>BM25 lexical chunks")]:::db
    end

    KGLink -.->|Fetch canonical mapping| KGDB
    Retrieve -.->|Search vectors| VectorDB

    %% 6. Output Delivery
    subgraph Output ["📤 Output Delivery & Diagnostics"]
        JSONOut[/"Structured JSON Output<br>[Markdown + Citations + Trace Latencies]"/]
        TTSOut[/"Native OS TTS Synthesis<br>(Optional Audio file)"/]
    end

    Policy --> JSONOut
    JSONOut -->|If Audio request| TTSOut
    
    JSONOut ---> Clients
    TTSOut ---> Clients
```

---

## 🔍 Workflow Breakdown by Phase

### Phase 1: Input & Hardware Distribution
1. **The Client Ecosystem** sends requests to the `FastAPI` Orchestrator. 
2. The orchestrator immediately implements **Concurrency Locks** to ensure the 4GB GPU or limited system RAM does not crash under simultaneous requests.
3. Requests route to **Strictly CPU-bound modules** based on modality:
   - **Images** bypass LLMs entirely and hit a specialized **MobileNetV3 + CBAM CNN** (which outputs textual condition keywords like *"Tomato Early Blight"*).
   - **Voice** audio is transcribed by an offline `faster-whisper` module. **Fallback:** If confidence is below 60%, it aborts the LLM run and queries the user via the UI to confirm what they said.

### Phase 2: The Agentic Core (LangGraph)
Once the input is transformed into standardized text/keywords, it enters the self-correcting RAG workflow:
1. **Normalize & Link:** The query is formatted (translated if needed via `BanglaT5`) and mapped against a local **SQLite Knowledge Graph** to convert regional slang into scientific terminology.
2. **Hybrid Search:** Both semantic (`FAISS/sentence-transformers`) and lexical (`BM25`) searching extracts candidate PDF chunks matching the expanded query.
3. **Reranking:** The CPU-bound Cross-Encoder scores exact matches, trimming noise.
4. **GPU Hand-off (Grading & Rewrite Loop):** The `Qwen2.5` LLM steps in to **Grade** the top chunks. 
   - If the chunks don't contain the answer, the LLM actively **Rewrites** the search query and loops back to the retrieval stage.
5. **Generation & Conversational Fallback:** Once sufficient evidence is found (or maximum search retries are hit), the system generates an answer. If visual context is missing (because the CNN gave keyword hints without full environmental understanding), the LLM skips answering and generates a *Follow-up question* to the farmer (e.g., *"How old is the affected crop?"*).

### Phase 3: Safety & Delivery
1. **Validation:** The final answer is passed *back* through the GPU to check for hallucinations against the raw evidence.
2. **Strict Grounding:** If verification fails, the response is explicitly heavily modified to include disclaimers or full refusals rather than risk giving a farmer fake agricultural advice.
3. **Response:** A highly structured payload featuring exact PDF source-citations and processing latencies is returned to the user, with an optional system TTS audio stream generated for low-literacy users.