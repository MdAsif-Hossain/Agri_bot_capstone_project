"""Fig. 1 generator — Colab snippet.

Paste the body of this file into a single Colab cell after:
    !pip install graphviz -q
    !apt install -y graphviz

The corrections vs. the original draft (commented inline below):
  FIX 1: Vision Pipeline output skips the BanglaT5 (Bn→En) translator —
         CBAM CNN labels are already English.
  FIX 2: Text input from FastAPI goes to KG entity-linking directly;
         only Bengali speech transcripts pass through BanglaT5 (Bn→En).
  FIX 3: Added an explicit "Query Rewrite" node in the INSUFFICIENT
         retry loop (matches the actual LangGraph workflow).
  CLEANUP A: Added a "Verify" node between LLM and Grounding & Policy
             (matches the agent graph; reinforces the safety story).
  CLEANUP B: The Safety→FastAPI dashed edge label changed from "Flags"
             to "Refusal/Disclaimer" so reviewers know what it means.
"""

import graphviz


def generate_architecture():
    dot = graphviz.Digraph('Agri_bot_Architecture', format='png', engine='dot')

    # Global attributes
    dot.attr(fontname='Helvetica', fontsize='12', rankdir='TB',
             splines='ortho', nodesep='0.6', ranksep='0.8')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica',
             fontsize='11', color='#333333', penwidth='1.2', margin='0.15,0.1')
    dot.attr('edge', fontname='Helvetica', fontsize='10',
             color='#555555', penwidth='1.0')

    # 1. Client tier
    with dot.subgraph(name='cluster_clients') as c:
        c.attr(label='Client Tier / User Interface', style='dashed',
               color='#aaaaaa', bgcolor='#FAFCFF')
        c.node('User', 'Farmer / User\n(Mobile/Web/Desktop)',
               shape='ellipse', fillcolor='#E3F2FD')
        c.node('Inputs', 'Multimodal Inputs\n(Audio, Image, Bengali/English Text)',
               fillcolor='#FFFFFF')
        c.edge('User', 'Inputs')

    # 2. Offline edge server tier
    with dot.subgraph(name='cluster_edge') as edge:
        edge.attr(label='Offline Edge Device Environment', style='solid',
                  color='#333333', bgcolor='#F5F5F5', penwidth='1.5')

        edge.node('FastAPI', 'FastAPI Backend\n(REST · rate-limit · worker semaphores)',
                  fillcolor='#FFF9C4', shape='component')

        # Multimodal preprocessors (CPU-pinned)
        with edge.subgraph(name='cluster_pre') as pre:
            pre.attr(label='Peripheral Models (CPU-pinned)',
                     style='dotted', bgcolor='#FFFFFF')
            pre.node('STT', 'Speech-to-Text\n(faster-whisper / Vosk)',
                     fillcolor='#E0F7FA')
            pre.node('Vision', 'Vision Pipeline\n(MobileNetV3+CBAM ONNX + OCR)',
                     fillcolor='#E0F7FA')
            pre.node('TransIn', 'Translation\n(BanglaT5: Bn → En)',
                     fillcolor='#E0F7FA')

        # Agent core
        with edge.subgraph(name='cluster_agent') as agent:
            agent.attr(label='Agentic RAG Orchestrator (LangGraph)',
                       style='solid', color='#2196F3', bgcolor='#E3F2FD')
            agent.node('KG', 'KG Entity Linker\n(Alias / Dialect Normalization)',
                       fillcolor='#FFFFFF')
            agent.node('Retrieve', 'Hybrid Retrieval\n(Dense + Sparse + RRF)',
                       fillcolor='#FFFFFF')
            agent.node('Rerank', 'Cross-Encoder Reranker\n(FlashRank)',
                       fillcolor='#FFFFFF')
            agent.node('Grade', 'State Grader\n(Context Evaluator)',
                       fillcolor='#FFE0B2')
            # FIX 3: explicit Rewrite node in the retry loop
            agent.node('Rewrite', 'Query Rewrite\n(LLM, on INSUFFICIENT)',
                       fillcolor='#FFE0B2')
            agent.node('LLM', 'Generative LLM (GPU)\n(llama.cpp + Qwen2.5-3B)',
                       fillcolor='#C8E6C9', penwidth='2.0')
            # CLEANUP A: Verify node before Grounding & Policy
            agent.node('Verify', 'Verify\n(LLM grounding check)',
                       fillcolor='#DCEDC8')
            agent.node('Safety', 'Grounding & Policy\n(Refusal Guard · risk-aware)',
                       fillcolor='#FFCDD2', fontcolor='black')

        # Local data stores
        with edge.subgraph(name='cluster_data') as data:
            data.attr(label='Local Data Stores', style='dotted', bgcolor='#FFFFFF')
            data.node('FAISS', 'Dense Vector DB\n(FAISS)',
                      shape='cylinder', fillcolor='#F3E5F5')
            data.node('BM25', 'Sparse DB\n(BM25)',
                      shape='cylinder', fillcolor='#F3E5F5')
            data.node('SQLiteKG', 'Knowledge Graph\n(SQLite · 28 ent / 84 aliases)',
                      shape='cylinder', fillcolor='#F3E5F5')

        # Output processing
        with edge.subgraph(name='cluster_out') as out:
            out.attr(label='Output Processing (CPU)', style='dotted',
                     bgcolor='#FFFFFF')
            out.node('TransOut', 'Translation\n(BanglaT5: En → Bn)',
                     fillcolor='#E0F7FA')
            out.node('TTS', 'Text-to-Speech\n(pyttsx3)', fillcolor='#E0F7FA')

    # 3. Offline ingestion pipeline
    with dot.subgraph(name='cluster_ingest') as ingest:
        ingest.attr(label='Offline Ingestion (one-time / on PDF update)',
                    style='dashed', bgcolor='#FAFAFA', color='#888888')
        ingest.node('PDFs', 'Agri-Documents', shape='note', fillcolor='#FFFFFF')
        ingest.node('Marker', 'Layout Parser\n(Marker + OCR fallback)',
                    fillcolor='#FFFFFF')
        ingest.node('Embed', 'Chunker & Embed\n(Sentence-Transformers MiniLM)',
                    fillcolor='#E0F7FA')
        ingest.edge('PDFs', 'Marker')
        ingest.edge('Marker', 'Embed')

    # ----- Edges -----

    # Client → FastAPI
    dot.edge('Inputs', 'FastAPI', label=' Payload')

    # FastAPI → preprocessors
    dot.edge('FastAPI', 'STT', label='Audio')
    dot.edge('FastAPI', 'Vision', label='Image')
    # FIX 2: text input goes directly into the agent (via KG linker);
    # only Bengali audio transcripts need translation.
    dot.edge('FastAPI', 'KG', label='Text\n(En or Bn)')

    # Preprocessors → agent
    dot.edge('STT', 'TransIn', label='Bn Text')
    dot.edge('TransIn', 'KG', label='En Text')
    # FIX 1: Vision skips the Bn→En translator — CBAM labels are already English.
    dot.edge('Vision', 'KG', label='Disease labels +\nsymptom hints')

    # KG ↔ store
    dot.edge('SQLiteKG', 'KG', dir='both')

    # Agent pipeline
    dot.edge('KG', 'Retrieve')
    dot.edge('FAISS', 'Retrieve', dir='back')
    dot.edge('BM25', 'Retrieve', dir='back')
    dot.edge('Retrieve', 'Rerank')
    dot.edge('Rerank', 'Grade')

    # FIX 3: INSUFFICIENT loop now goes through Rewrite, not direct to Retrieve.
    dot.edge('Grade', 'Rewrite', label=' INSUFFICIENT\n(retry < 2)',
             color='#E65100', fontcolor='#E65100')
    dot.edge('Rewrite', 'Retrieve', color='#E65100')

    dot.edge('Grade', 'LLM', label=' PASS')

    # CLEANUP A: LLM → Verify → Safety
    dot.edge('LLM', 'Verify')
    dot.edge('Verify', 'Safety')

    # Safety output
    dot.edge('Safety', 'TransOut', label=' Safe answer')
    # CLEANUP B: rename "Flags" → "Refusal/Disclaimer"
    dot.edge('Safety', 'FastAPI', label=' Refusal /\nDisclaimer',
             style='dashed', color='#D32F2F', fontcolor='#D32F2F')

    dot.edge('TransOut', 'TTS', label=' Audio req')
    dot.edge('TransOut', 'FastAPI', label=' Bn answer')
    dot.edge('TTS', 'FastAPI', label=' Audio response')

    # Ingestion → stores
    dot.edge('Embed', 'FAISS')
    dot.edge('Embed', 'BM25')

    dot.render('fig1', format='png', cleanup=True)
    dot.render('fig1', format='pdf', cleanup=True)
    print('wrote fig1.png / fig1.pdf')
    return 'fig1.png'


if __name__ == "__main__":
    output_file = generate_architecture()
    # Auto-download in Colab; ignored when run locally.
    try:
        from google.colab import files
        for f in ('fig1.png', 'fig1.pdf'):
            files.download(f)
    except Exception:
        print(f"Generated: {output_file}")
