# AgriBot Remaining Tasks to Complete Project

This document tracks the final tasks required to reach true "Production-Grade" status and finalize the Final Year Project.

## 1. Vision Pipeline Integration (Phase 11)
- [ ] Implement `MobileNetV3` ONNX model loading and inference in `agribot/vision/classifier.py`.
- [ ] Add PIL/NumPy color and symptom heuristics (e.g., yellowing/spotting detection) in `agribot/vision/image_processor.py`.
- [ ] Update `agribot/agent/graph.py` to handle the "conversational fallback": explicitly instruct the agent to ask the user clarifying questions if the vision confidence score is low.
- [ ] Remove the config gates to make the vision pipeline an active part of the system.

## 2. DevOps & Deployment Artifacts
- [ ] Create a `Dockerfile` for the FastAPI Python backend (including llama-cpp-python, onnxruntime, faster-whisper).
- [ ] Create a `Dockerfile` for the Vite/React frontend.
- [ ] Create a `docker-compose.yml` to stand up both services, the database, and local indexes simultaneously.
- [ ] Write an Operations/Deployment Runbook (`docs/DEPLOYMENT.md`) explaining how field officers can install the system on a fresh offline laptop.

## 3. Evaluation & Benchmarks (Crucial for Research Papers)
- [ ] Build a Ground-Truth Evaluation Dataset: A JSON file mapping ~50-100 real-world agricultural queries in colloquial Bangla to their correct manual/page target.
- [ ] Run the evaluation scripts to get hard numbers on: Recall@k, Mean Reciprocal Rank (MRR), and Groundedness/Refusal correctness.
- [ ] Perform and document hardware stress testing: Prove the "0 OOM crashes on 4GB VRAM" claim by sending concurrent requests.
- [ ] Finalize `docs/EVALUATION.md` with charts, metrics, and Time-to-First-Token latency statistics.

## 4. Security & Observability Hardening
- [ ] Ensure unique Request Trace IDs are attached to all JSON logs to track an entire multimodal request lifecycle.
- [ ] Add basic API rate-limiting middleware in `api.py` to prevent mobile thin-clients from accidentally flooding the system.
- [ ] Test the `asyncio.Semaphore` implementations to ensure the system safely queues or rejects requests when the LLM or Vision engine is busy.
- [ ] Add a simple local telemetry tracker (e.g., appending basic usage stats like "Top queries" to a local SQLite table) to demonstrate operational readiness.
