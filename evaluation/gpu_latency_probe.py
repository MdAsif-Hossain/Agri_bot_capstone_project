"""Standalone GPU-vs-CPU LLM latency probe (fix-list item 3).

This does NOT touch the working venv. It is meant to be run from the
isolated `venv-gpu` (which has a CUDA build of llama-cpp-python) to measure
the generation-stage speedup of GPU offload over CPU on the SAME model and
the SAME prompts. It deliberately isolates the LLM generation stage (which
the paper's latency breakdown shows dominates wall-clock time) rather than
standing up the whole agent, so it needs only llama-cpp-python.

Usage (from the GPU venv):
    venv-gpu/Scripts/python.exe evaluation/gpu_latency_probe.py --device gpu
    venv-gpu/Scripts/python.exe evaluation/gpu_latency_probe.py --device cpu
    venv-gpu/Scripts/python.exe evaluation/gpu_latency_probe.py --device both

Writes evaluation/results/gpu_probe_latest.json. Reports per-prompt and
aggregate generation time, tokens/s, and (for --device both) the GPU/CPU
speedup. If the model cannot be placed on the GPU (OOM on 4 GB, or missing
CUDA runtime), the script prints the exact exception so it can be quoted in
the paper's limitations rather than guessed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "qwen2.5-1.5b-instruct-q8_0.gguf"
RESULTS = ROOT / "evaluation" / "results"

# Representative agronomy prompts (mirror the kinds of generate() calls the
# agent makes: a grade, a generate, a verify). Kept short and fixed so the
# CPU and GPU runs are directly comparable.
PROMPTS = [
    "List five good agricultural practices for rice fertiliser application.",
    "Explain what integrated pest management is in two sentences.",
    "What causes zinc deficiency (Khaira disease) in rice? Answer briefly.",
    "Summarise why climate-smart agriculture matters for smallholder farmers.",
    "Describe the dapog method of rice nursery raising in two sentences.",
]

MAX_TOKENS = 256
N_CTX = 4096


def _load(device: str, n_gpu_layers: int):
    from llama_cpp import Llama
    layers = n_gpu_layers if device == "gpu" else 0
    t0 = time.perf_counter()
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=N_CTX,
        n_gpu_layers=layers,
        verbose=False,
    )
    load_s = time.perf_counter() - t0
    return llm, load_s


def _run_device(device: str, n_gpu_layers: int) -> dict:
    print(f"\n=== device={device} (n_gpu_layers={n_gpu_layers if device=='gpu' else 0}) ===")
    llm, load_s = _load(device, n_gpu_layers)
    print(f"  model loaded in {load_s:.1f}s")

    per_prompt = []
    for i, p in enumerate(PROMPTS, start=1):
        prompt = (f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant")
        t0 = time.perf_counter()
        out = llm(prompt, max_tokens=MAX_TOKENS, temperature=0.1,
                  stop=["<|im_end|>"])
        dt = time.perf_counter() - t0
        n_tok = out["usage"]["completion_tokens"]
        tps = n_tok / dt if dt else 0
        per_prompt.append({"prompt_idx": i, "seconds": round(dt, 2),
                           "completion_tokens": n_tok, "tokens_per_s": round(tps, 1)})
        print(f"  [{i}/{len(PROMPTS)}] {dt:.1f}s, {n_tok} tok, {tps:.1f} tok/s")

    secs = [r["seconds"] for r in per_prompt]
    toks = sum(r["completion_tokens"] for r in per_prompt)
    total = sum(secs)
    return {
        "device": device,
        "n_gpu_layers": n_gpu_layers if device == "gpu" else 0,
        "model_load_s": round(load_s, 2),
        "mean_generation_s": round(total / len(secs), 2),
        "total_generation_s": round(total, 2),
        "aggregate_tokens_per_s": round(toks / total, 1) if total else 0,
        "per_prompt": per_prompt,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "gpu", "both"], default="both")
    ap.add_argument("--n-gpu-layers", type=int, default=-1,
                    help="-1 = all layers on GPU; fall back to a smaller number "
                         "if 4 GB OOMs.")
    args = ap.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}")

    results: dict = {"model": MODEL_PATH.name, "max_tokens": MAX_TOKENS,
                     "n_ctx": N_CTX, "n_prompts": len(PROMPTS)}

    devices = ["cpu", "gpu"] if args.device == "both" else [args.device]
    for dev in devices:
        try:
            results[dev] = _run_device(dev, args.n_gpu_layers)
        except Exception as e:  # noqa: BLE001 - we want the exact error text
            import traceback
            results[dev] = {"device": dev, "error": f"{type(e).__name__}: {e}",
                            "traceback": traceback.format_exc()[-800:]}
            print(f"  !! {dev} FAILED: {type(e).__name__}: {e}")

    if "cpu" in results and "gpu" in results \
            and "error" not in results["cpu"] and "error" not in results["gpu"]:
        cpu_s = results["cpu"]["mean_generation_s"]
        gpu_s = results["gpu"]["mean_generation_s"]
        results["speedup_gpu_over_cpu"] = round(cpu_s / gpu_s, 2) if gpu_s else None
        print(f"\n  GPU speedup over CPU (generation): "
              f"{results['speedup_gpu_over_cpu']}x")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "gpu_probe_latest.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults written to: {out}")


if __name__ == "__main__":
    main()
