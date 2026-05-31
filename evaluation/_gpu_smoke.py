"""GPU smoke test: confirm llama-cpp-python (cu124 wheel) loads the model on
the RTX 3050 with the pip-installed CUDA runtime DLLs, and time one generation.

Run with the GPU venv:
    venv-gpu\\Scripts\\python.exe evaluation\\_gpu_smoke.py
"""
import os
import sys
import time
from pathlib import Path

# Register the pip-installed NVIDIA runtime DLL directories on the Windows
# DLL search path BEFORE importing llama_cpp (cudart64_12, cublas64_12,
# cublasLt64_12 live here; no system CUDA toolkit is installed).
site = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
llama_lib = Path(sys.prefix) / "Lib" / "site-packages" / "llama_cpp" / "lib"
dll_dirs = list(site.glob("*/bin")) + [llama_lib]
for d in dll_dirs:
    if d.is_dir():
        os.add_dll_directory(str(d))
        # Also prepend to PATH: add_dll_directory alone does not always cover
        # transitive/lazy native loads (ggml-cuda.dll -> cudart/cublas).
        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
        print(f"  + DLL dir: {d}")

import llama_cpp  # noqa: E402

print("llama_cpp version:", llama_cpp.__version__)
print("supports_gpu_offload:", llama_cpp.llama_supports_gpu_offload())

MODEL = r"F:\Projects\Agri_bot\models\qwen2.5-1.5b-instruct-q8_0.gguf"

t0 = time.perf_counter()
llm = llama_cpp.Llama(
    model_path=MODEL,
    n_ctx=4096,
    n_gpu_layers=-1,  # offload ALL layers to GPU
    verbose=True,     # prints "offloaded N/N layers to GPU" if CUDA active
)
load_s = time.perf_counter() - t0
print(f"\nMODEL LOADED in {load_s:.1f}s")

t1 = time.perf_counter()
out = llm("<|im_start|>user\nName three rice diseases.<|im_end|>\n<|im_start|>assistant",
          max_tokens=64, temperature=0.1, stop=["<|im_end|>"])
gen_s = time.perf_counter() - t1
txt = out["choices"][0]["text"].strip()
print(f"\nGEN in {gen_s:.1f}s ({64/gen_s:.1f} tok/s approx):")
print(txt[:200])
