"""GPU launcher for the AgriBot backend.

Run with the CUDA venv:  venv-gpu\\Scripts\\python.exe run_gpu.py

This wires the pip-installed CUDA runtime DLLs onto the DLL search path BEFORE
api.py imports llama_cpp (otherwise the cu124 build fails to load), forces full
GPU offload of the generation model, then starts the normal FastAPI app.

The CPU path (python api.py from the main venv) is unchanged and remains the
stable fallback.
"""

import os
import sys
from pathlib import Path

# --- CUDA DLL path preamble: MUST run before llama_cpp is imported anywhere ---
_site = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
_llama_lib = Path(sys.prefix) / "Lib" / "site-packages" / "llama_cpp" / "lib"
for _d in list(_site.glob("*/bin")) + [_llama_lib]:
    if _d.is_dir():
        os.add_dll_directory(str(_d))
        os.environ["PATH"] = str(_d) + os.pathsep + os.environ.get("PATH", "")

# Force full GPU offload of all transformer layers (config reads AGRIBOT_* env).
# -1 = offload every layer; the 3B-Q4 model fits in 4 GB VRAM (~2.2 GB resident).
os.environ.setdefault("AGRIBOT_LLM_N_GPU_LAYERS", "-1")

if __name__ == "__main__":
    import uvicorn

    print("Starting AgriBot backend on GPU (n_gpu_layers="
          f"{os.environ['AGRIBOT_LLM_N_GPU_LAYERS']})...", flush=True)
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
