"""GPU launcher for the ablation harness.

eval_ablation.py is device-agnostic (it reads settings.LLM_N_GPU_LAYERS). This
thin wrapper does the same CUDA DLL preamble as eval_latency_gpu.py so the
cu124 llama-cpp-python build in venv-gpu loads, forces full GPU offload, and
then runs the ablation in-process (no HTTP backend, so it is robust to the
long-lived-server instability we saw under sustained load).

Run (from the GPU venv):
    venv-gpu\\Scripts\\python.exe evaluation/run_ablation_gpu.py            # A B C D
    venv-gpu\\Scripts\\python.exe evaluation/run_ablation_gpu.py D          # one config
"""

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# --- CUDA DLL path preamble: MUST run before importing llama_cpp ---
_site = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
_llama_lib = Path(sys.prefix) / "Lib" / "site-packages" / "llama_cpp" / "lib"
for _d in list(_site.glob("*/bin")) + [_llama_lib]:
    if _d.is_dir():
        os.add_dll_directory(str(_d))
        os.environ["PATH"] = str(_d) + os.pathsep + os.environ.get("PATH", "")

# Force full GPU offload before config/llama are imported.
os.environ.setdefault("AGRIBOT_LLM_N_GPU_LAYERS", "-1")

if __name__ == "__main__":
    from evaluation.eval_ablation import run

    configs = [a.upper() for a in sys.argv[1:]] or ["A", "B", "C", "D"]
    print(f"Running ablation on GPU for configs={configs} "
          f"(n_gpu_layers={os.environ['AGRIBOT_LLM_N_GPU_LAYERS']})", flush=True)
    run(configs, None)
