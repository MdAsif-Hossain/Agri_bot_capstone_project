"""Run the FULL ablation (configs A, B, C, D) on GPU in one coherent pass.

Produces ablation_latest.json (A/B/C/D table) plus grounding_latest.json and
refusal_latest.json from config D's in-process pass. The verifier grounding
backstop affects config D only; A/B/C have no verify node.
"""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# CUDA DLL preamble (Windows): make the GPU llama-cpp build offload to the RTX.
_site = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
_llama = Path(sys.prefix) / "Lib" / "site-packages" / "llama_cpp" / "lib"
for _d in list(_site.glob("*/bin")) + [_llama]:
    if _d.is_dir():
        os.add_dll_directory(str(_d))
        os.environ["PATH"] = str(_d) + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("AGRIBOT_LLM_N_GPU_LAYERS", "-1")

from evaluation.eval_ablation import run  # noqa: E402

if __name__ == "__main__":
    run(["A", "B", "C", "D"], None)
