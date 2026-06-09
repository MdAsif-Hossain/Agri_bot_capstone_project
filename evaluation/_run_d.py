"""Re-run ONLY ablation config D (the full agent) on GPU.

Config D is the only configuration with a verify node, so the grounding-overlap
backstop added to verify_answer affects D alone — A/B/C are unchanged and reused
from the prior full run. Writes ablation_latest (D-only table; a timestamped copy
of the prior A/B/C/D run is preserved) plus grounding_latest + refusal_latest from
config D's in-process pass.
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
    run(["D"], None)
