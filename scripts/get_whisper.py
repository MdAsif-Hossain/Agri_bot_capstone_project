"""Pre-download the faster-whisper model named in config into the HF cache.

The runtime serves STT fully offline, so the model must be cached once. This
script downloads (with resume) the Systran/faster-whisper-<size> repo named by
``settings.WHISPER_MODEL_SIZE`` into ``HF_HOME`` (kept on the F: drive).

Usage:
    venv\\Scripts\\python.exe scripts/get_whisper.py            # uses config size
    venv\\Scripts\\python.exe scripts/get_whisper.py medium     # explicit override
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import snapshot_download  # noqa: E402


def main() -> None:
    from config import settings

    size = sys.argv[1] if len(sys.argv) > 1 else settings.WHISPER_MODEL_SIZE
    repo = f"Systran/faster-whisper-{size}"
    print(f"Downloading {repo} (resume enabled) into the HF cache...", flush=True)
    path = snapshot_download(repo_id=repo, resume_download=True)
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    print(f"Done. {repo} cached at {path} ({total / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
