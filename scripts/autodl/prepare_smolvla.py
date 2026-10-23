"""Prepare pinned public SmolVLA checkpoint and tokenizer/configuration files."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from so101_project.model_assets import main

if __name__ == "__main__":
    raise SystemExit(main())
