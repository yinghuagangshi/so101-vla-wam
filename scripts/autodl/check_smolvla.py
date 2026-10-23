"""Check SmolVLA software with synthetic inputs and no hardware."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from so101_project.software_check import main

if __name__ == "__main__":
    raise SystemExit(main())
