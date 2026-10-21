"""Local hardware entry point; preview is the default."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from so101_project.launch import LOCAL_ACTIONS, main

if __name__ == "__main__":
    raise SystemExit(main(LOCAL_ACTIONS))
