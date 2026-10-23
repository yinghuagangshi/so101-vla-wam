"""Check an exported model on the server without connecting to a robot."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from so101_project.checkpoint import main

if __name__ == "__main__":
    raise SystemExit(main())
