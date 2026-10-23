"""Local SSH tunnel entry point; uses the system SSH client."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from so101_project.tunnel import main

if __name__ == "__main__":
    raise SystemExit(main())
