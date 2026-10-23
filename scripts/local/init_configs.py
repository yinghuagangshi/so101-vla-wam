"""Create personal configuration copies without overwriting existing files."""
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    templates = ["configs/hardware/smolvla.local.example.json",
                 "configs/deployment/smolvla.local.example.json",
                 "configs/deployment/smolvla.autodl.example.json",
                 "configs/deployment/ssh.example.json",
                 "experiments/smolvla_language/experiment.example.json"]
    for name in templates:
        source = root / name
        destination = source.with_name(source.name.replace(".example", ""))
        try:
            with destination.open("xb") as stream:
                stream.write(source.read_bytes())
            print(f"Created: {destination.relative_to(root)}")
        except FileExistsError:
            print(f"Kept: {destination.relative_to(root)}")


if __name__ == "__main__":
    main()
