"""User-started, sequential, CPU-limited training on a synthetic practice dataset."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/synthetic"

def run_module(module: str, *arguments: str) -> None:
    subprocess.run([sys.executable, "-u", "-m", module, *arguments], cwd=ROOT, check=True)

def main() -> int:
    print("PLATEVISION - short practice training", flush=True)
    print("Synthetic data only: this is not training on real Indian vehicle photographs.", flush=True)
    print("2 CPU threads. Models train sequentially, about 2 minutes each.", flush=True)
    print("The current batch and checkpoint save may extend that budget slightly.", flush=True)
    try:
        import torch
        import cv2
        expected = [DATA / kind / (split + ".json")
                    for kind in ("plates", "characters") for split in ("train", "val", "test")]
        if not all(path.exists() for path in expected):
            if any(path.exists() for path in expected):
                print("An incomplete dataset was found. Please inspect data/synthetic before continuing.")
                return 1
            print("\nCreating the small practice dataset...", flush=True)
            run_module("training.generate_synthetic_data", "--scenes", "300", "--chars-per-class", "40")
        for kind, dataset in (("plate_detector", "plates"), ("character_classifier", "characters")):
            out = ROOT / "models" / kind
            checkpoint = out / "last.pt"
            args = ["--data", str(DATA / dataset), "--out", str(out),
                    "--epochs", "30", "--max-minutes", "2", "--threads", "2", "--cooldown", "0.1"]
            if checkpoint.exists():
                print(f"\nResuming {kind} from its saved checkpoint...", flush=True)
                args += ["--resume", str(checkpoint)]
            else:
                print(f"\nStarting {kind} from random weights...", flush=True)
            run_module("training.train_" + kind, *args)
            print(f"Saved files: {out}", flush=True)
        print("\nThis training session has finished. Double-click TRAIN_MODELS.cmd again to resume.")
        print("A best.pt file appears only after a complete training epoch and validation.")
        print("Synthetic practice models are not validated for real vehicle images.")
        print("The separate vehicle detector needs no training. See README.md for its download command.")
        return 0
    except ImportError as exc:
        print(f"Missing dependency: {exc}. Follow the setup steps in README.md.")
        return 1
    except subprocess.CalledProcessError:
        print("\nA step did not finish. Read the message above; existing checkpoints were not deleted.")
        return 1
    except KeyboardInterrupt:
        print("\nTraining stopped. Prefer the saved last.pt checkpoint when restarting.")
        return 130

if __name__ == "__main__":
    raise SystemExit(main())
