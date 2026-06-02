"""
Lance tous les modules CNN en séquence — pas de timeout
Usage: python scripts/train_all_fast.py
"""
import sys, subprocess
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]

modules = [
    ("pulmoscan", "data/pulmoscan"),
    ("osteo",     "data/osteo"),
    ("gastro",    "data/gastro"),
    ("retina",    "data/retina"),
]

for module, dataset in modules:
    print(f"\n{'='*50}")
    print(f"LANCEMENT: {module}")
    print(f"{'='*50}")
    result = subprocess.run(
        [sys.executable, "scripts/train_fast_cnn.py",
         "--module", module, "--dataset", dataset,
         "--epochs", "5", "--max_imgs", "1500"],
        cwd=str(ROOT),
        env={**__import__('os').environ, "PYTHONIOENCODING": "utf-8"},
    )
    print(f"{module}: exit code {result.returncode}")

print("\nTous les modules traités.")
