"""
SURYOKTI: CNN Architecture Reference and Inference Pipeline Stub
Classifies solar PV panel surface defects into 6 operational classes.
"""

import os
import json
from pathlib import Path

EXTERNAL_IMAGE_DIR = r"C:\VITB\suryokti vision"

CLASS_NAMES = [
    "Clean",
    "Dusty",
    "Bird-drop",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered"
]

def get_class_manifest():
    manifest_path = Path(__file__).resolve().parents[1] / "data" / "class_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"classes": [{"id": i, "name": n} for i, n in enumerate(CLASS_NAMES)]}

def inspect_external_dataset(dataset_dir=EXTERNAL_IMAGE_DIR):
    """
    Checks availability of external dataset folders at C:\\VITB\\suryokti vision.
    """
    p = Path(dataset_dir)
    print(f"Checking external image directory: {p}")
    if not p.exists():
        print(f"Directory {dataset_dir} is not mounted or not accessible.")
        return False
        
    counts = {}
    for c in CLASS_NAMES:
        sub = p / c
        if sub.exists():
            files = [f for f in sub.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
            counts[c] = len(files)
        else:
            counts[c] = 0
            
    print("Class image counts:")
    for c, cnt in counts.items():
        print(f"  - {c}: {cnt} images")
    return counts

if __name__ == "__main__":
    print("=== SURYOKTI CNN Fault Classification Module ===")
    manifest = get_class_manifest()
    print(f"Loaded manifest with {len(manifest.get('classes', []))} classes.")
    inspect_external_dataset()

