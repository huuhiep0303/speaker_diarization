#!/usr/bin/env python3
"""Download SpeechBrain speaker model files"""
import os
import urllib.request
from pathlib import Path

MODEL_DIR = Path("pretrained_models/spkrec-ecapa-voxceleb")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "embedding_model.ckpt": "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/embedding_model.ckpt",
    "mean_var_norm_emb.ckpt": "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/mean_var_norm_emb.ckpt",
    "classifier.ckpt": "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/classifier.ckpt",
    "label_encoder.txt": "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/label_encoder.txt",
}

print("Downloading SpeechBrain speaker model files...")
print(f"Target directory: {MODEL_DIR.absolute()}")
print()

for filename, url in FILES.items():
    filepath = MODEL_DIR / filename
    
    if filepath.exists():
        print(f"✓ {filename} already exists, skipping")
        continue
    
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        size_mb = filepath.stat().st_size / 1024 / 1024
        print(f"  ✓ Downloaded ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

print("\nDone! Now try running: python realtime_diarization_improved.py")
