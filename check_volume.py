#!/usr/bin/env python3
"""
Check Modal volume structure to verify dataset upload
"""

import modal
import os

app = modal.App("check-volume")

volume = modal.Volume.from_name("nemo-dataset")

image = modal.Image.debian_slim()

@app.function(
    image=image,
    volumes={"/mnt/dataset": volume},
)
def check_volume_structure():
    """Check and print volume directory structure"""
    import os
    
    base_path = "/mnt/dataset/jvs_ver1"
    
    print("=" * 80)
    print("📂 Modal Volume Structure Check")
    print("=" * 80)
    print(f"Base path: {base_path}")
    print()
    
    if not os.path.exists(base_path):
        print(f"❌ Path does not exist: {base_path}")
        return
    
    # List top-level directories
    print("📁 Top-level contents:")
    items = sorted(os.listdir(base_path))
    dirs = [d for d in items if os.path.isdir(os.path.join(base_path, d))]
    files = [f for f in items if os.path.isfile(os.path.join(base_path, f))]
    
    print(f"  Directories: {len(dirs)}")
    print(f"  Files: {len(files)}")
    print()
    
    if dirs:
        print("📂 Speaker directories (first 10):")
        for d in dirs[:10]:
            speaker_path = os.path.join(base_path, d)
            
            # Count subdirs in speaker
            subdirs = [s for s in os.listdir(speaker_path) if os.path.isdir(os.path.join(speaker_path, s))]
            
            # Count files recursively
            file_count = 0
            for root, _, files_in_dir in os.walk(speaker_path):
                file_count += len([f for f in files_in_dir if f.endswith('.wav')])
            
            print(f"  {d}/ - subdirs: {subdirs}, wav files: {file_count}")
    else:
        print("⚠️  No speaker directories found!")
        print()
        print("📄 Files in base path (first 20):")
        for f in files[:20]:
            print(f"  {f}")
    
    print()
    print("=" * 80)


@app.local_entrypoint()
def main():
    check_volume_structure.remote()
