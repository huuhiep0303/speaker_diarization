"""
Upload Voxconverse-dev dataset to Modal cloud for NeMo diarization evaluation

Dataset structure:
- audio/*.wav: Audio files (có overlap, speakers chồng nhau)
- rttm/*.rttm: RTTM annotation files

Usage:
    modal run upload_voxconverse.py
"""
import modal
import os
from pathlib import Path

app = modal.App("upload-voxconverse-dataset")

volume = modal.Volume.from_name(
    "nemo-dataset",
    create_if_missing=True,
)

# Path to local voxconverse dataset
LOCAL_DATASET = Path(r"D:\WORKSPACE\VJ\speaker-diarization\realtime\dataset\voxconverse_dev")

image = modal.Image.debian_slim()

@app.function(
    image=image,
    volumes={"/mnt/dataset": volume},
    timeout=3600,  # 1 hour for upload
)
def upload_batch_to_volume(batch_name: str, files_dict: dict):
    """Upload batch of files to Modal volume"""
    import os
    
    print(f"📦 Uploading {batch_name} ({len(files_dict)} files)...")
    
    # Create base directory
    base_dir = "/mnt/dataset/voxconverse_dev"
    os.makedirs(base_dir, exist_ok=True)
    
    # Write files
    files_written = 0
    for rel_path, content in files_dict.items():
        full_path = os.path.join(base_dir, rel_path)
        
        # Create parent directories
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # Write file
        with open(full_path, 'wb') as f:
            f.write(content)
        files_written += 1
    
    # Commit changes to volume
    volume.commit()
    
    print(f"✅ {batch_name}: uploaded {files_written} files")
    return files_written

@app.local_entrypoint()
def main():
    """Upload voxconverse-dev dataset to Modal volume"""
    
    # Check if dataset exists
    if not LOCAL_DATASET.exists():
        print(f"❌ Dataset not found: {LOCAL_DATASET}")
        print("\nExpected structure:")
        print("  dataset/voxconverse_dev/")
        print("    ├── audio/")
        print("    │   ├── abjxc.wav")
        print("    │   ├── afjiv.wav")
        print("    │   └── ...")
        print("    └── rttm/")
        print("        ├── abjxc.rttm")
        print("        ├── afjiv.rttm")
        print("        └── ...")
        return
    
    print("=" * 80)
    print("📦 Uploading Voxconverse-dev Dataset to Modal Volume")
    print("=" * 80)
    print(f"  Local path:  {LOCAL_DATASET}")
    print(f"  Remote path: /mnt/dataset/voxconverse_dev")
    print(f"  Dataset type: High overlap (speakers chồng nhau)")
    print("  Note: Large dataset (~2.2GB), will upload in smaller batches")
    print("=" * 80)
    print()
    
    # Scan files first (don't load into memory yet)
    print("📁 Scanning dataset files...")
    
    audio_dir = LOCAL_DATASET / "audio"
    rttm_dir = LOCAL_DATASET / "rttm"
    
    audio_files = sorted(list(audio_dir.glob("*.wav"))) if audio_dir.exists() else []
    rttm_files = sorted(list(rttm_dir.glob("*.rttm"))) if rttm_dir.exists() else []
    
    total_files = len(audio_files) + len(rttm_files)
    num_batches = (len(audio_files) + 29) // 30 + 1  # 30 files per batch + 1 for RTTM
    
    print(f"\n✓ Found {total_files} files ({len(audio_files)} audio, {len(rttm_files)} rttm)")
    print(f"✓ Will upload in {num_batches} batches")
    print()
    
    # Upload batches (read and upload immediately, don't store all in memory)
    print(f"📤 Uploading to Modal volume...")
    print("   Each batch uploads immediately to avoid memory issues...")
    print()
    
    # Upload audio files in batches of 30
    batch_size = 30
    for i in range(0, len(audio_files), batch_size):
        batch_num = i // batch_size + 1
        batch_files = audio_files[i:i+batch_size]
        batch_name = f"audio_batch_{batch_num}"
        
        print(f"  📦 [{batch_num}/{(len(audio_files) + batch_size - 1) // batch_size}] Reading {len(batch_files)} audio files into memory...")
        
        # Read batch into memory
        batch_dict = {}
        for wav_file in batch_files:
            rel_path = f"audio/{wav_file.name}"
            with open(wav_file, 'rb') as f:
                batch_dict[rel_path] = f.read()
        
        # Upload immediately and free memory
        print(f"  📤 [{batch_num}] Uploading {batch_name}...")
        upload_batch_to_volume.remote(batch_name, batch_dict)
        del batch_dict  # Free memory immediately
        print(f"  ✅ [{batch_num}] {batch_name} uploaded and memory freed")
    
    # Upload RTTM files (small files, can upload in one batch)
    if rttm_files:
        print(f"\n  📦 Reading {len(rttm_files)} RTTM files into memory...")
        rttm_batch = {}
        for rttm_file in rttm_files:
            rel_path = f"rttm/{rttm_file.name}"
            with open(rttm_file, 'rb') as f:
                rttm_batch[rel_path] = f.read()
        
        print(f"  📤 Uploading RTTM files...")
        upload_batch_to_volume.remote("rttm_files", rttm_batch)
        del rttm_batch  # Free memory
        print(f"  ✅ RTTM files uploaded")
    
    print()
    print("=" * 80)
    print("✅ Voxconverse-dev dataset uploaded successfully to Modal volume!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Evaluate on Voxconverse:")
    print("     modal run eval_diarization_modal.py --dataset voxconverse")
    print()
    print("  2. Check volume contents:")
    print("     modal volume ls nemo-dataset/voxconverse_dev")
    print()
