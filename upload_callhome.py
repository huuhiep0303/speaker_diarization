"""
Upload Callhome dataset to Modal cloud for NeMo diarization evaluation

Dataset structure:
- audio/*.wav: Audio files (ít overlap, speakers xen kẽ)
- labels/*.rttm: RTTM annotation files

Usage:
    modal run upload_callhome.py
"""
import modal
import os
from pathlib import Path

app = modal.App("upload-callhome-dataset")

volume = modal.Volume.from_name(
    "nemo-dataset",
    create_if_missing=True,
)

# Path to local callhome dataset
LOCAL_DATASET = Path(r"D:\WORKSPACE\VJ\speaker-diarization\realtime\dataset\callhome")

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
    base_dir = "/mnt/dataset/callhome"
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
    """Upload callhome dataset to Modal volume"""
    
    # Check if dataset exists
    if not LOCAL_DATASET.exists():
        print(f"❌ Dataset not found: {LOCAL_DATASET}")
        print("\nExpected structure:")
        print("  dataset/callhome/")
        print("    ├── audio/")
        print("    │   ├── audio_0.wav")
        print("    │   ├── audio_1.wav")
        print("    │   └── ...")
        print("    └── labels/")
        print("        ├── labels_0.rttm")
        print("        ├── labels_1.rttm")
        print("        └── ...")
        return
    
    print("=" * 80)
    print("📦 Uploading Callhome Dataset to Modal Volume")
    print("=" * 80)
    print(f"  Local path:  {LOCAL_DATASET}")
    print(f"  Remote path: /mnt/dataset/callhome")
    print(f"  Dataset type: Low overlap (speakers xen kẽ)")
    print("=" * 80)
    print()
    
    # Scan and count files first (don't load into memory yet)
    print("📁 Scanning dataset files...")
    
    audio_dir = LOCAL_DATASET / "audio"
    labels_dir = LOCAL_DATASET / "labels"
    
    audio_files_list = sorted(list(audio_dir.glob("*.wav"))) if audio_dir.exists() else []
    label_files_list = sorted(list(labels_dir.glob("*.rttm"))) if labels_dir.exists() else []
    
    total_files = len(audio_files_list) + len(label_files_list)
    print(f"\n✓ Found {total_files} files ({len(audio_files_list)} audio, {len(label_files_list)} labels)")
    print()
    
    # Upload audio files (read and upload immediately, don't store in memory)
    print(f"📤 Uploading to Modal volume...")
    
    if audio_files_list:
        print(f"  📦 Uploading audio files ({len(audio_files_list)} files)...")
        audio_batch = {}
        for wav_file in audio_files_list:
            rel_path = f"audio/{wav_file.name}"
            with open(wav_file, 'rb') as f:
                audio_batch[rel_path] = f.read()
        
        # Upload and immediately free memory
        upload_batch_to_volume.remote("audio", audio_batch)
        del audio_batch  # Free memory
        print(f"  ✅ Audio files uploaded")
    
    # Upload label files
    if label_files_list:
        print(f"  📦 Uploading label files ({len(label_files_list)} files)...")
        label_batch = {}
        for rttm_file in label_files_list:
            rel_path = f"labels/{rttm_file.name}"
            with open(rttm_file, 'rb') as f:
                label_batch[rel_path] = f.read()
        
        # Upload and immediately free memory
        upload_batch_to_volume.remote("labels", label_batch)
        del label_batch  # Free memory
        print(f"  ✅ Label files uploaded")
    
    print()
    print("=" * 80)
    print("✅ Callhome dataset uploaded successfully to Modal volume!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Evaluate on Callhome:")
    print("     modal run eval_diarization_modal.py --dataset callhome")
    print()
    print("  2. Check volume contents:")
    print("     modal volume ls nemo-dataset/callhome")
    print()
