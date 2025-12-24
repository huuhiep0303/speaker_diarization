import modal
import os
import shutil
from pathlib import Path

app = modal.App("upload-nemo-dataset")

volume = modal.Volume.from_name(
    "nemo-dataset",
    create_if_missing=True,
)

# Path to local dataset (adjust if needed)
LOCAL_DATASET = Path(r"D:\WORKSPACE\VJ\speaker-diarization\realtime\dataset\jvs_ver1\jvs_ver1")

image = modal.Image.debian_slim()

@app.function(
    image=image,
    volumes={"/mnt/dataset": volume},
    timeout=3600,  # 1 hour for upload
)
def upload_to_volume(speaker_name: str, files_dict: dict):
    """Upload speaker files to Modal volume while preserving directory structure"""
    import os
    
    print(f"📦 Uploading speaker {speaker_name} ({len(files_dict)} files)...")
    
    # Create base directory
    base_dir = "/mnt/dataset/jvs_ver1"
    os.makedirs(base_dir, exist_ok=True)
    
    # Write files with full directory structure
    files_written = 0
    for rel_path, content in files_dict.items():
        full_path = os.path.join(base_dir, rel_path)
        
        # Create parent directories
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # Write file
        with open(full_path, 'wb') as f:
            f.write(content)
        files_written += 1
    
    # Verify structure (check first file to confirm path)
    if files_dict:
        first_rel_path = next(iter(files_dict.keys()))
        first_full_path = os.path.join(base_dir, first_rel_path)
        if os.path.exists(first_full_path):
            print(f"  ✓ Verified structure: {first_rel_path}")
        else:
            print(f"  ⚠️  Warning: File not found after write: {first_full_path}")
    
    # List speaker directory to verify
    speaker_dir = os.path.join(base_dir, speaker_name)
    if os.path.exists(speaker_dir):
        subdirs = [d for d in os.listdir(speaker_dir) if os.path.isdir(os.path.join(speaker_dir, d))]
        print(f"  ✓ Speaker directory has subdirs: {subdirs}")
    
    # Commit changes to volume
    volume.commit()
    
    print(f"✅ Speaker {speaker_name}: uploaded {files_written} files and committed to volume")
    return files_written


@app.local_entrypoint()
def main():
    """Upload local dataset to Modal volume"""
    
    # Check if dataset exists
    if not LOCAL_DATASET.exists():
        print(f"❌ Dataset not found: {LOCAL_DATASET}")
        print("\nExpected structure:")
        print("  dataset/jvs_ver1/jvs_ver1/")
        print("    ├── jvs001/")
        print("    ├── jvs002/")
        print("    └── ...")
        return
    
    print("=" * 80)
    print("📦 Uploading Dataset to Modal Volume")
    print("=" * 80)
    print(f"  Local path:  {LOCAL_DATASET}")
    print(f"  Remote path: /mnt/dataset/jvs_ver1")
    print("=" * 80)
    print()
    
    # Collect all files to upload
    print("📁 Scanning dataset files...")
    
    # Group files by speaker to preserve directory structure
    speakers_data = {}
    total_files = 0
    total_size = 0
    
    for speaker_dir in sorted(LOCAL_DATASET.iterdir()):
        if not speaker_dir.is_dir() or not speaker_dir.name.startswith('jvs'):
            continue
        
        speaker_name = speaker_dir.name
        print(f"  + Scanning {speaker_name}...")
        
        speaker_files = {}
        
        # Scan all subdirectories (parallel100, falset10, nonpara30, whisper10)
        for sub_dir in speaker_dir.iterdir():
            if not sub_dir.is_dir():
                continue
            
            # Look for wav24kHz16bit subdirectory
            wav_dir = sub_dir / 'wav24kHz16bit'
            if wav_dir.exists():
                for wav_file in wav_dir.glob('*.wav'):
                    # Create relative path from speaker directory
                    # e.g., "jvs001/parallel100/wav24kHz16bit/BASIC5000_0001.wav"
                    rel_path = wav_file.relative_to(LOCAL_DATASET)
                    
                    # Convert Windows path to POSIX (forward slashes)
                    rel_path_posix = rel_path.as_posix()
                    
                    # Read file content
                    with open(wav_file, 'rb') as f:
                        content = f.read()
                        speaker_files[rel_path_posix] = content
                        total_size += len(content)
        
        if speaker_files:
            speakers_data[speaker_name] = speaker_files
            total_files += len(speaker_files)
    
    print(f"\n✓ Found {len(speakers_data)} speakers")
    print(f"✓ Found {total_files} audio files")
    print(f"✓ Total size: {total_size / 1024 / 1024:.1f} MB")
    print()
    
    # Upload speaker by speaker to preserve structure
    print("📤 Uploading to Modal volume...")
    
    for speaker_name, files_dict in speakers_data.items():
        print(f"  Uploading {speaker_name} ({len(files_dict)} files)...")
        upload_to_volume.remote(speaker_name, files_dict)
    
    print()
    print("=" * 80)
    print("✅ Dataset uploaded successfully to Modal volume!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Train with volume:")
    print("     modal run modal_setup.py --use-volume --max-speakers 50")
    print()
    print("  2. Check volume contents:")
    print("     modal volume ls nemo-dataset")
    print()
