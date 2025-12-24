"""
Upload Datasets to Modal Cloud for NeMo Diarization Finetuning

This script uploads the following datasets to Modal's nemo-dataset volume:
- Training: voxconverse_dev, jvs_ver1, callhome_eng (70%)
- Testing: voxconverse_test, callhome_jpn, callhome_eng (30%)

Usage:
    modal run upload_nemo_datasets.py
"""

import modal
import os
from pathlib import Path

app = modal.App("upload-nemo-datasets")

# Simple image for file operations
image = modal.Image.debian_slim(python_version="3.10").pip_install("soundfile")

# Dataset volume
dataset_volume = modal.Volume.from_name("nemo-dataset", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/dataset": dataset_volume},
    timeout=3600,
)
def upload_file_batch(dataset_name: str, files_batch: list, dataset_type: str, batch_num: int):
    """Upload a batch of files to Modal volume"""
    import os
    
    # Determine base directory based on dataset type (train/test)
    base_dir = f"/dataset/{dataset_type}/{dataset_name}"
    os.makedirs(base_dir, exist_ok=True)
    
    # Write files with full directory structure
    files_written = 0
    # for rel_path, content in files_batch:
    #     full_path = os.path.join(base_dir, rel_path)
        
    #     # Create parent directories
    #     os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
    #     # Write file
    #     with open(full_path, 'wb') as f:
    #         f.write(content)
    #     files_written += 1
    for rel_path, content in files_batch:
        full_path = os.path.join(base_dir, rel_path)

        # ✅ SKIP nếu file đã tồn tại
        if os.path.exists(full_path):
            continue

        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        with open(full_path, 'wb') as f:
            f.write(content)

        files_written += 1

    
    # Commit changes to volume
    dataset_volume.commit()
    
    return files_written


def scan_dataset_files_generator(local_path, dataset_type: str):
    """Generate dataset files one by one (memory efficient)"""
    
    if dataset_type == "voxconverse":
        # Voxconverse: audio/ + rttm/
        audio_dir = local_path / "audio"
        rttm_dir = local_path / "rttm"
        
        if audio_dir.exists():
            for audio_file in audio_dir.glob("*.wav"):
                rel_path = f"audio/{audio_file.name}"
                with open(audio_file, 'rb') as f:
                    yield (rel_path, f.read())
        
        if rttm_dir.exists():
            for rttm_file in rttm_dir.glob("*.rttm"):
                rel_path = f"rttm/{rttm_file.name}"
                with open(rttm_file, 'rb') as f:
                    yield (rel_path, f.read())
    
    elif dataset_type == "callhome":
        # Callhome: audio/ + labels/
        audio_dir = local_path / "audio"
        labels_dir = local_path / "labels"
        
        if audio_dir.exists():
            for audio_file in audio_dir.glob("*.wav"):
                rel_path = f"audio/{audio_file.name}"
                with open(audio_file, 'rb') as f:
                    yield (rel_path, f.read())
        
        if labels_dir.exists():
            for label_file in labels_dir.glob("*.rttm"):
                rel_path = f"labels/{label_file.name}"
                with open(label_file, 'rb') as f:
                    yield (rel_path, f.read())
    
    elif dataset_type == "jvs":
        # JVS: speaker folders with parallel100/wav24kHz16bit/*.wav
        import soundfile as sf
        
        # First, yield all audio files
        for speaker_dir in local_path.iterdir():
            if not speaker_dir.is_dir() or not speaker_dir.name.startswith('jvs'):
                continue
            
            # Process parallel100 folder
            parallel_folder = speaker_dir / "parallel100" / "wav24kHz16bit"
            if parallel_folder.exists():
                for wav_file in parallel_folder.glob("*.wav"):
                    # Create audio/ structure with speaker_id prefix
                    rel_path = f"audio/{speaker_dir.name}_{wav_file.stem}.wav"
                    with open(wav_file, 'rb') as f:
                        yield (rel_path, f.read())
        
        # Then create and yield single RTTM file
        rttm_lines = []
        for speaker_dir in local_path.iterdir():
            if not speaker_dir.is_dir() or not speaker_dir.name.startswith('jvs'):
                continue
            
            speaker_id = speaker_dir.name
            parallel_folder = speaker_dir / "parallel100" / "wav24kHz16bit"
            
            if parallel_folder.exists():
                for wav_file in parallel_folder.glob("*.wav"):
                    try:
                        info = sf.info(str(wav_file))
                        duration = info.duration
                        audio_name = f"{speaker_id}_{wav_file.stem}"
                        
                        rttm_lines.append(
                            f"SPEAKER {audio_name} 1 0.0 {duration:.3f} <NA> <NA> {speaker_id} <NA> <NA>\n"
                        )
                    except:
                        continue
        
        # Yield RTTM content
        if rttm_lines:
            rttm_content = ''.join(rttm_lines).encode('utf-8')
            yield ("rttm/jvs_all.rttm", rttm_content)


def scan_callhome_split_generator(local_path, split_ratio: float = 1.0, split_offset: float = 0.0):
    """Generate callhome files with train/test split (memory efficient)"""
    
    audio_dir = local_path / "audio"
    labels_dir = local_path / "labels"
    
    if not audio_dir.exists():
        return
    
    # Get all audio files and sort for consistent splitting
    all_audio_files = sorted(list(audio_dir.glob("*.wav")))
    
    # Calculate split indices
    total_files = len(all_audio_files)
    start_idx = int(total_files * split_offset)
    end_idx = int(total_files * (split_offset + split_ratio))
    
    # Select files for this split
    audio_files = all_audio_files[start_idx:end_idx]
    
    print(f"      Split: {start_idx}-{end_idx} ({len(audio_files)} / {total_files} files)")
    
    # Yield audio files
    for audio_file in audio_files:
        rel_path = f"audio/{audio_file.name}"
        with open(audio_file, 'rb') as f:
            yield (rel_path, f.read())
    
    # Yield corresponding label files
    if labels_dir.exists():
        for audio_file in audio_files:
            file_id = audio_file.stem.replace("audio_", "")
            label_file = labels_dir / f"labels_{file_id}.rttm"
            
            if label_file.exists():
                rel_path = f"labels/{label_file.name}"
                with open(label_file, 'rb') as f:
                    yield (rel_path, f.read())


@app.local_entrypoint()
def main():
    """Main entry point"""
    
    print("\n🚀 Starting dataset upload to Modal cloud...")
    print()
    
    local_base = Path("D:/WORKSPACE/VJ/speaker-diarization/realtime/dataset")
    
    # Dataset configurations
    datasets_config = {
        "train": [
            # {
            #     "name": "voxconverse_dev",
            #     "local": local_base / "voxconverse_dev",
            #     "type": "voxconverse",
            # },
            # {
            #     "name": "jvs_ver1",
            #     "local": local_base / "jvs_ver1" / "jvs_ver1",
            #     "type": "jvs",
            # },
            # {
            #     "name": "callhome_eng",
            #     "local": local_base / "callhome_eng",
            #     "type": "callhome",
            #     "split_ratio": 0.7,
            # },
        ],
        "test": [
            {
                "name": "voxconverse_test",
                "local": local_base / "voxconverse_test",
                "type": "voxconverse",
            },
            # {
            #     "name": "callhome_jpn",
            #     "local": local_base / "callhome_jpn",
            #     "type": "callhome",
            # },
            # {
            #     "name": "callhome_eng",
            #     "local": local_base / "callhome_eng",
            #     "type": "callhome",
            #     "split_ratio": 0.3,
            #     "split_offset": 0.7,
            # },
        ],
    }
    
    print("=" * 80)
    print("📦 UPLOADING DATASETS TO MODAL CLOUD")
    print("=" * 80)
    print()
    
    # === UPLOAD TRAINING DATASETS ===
    print("🔄 UPLOADING TRAINING DATASETS")
    print("=" * 80)
    
    BATCH_SIZE = 5  # Upload 10 files at a time to avoid memory issues
    
    for dataset_info in datasets_config["train"]:
        dataset_name = dataset_info["name"]
        local_path = dataset_info["local"]
        dataset_type = dataset_info["type"]
        
        print(f"\n📁 Processing: {dataset_name}")
        print(f"   Local: {local_path}")
        
        if not local_path.exists():
            print(f"   ❌ Local path not found!")
            continue
        
        print(f"   📊 Scanning and uploading files in batches...")
        
        # Get file generator based on dataset type
        if dataset_type == "callhome" and "split_ratio" in dataset_info:
            file_generator = scan_callhome_split_generator(
                local_path,
                split_ratio=dataset_info.get("split_ratio", 1.0),
                split_offset=dataset_info.get("split_offset", 0.0),
            )
        else:
            file_generator = scan_dataset_files_generator(local_path, dataset_type)
        
        # Upload in batches
        batch = []
        batch_num = 0
        total_uploaded = 0
        
        for file_item in file_generator:
            batch.append(file_item)
            
            # Upload when batch is full
            if len(batch) >= BATCH_SIZE:
                batch_num += 1
                print(f"      Batch {batch_num}: uploading {len(batch)} files...")
                uploaded = upload_file_batch.remote(dataset_name, batch, "train", batch_num)
                total_uploaded += uploaded
                batch = []
        
        # Upload remaining files
        if batch:
            batch_num += 1
            print(f"      Batch {batch_num}: uploading {len(batch)} files...")
            uploaded = upload_file_batch.remote(dataset_name, batch, "train", batch_num)
            total_uploaded += uploaded
        
        print(f"   ✅ {dataset_name}: uploaded {total_uploaded} files")
    
    print()
    print("=" * 80)
    
    # === UPLOAD TESTING DATASETS ===
    print("\n🔄 UPLOADING TESTING DATASETS")
    print("=" * 80)
    
    for dataset_info in datasets_config["test"]:
        dataset_name = dataset_info["name"]
        local_path = dataset_info["local"]
        dataset_type = dataset_info["type"]
        
        print(f"\n📁 Processing: {dataset_name}")
        print(f"   Local: {local_path}")
        
        if not local_path.exists():
            print(f"   ❌ Local path not found!")
            continue
        
        print(f"   📊 Scanning and uploading files in batches...")
        
        # Get file generator based on dataset type
        if dataset_type == "callhome" and "split_ratio" in dataset_info:
            file_generator = scan_callhome_split_generator(
                local_path,
                split_ratio=dataset_info.get("split_ratio", 1.0),
                split_offset=dataset_info.get("split_offset", 0.0),
            )
        else:
            file_generator = scan_dataset_files_generator(local_path, dataset_type)
        
        # Upload in batches
        batch = []
        batch_num = 0
        total_uploaded = 0
        
        for file_item in file_generator:
            batch.append(file_item)
            
            # Upload when batch is full
            if len(batch) >= BATCH_SIZE:
                batch_num += 1
                print(f"      Batch {batch_num}: uploading {len(batch)} files...")
                uploaded = upload_file_batch.remote(dataset_name, batch, "test", batch_num)
                total_uploaded += uploaded
                batch = []
        
        # Upload remaining files
        if batch:
            batch_num += 1
            print(f"      Batch {batch_num}: uploading {len(batch)} files...")
            uploaded = upload_file_batch.remote(dataset_name, batch, "test", batch_num)
            total_uploaded += uploaded
        
        print(f"   ✅ {dataset_name}: uploaded {total_uploaded} files")
    
    print()
    print("=" * 80)
    print("✅ UPLOAD COMPLETED")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Check volume: modal volume ls nemo-dataset")
    print("  2. Start training: modal run finetune_nemo_vad.py")
    print()
