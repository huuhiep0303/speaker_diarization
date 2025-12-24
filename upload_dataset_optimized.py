"""
Upload Callhome & Voxconverse datasets to Modal cloud - OPTIMIZED for large files

Uploads dataset in small batches to avoid connection timeout.
Once uploaded, you can evaluate many times without re-uploading.

Usage:
    # Upload Callhome (140 files, ~2.3GB)
    modal run upload_dataset_optimized.py --dataset callhome
    
    # Upload Voxconverse (216 files)
    modal run upload_dataset_optimized.py --dataset voxconverse
    
    # Upload both
    modal run upload_dataset_optimized.py --dataset both
"""
import modal
import os
from pathlib import Path

app = modal.App("upload-dataset-optimized")

volume = modal.Volume.from_name("nemo-dataset", create_if_missing=True)
image = modal.Image.debian_slim()


@app.function(
    image=image,
    volumes={"/mnt/dataset": volume},
    timeout=7200,  # 2 hours
)
def upload_files_batch(dataset_name: str, batch_files: dict, batch_num: int, total_batches: int):
    """Upload a batch of files to Modal volume"""
    import os
    
    print(f"📦 Batch {batch_num}/{total_batches}: Uploading {len(batch_files)} files...")
    
    # Determine base directory
    base_dir = f"/mnt/dataset/{dataset_name}"
    os.makedirs(base_dir, exist_ok=True)
    
    # Write files
    for rel_path, content in batch_files.items():
        full_path = os.path.join(base_dir, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, 'wb') as f:
            f.write(content)
    
    # Commit after each batch
    volume.commit()
    
    total_size = sum(len(v) for v in batch_files.values())
    print(f"  ✓ Uploaded {len(batch_files)} files ({total_size / 1024 / 1024:.1f} MB)")
    
    return len(batch_files), total_size


@app.local_entrypoint()
def main(dataset: str = "callhome", batch_size: int = 10):
    """
    Upload dataset to Modal volume in batches
    
    Args:
        dataset: 'callhome', 'voxconverse', or 'both'
        batch_size: Number of files per batch (default: 10, smaller = safer)
    """
    from pathlib import Path
    
    print("="*80)
    print("📤 Upload Dataset to Modal Volume - OPTIMIZED")
    print("="*80)
    print(f"  Dataset: {dataset}")
    print(f"  Batch size: {batch_size} files per batch")
    print("="*80)
    print()
    
    # Determine datasets to upload
    if dataset == "both":
        datasets = ["callhome", "voxconverse"]
    else:
        datasets = [dataset]
    
    base_path = Path(r"D:\WORKSPACE\VJ\speaker-diarization\realtime\dataset")
    
    for ds_name in datasets:
        print(f"\n{'='*80}")
        print(f"📊 Uploading {ds_name.upper()}")
        print(f"{'='*80}\n")
        
        # Get dataset path
        if ds_name == "callhome":
            dataset_path = base_path / "callhome"
            audio_dir = dataset_path / "audio"
            label_dir = dataset_path / "labels"
        else:
            dataset_path = base_path / "voxconverse_dev"
            audio_dir = dataset_path / "audio"
            label_dir = dataset_path / "rttm"
        
        if not dataset_path.exists():
            print(f"❌ Dataset not found: {dataset_path}")
            continue
        
        print(f"📂 Loading files from: {dataset_path}")
        
        # Collect all files
        all_files = {}
        
        # Load audio files
        audio_files = sorted(list(audio_dir.glob("*.wav")))
        print(f"  Found {len(audio_files)} audio files")
        
        for audio_file in audio_files:
            rel_path = f"audio/{audio_file.name}"
            with open(audio_file, 'rb') as f:
                all_files[rel_path] = f.read()
        
        # Load label files
        if ds_name == "callhome":
            label_files = sorted(list(label_dir.glob("*.rttm")))
            label_prefix = "labels"
        else:
            label_files = sorted(list(label_dir.glob("*.rttm")))
            label_prefix = "rttm"
        
        print(f"  Found {len(label_files)} label files")
        
        for label_file in label_files:
            rel_path = f"{label_prefix}/{label_file.name}"
            with open(label_file, 'rb') as f:
                all_files[rel_path] = f.read()
        
        total_size = sum(len(v) for v in all_files.values())
        print(f"  Total: {len(all_files)} files ({total_size / 1024 / 1024:.1f} MB)")
        print()
        
        # Split into batches
        file_items = list(all_files.items())
        batches = []
        for i in range(0, len(file_items), batch_size):
            batch = dict(file_items[i:i+batch_size])
            batches.append(batch)
        
        print(f"📦 Uploading in {len(batches)} batches...")
        print()
        
        # Upload each batch
        total_uploaded = 0
        for i, batch in enumerate(batches, 1):
            print(f"Batch {i}/{len(batches)}:")
            
            count, size = upload_files_batch.remote(
                ds_name,
                batch,
                i,
                len(batches)
            )
            
            total_uploaded += count
            print(f"  Progress: {total_uploaded}/{len(all_files)} files uploaded")
            print()
        
        print("="*80)
        print(f"✅ {ds_name.upper()} upload completed!")
        print("="*80)
        print(f"  Files: {len(all_files)}")
        print(f"  Size: {total_size / 1024 / 1024:.1f} MB")
        print()
    
    print("\n" + "="*80)
    print("✅ ALL UPLOADS COMPLETED!")
    print("="*80)
    print("\n💡 Now you can evaluate without uploading:")
    print("  modal run --detach eval_finetuned_diarization.py --dataset callhome")
    print("  modal run --detach eval_finetuned_diarization.py --dataset voxconverse")
    print("  modal run --detach eval_finetuned_diarization.py --dataset both")
    print()
