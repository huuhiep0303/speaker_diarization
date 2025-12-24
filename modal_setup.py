"""
NeMo Speaker Fine-tuning on Modal - Train speaker model on GPU cloud
Yêu cầu: modal đã được cài đặt và setup (modal setup)
"""
import modal
import os
import io
import tarfile
from pathlib import Path

# Tạo Modal App
app = modal.App("nemo-speaker-finetuning")

# Định nghĩa image với dependencies cho NeMo training
# CRITICAL: Order matters! Install in specific order to avoid conflicts
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "libsndfile1",
        "ffmpeg",
        "sox",
        "libsox-dev",
        "git",
        "build-essential",
    )
    # Step 1: Core build tools
    .pip_install(
        "pip==23.3.2",
        "setuptools==69.0.3",
        "wheel==0.42.0",
        "Cython==3.0.8",
    )
    # Step 2: NumPy 1.x (MUST be first to prevent override)
    .pip_install(
        "numpy==1.24.3",
    )
    # Step 3: Data libraries with compatible versions
    .pip_install(
        "pyarrow==14.0.1",  # Compatible with datasets and has PyExtensionType
        "datasets==2.16.1",  # Compatible with NeMo 1.23.0
    )
    # Step 4: PyTorch
    .pip_install(
        "torch==2.1.0",
        "torchaudio==2.1.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    # Step 5: HuggingFace stack
    .pip_install(
        "huggingface-hub==0.20.3",
        "transformers==4.36.2",
        "tokenizers==0.15.0",
    )
    # Step 6: PyTorch Lightning
    .pip_install(
        "pytorch-lightning==2.1.0",
        "torchmetrics==1.2.1",
    )
    # Step 7: Audio and ML libs
    .pip_install(
        "soundfile==0.12.1",
        "librosa==0.10.1",
        "scikit-learn==1.3.2",
    )
    # Step 8: Visualization
    .pip_install(
        "matplotlib==3.8.2",
        "seaborn==0.13.0",
    )
    # Step 9: Utils
    .pip_install(
        "tqdm==4.66.1",
        "scipy==1.11.4",
    )
    # Step 10: NeMo last (will respect existing numpy)
    .pip_install(
        "nemo_toolkit[asr]==1.23.0",
    )
)

# Tạo Volumes
dataset_volume = modal.Volume.from_name("nemo-dataset", create_if_missing=True)
results_volume = modal.Volume.from_name("nemo-results", create_if_missing=True)
@app.function(
    image=image,
    gpu="A10G",  # GPU A10G (24GB VRAM) - đủ cho NeMo training
    timeout=28800,  # 8 hours timeout for large datasets
    volumes={
        "/dataset": dataset_volume,  # Mount dataset volume
        "/results": results_volume,  # Mount results volume
    },
    memory=32768,  # 32GB RAM
    cpu=8.0,
)
def train_nemo_speaker(
    dataset_tar_bytes: bytes,
    script_content: str,
    epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    max_speakers: int = None,
    use_uploaded_dataset: bool = False,
):
    """
    Fine-tune NeMo speaker model trên Modal GPU
    
    Args:
        dataset_tar_bytes: Dataset compressed as tar.gz bytes
        script_content: Content of finetune_nemo_speaker.py
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        max_speakers: Maximum number of speakers to use
    """
    import subprocess
    import shutil
    
    print("🚀 Starting NeMo Speaker Fine-tuning on Modal GPU")
    print("=" * 80)
    print(f"⚙️  GPU: A10G (24GB VRAM)")
    print(f"⚙️  RAM: 32GB")
    print(f"⚙️  CPU: 8 cores")
    print(f"⚙️  Epochs: {epochs}")
    print(f"⚙️  Batch size: {batch_size}")
    print(f"⚙️  Learning rate: {learning_rate}")
    if max_speakers:
        print(f"⚙️  Max speakers: {max_speakers}")
    print("=" * 80)
    
    # Setup directories
    workspace_dir = "/tmp/workspace"
    results_dir = "/results"
    
    os.makedirs(workspace_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Use dataset from volume or extract from bytes
    if use_uploaded_dataset:
        print(f"\n📦 Using pre-uploaded dataset from Modal volume...")
        dataset_dir = "/dataset/jvs_ver1"
        if not os.path.exists(dataset_dir):
            raise RuntimeError(f"Dataset not found in volume: {dataset_dir}. Please upload first with: python upload_dataset.py")
        extracted_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        print(f"✓ Using dataset from {dataset_dir}")
        print(f"✓ Found {len(extracted_dirs)} directories: {extracted_dirs[:5]}...")
    else:
        dataset_dir = "/tmp/dataset"
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Extract dataset
        print(f"\n📦 Extracting dataset ({len(dataset_tar_bytes):,} bytes)...")
        tar_buffer = io.BytesIO(dataset_tar_bytes)
        with tarfile.open(fileobj=tar_buffer, mode="r:gz") as tar:
            tar.extractall(path=dataset_dir)
        print(f"✓ Dataset extracted to {dataset_dir}")
        
        # List extracted files
        extracted_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        print(f"✓ Found {len(extracted_dirs)} directories: {extracted_dirs[:5]}...")
    
    # Write training script
    training_script = os.path.join(workspace_dir, "finetune_nemo_speaker.py")
    print(f"\n📝 Writing training script to {training_script}")
    with open(training_script, 'w', encoding='utf-8') as f:
        f.write(script_content)
    print(f"✓ Training script written ({len(script_content)} bytes)")
    
    # Build command
    cmd = [
        "python", "-u", training_script,
        "--dataset", dataset_dir,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(learning_rate),
        "--output_dir", results_dir,
    ]
    
    if max_speakers:
        cmd.extend(["--max_speakers", str(max_speakers)])
    
    print("\n🏋️  Starting training...")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80)
    print()
    
    # Run training and write to log file
    # Don't stream to avoid client disconnect issues
    log_file_path = os.path.join(results_dir, "training.log")
    
    with open(log_file_path, "w", buffering=1) as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=workspace_dir,
        )
        
        print("⏳ Training in progress...")
        print(f"📝 Logs being written to: {log_file_path}")
        print("💡 This may take a while. Check Modal dashboard for progress.")
        print()
        
        # Wait for completion
        process.wait()
    
    if process.returncode != 0:
        print("\n" + "=" * 80)
        print("❌ TRAINING FAILED!")
        print("=" * 80)
        print(f"Exit code: {process.returncode}")
        print(f"Log file: {log_file_path}")
        
        # Try to show last 50 lines of log
        print("\n📋 Last 50 lines of log:")
        print("-" * 80)
        try:
            with open(log_file_path, 'r') as f:
                lines = f.readlines()
                for line in lines[-50:]:
                    print(line, end="")
        except Exception as e:
            print(f"Could not read log: {e}")
        print("-" * 80)
        
        results_volume.commit()
        raise RuntimeError(f"Training failed with exit code {process.returncode}")
    
    print("\n" + "=" * 80)
    print("✅ Training completed successfully!")
    print("=" * 80)
    
    # Copy all results from /tmp/workspace/finetuned_models to /results
    print("\n📦 Copying results to Modal volume...")
    workspace_results = os.path.join(workspace_dir, "finetuned_models")
    
    if os.path.exists(workspace_results):
        # Copy checkpoints
        src_checkpoints = os.path.join(workspace_results, "checkpoints")
        dst_checkpoints = os.path.join(results_dir, "checkpoints")
        if os.path.exists(src_checkpoints):
            shutil.copytree(src_checkpoints, dst_checkpoints, dirs_exist_ok=True)
            print(f"✓ Copied checkpoints to {dst_checkpoints}")
        
        # Copy plots
        src_plots = os.path.join(workspace_results, "plots")
        dst_plots = os.path.join(results_dir, "plots")
        if os.path.exists(src_plots):
            shutil.copytree(src_plots, dst_plots, dirs_exist_ok=True)
            print(f"✓ Copied plots to {dst_plots}")
        
        # Copy logs
        src_logs = os.path.join(workspace_results, "logs")
        dst_logs = os.path.join(results_dir, "logs")
        if os.path.exists(src_logs):
            shutil.copytree(src_logs, dst_logs, dirs_exist_ok=True)
            print(f"✓ Copied logs to {dst_logs}")
    
    # Commit results to volume
    print("\n💾 Committing results to Modal volume...")
    results_volume.commit()
    print("✓ Results committed to volume 'nemo-results'")
    
    # Collect results
    results = {
        "status": "success",
        "output_dir": results_dir,
    }
    
    # Find checkpoints
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    if os.path.exists(checkpoint_dir):
        checkpoints = os.listdir(checkpoint_dir)
        results["checkpoints"] = checkpoints
        print(f"\n📁 Checkpoints: {len(checkpoints)} files")
    
    # Find plots
    plot_dir = os.path.join(results_dir, "plots")
    if os.path.exists(plot_dir):
        plots = os.listdir(plot_dir)
        results["plots"] = plots
        print(f"📊 Plots: {plots}")
    
    # Read final results
    results_json = os.path.join(results_dir, "logs", "final_results.json")
    if os.path.exists(results_json):
        import json
        with open(results_json, 'r') as f:
            training_results = json.load(f)
        results["training_results"] = training_results
        print(f"\n📊 Final Results:")
        print(f"  Test Accuracy: {training_results.get('test_accuracy', 0)*100:.2f}%")
        print(f"  Test F1: {training_results.get('test_f1', 0):.4f}")
        print(f"  Best Val Acc: {training_results.get('best_val_acc', 0)*100:.2f}%")
    
    # Print download instructions
    print("\n" + "=" * 80)
    print("📥 TO DOWNLOAD RESULTS TO LOCAL:")
    print("=" * 80)
    print("modal volume get nemo-results /checkpoints ./finetuned_models/checkpoints")
    print("modal volume get nemo-results /plots ./finetuned_models/plots")
    print("modal volume get nemo-results /logs ./finetuned_models/logs")
    print("modal volume get nemo-results /training.log ./finetuned_models/training.log")
    print("=" * 80)
    
    return results
@app.local_entrypoint()
def main(
    dataset_path: str = "dataset/jvs_ver1/jvs_ver1",
    epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    max_speakers: int = None,
    use_volume: bool = False,
):
    """
    Entry point để chạy từ local machine
    
    Usage:
        # Small dataset (upload each time)
        modal run modal_setup.py --max-speakers 10
        
        # Large dataset (use pre-uploaded volume) - RECOMMENDED for 30+ speakers
        python upload_dataset.py  # Upload once
        modal run modal_setup.py --use-volume --max-speakers 50 --epochs 30
        
        # Other options
        modal run modal_setup.py --epochs 100 --batch-size 32
        modal run modal_setup.py --dataset-path "../dataset/jvs_ver1/jvs_ver1"
    """
    print("=" * 80)
    print("🌐 NeMo Speaker Fine-tuning on Modal Cloud")
    print("=" * 80)
    print(f"📁 Dataset: {dataset_path}")
    print(f"📊 Epochs: {epochs}")
    print(f"📊 Batch size: {batch_size}")
    print(f"📊 Learning rate: {learning_rate}")
    if max_speakers:
        print(f"📊 Max speakers: {max_speakers}")
    if use_volume:
        print(f"📦 Using pre-uploaded dataset from Modal volume")
        print(f"   (Skipping dataset upload - recommended for large datasets)")
    print("=" * 80)
    print()
    
    # Prepare dataset bytes
    tar_bytes = b""
    
    if use_volume:
        print("✓ Using pre-uploaded dataset from Modal volume")
        print("   Make sure you've uploaded with: python upload_dataset.py")
        print()
    else:
        # Check dataset exists
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            print(f"❌ Error: Dataset not found: {dataset_path}")
            print("\nPlease provide the correct path to JVS dataset.")
            print("Expected structure: jvs_ver1/jvs001/, jvs_ver1/jvs002/, ...")
            print("\n💡 TIP: For large datasets (30+ speakers), use --use-volume flag:")
            print("   1. python upload_dataset.py  (upload once)")
            print("   2. modal run modal_setup.py --use-volume --max-speakers 50")
            return
        
        # Compress dataset to tar.gz
        print("📦 Compressing dataset for upload...")
        tar_buffer = io.BytesIO()
        
        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
            speakers = [
                d for d in sorted(dataset_path.iterdir())
                if d.is_dir() and d.name.startswith("jvs")
            ]

            if max_speakers:
                speakers = speakers[:max_speakers]

            for speaker_dir in speakers:
                print(f"  + Adding {speaker_dir.name}")
                tar.add(speaker_dir, arcname=speaker_dir.name)
        
        tar_bytes = tar_buffer.getvalue()
        print(f"✓ Compressed dataset: {len(tar_bytes):,} bytes ({len(tar_bytes) / 1024 / 1024:.1f} MB)")
        
        if len(tar_bytes) > 500 * 1024 * 1024:  # > 500MB
            print()
            print("⚠️  WARNING: Large dataset detected (>500MB)")
            print("   Uploading large datasets may cause connection timeouts.")
            print("   Consider using --use-volume flag instead:")
            print("     1. python upload_dataset.py  (upload once)")
            print("     2. modal run modal_setup.py --use-volume --max-speakers 50")
        print()
    
    # Upload training script
    print("📤 Uploading training script and dataset to Modal...")
    
    # Read training script
    script_path = Path(__file__).parent / "finetune_nemo_speaker.py"
    if not script_path.exists():
        print(f"❌ Error: Training script not found: {script_path}")
        return
    
    with open(script_path, 'r', encoding='utf-8') as f:
        script_content = f.read()
    
    print(f"✓ Loaded training script: {script_path.name}")
    print()
    
    # Deploy to Modal
    print("🚀 Deploying to Modal GPU...")
    print("=" * 80)
    
    result = train_nemo_speaker.remote(
        dataset_tar_bytes=tar_bytes,
        script_content=script_content,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_speakers=max_speakers,
        use_uploaded_dataset=use_volume,
    )
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETED!")
    print("=" * 80)
    print(f"\n📊 Status: {result['status']}")
    
    if result['status'] == 'success':
        print(f"\n📁 Results saved to Modal volume:")
        print(f"  - Checkpoints: {result.get('checkpoints', 'N/A')}")
        print(f"  - Best model: {result.get('best_model', 'N/A')}")
        print(f"  - Training curves: {result.get('plots', 'N/A')}")
        
        # Download results to local
        print("\n💾 Downloading results to local...")
        local_results_dir = Path("./finetuned_models")
        local_results_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"✓ Results downloaded to: {local_results_dir}")
        print("\n💡 To access results:")
        print(f"  - Check {local_results_dir}/ directory")
        print(f"  - Or use Modal dashboard: https://modal.com/")
    else:
        print(f"\n⚠️  Message: {result.get('message', 'Unknown status')}")
    
    print("=" * 80)