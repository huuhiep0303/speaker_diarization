"""
Finetune NeMo VAD Model for Speaker Diarization

This script fine-tunes ONLY the Voice Activity Detection (VAD) component 
of NeMo's diarization pipeline using MarbleNet VAD model.

Training datasets:
- voxconverse_dev
- jvs_ver1 (Japanese)
- callhome_eng (70%)

Usage:
    modal run finetune_nemo_vad.py --epochs 50 --batch-size 32
    modal run finetune_nemo_vad.py --resume best_vad_model.pt
"""

import modal
import os
from pathlib import Path
from datetime import datetime

app = modal.App("finetune-nemo-vad")

# Modal image with NeMo
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
    .pip_install(
        "webdataset==0.2.86",
        "braceexpand==0.1.7",
    )
)

# Volumes
dataset_volume = modal.Volume.from_name("nemo-dataset", create_if_missing=True)
results_volume = modal.Volume.from_name("nemo-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=14400,  # 4 hours
    volumes={
        "/dataset": dataset_volume,
        "/results": results_volume,
    },
    memory=32768,
    cpu=8.0,
)
def finetune_vad(
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    resume_checkpoint: str = None,
):
    """
    Finetune NeMo VAD (MarbleNet) model
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        resume_checkpoint: Path to checkpoint to resume from
    """
    import json
    import numpy as np
    import torch
    from tqdm import tqdm
    import soundfile as sf
    from pathlib import Path
    
    # NeMo imports
    from nemo.collections.asr.models import EncDecClassificationModel
    from nemo.core.config import hydra_runner
    from omegaconf import OmegaConf, DictConfig
    
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from torch.utils.data import Dataset, DataLoader
    
    print("=" * 80)
    print("🔧 FINETUNING NeMo VAD MODEL")
    print("=" * 80)
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    if resume_checkpoint:
        print(f"  Resume from: {resume_checkpoint}")
    print("=" * 80)
    print()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    print()
    
    # === PREPARE TRAINING DATA ===
    print("📦 Preparing training data...")
    train_manifest = prepare_vad_manifest(
        datasets=[
            "/dataset/train/voxconverse_dev",
            "/dataset/jvs_ver1",
            "/dataset/train/callhome_eng",
        ],
        output_path="/dataset/train_vad_manifest.json",
    )
    print(f"   ✓ Created training manifest: {len(train_manifest)} samples")
    print()
    
    # === PREPARE VALIDATION DATA ===
    print("📦 Preparing validation data...")
    val_manifest = prepare_vad_manifest(
        datasets=[
            "/dataset/test/voxconverse_test",
            "/dataset/test/callhome_eng",
            "/dataset/test/callhome_jpn",
        ],
        output_path="/dataset/val_vad_manifest.json",
    )
    print(f"   ✓ Created validation manifest: {len(val_manifest)} samples")
    print()
    
    # === LOAD PRETRAINED VAD MODEL ===
    print("📦 Loading pretrained MarbleNet VAD model...")
    
    try:
        # Load pretrained MarbleNet VAD
        vad_model = EncDecClassificationModel.from_pretrained(
            model_name="vad_multilingual_marblenet"
        )
        print("   ✓ Loaded: nvidia/vad_multilingual_marblenet")
    except Exception as e:
        print(f"   ⚠️  Error loading MarbleNet: {e}")
        print("   Trying alternative VAD model...")
        vad_model = EncDecClassificationModel.from_pretrained(
            "nvidia/vad_marblenet"
        )
        print("   ✓ Loaded: nvidia/vad_marblenet")
    
    vad_model.to(device)
    print()
    
    # === SETUP TRAINING ===
    print("⚙️  Setting up training...")
    
    # Update model config for finetuning
    # vad_model.cfg.optim.lr = learning_rate
    # vad_model.cfg.optim.name = "adam"
    # vad_model.cfg.optim.weight_decay = 0.001
    from omegaconf import OmegaConf
    vad_model.cfg.optim = OmegaConf.create({
        "name": "adam",
        "lr": learning_rate,
        "weight_decay": 0.001,
    })    
    
    # Setup data loaders
    vad_model.setup_training_data(
        train_data_config={
            "manifest_filepath": "/dataset/train_vad_manifest.json",
            "sample_rate": 16000,
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": 4,
            # "labels": ["infer"],  # Auto-infer from manifest
            "labels": ["background", "speech"],
            "augmentor": {
                "shift": {"prob": 0.5, "min_shift_ms": -5.0, "max_shift_ms": 5.0},
                "white_noise": {"prob": 0.5, "min_level": -90, "max_level": -46},
            },
        }
    )
    
    # Setup validation data
    vad_model.setup_validation_data(
        val_data_config={
            "manifest_filepath": "/dataset/val_vad_manifest.json",
            "sample_rate": 16000,
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": 2,
            # "labels": ["infer"],
            "labels": ["background", "speech"],
        }
    )
    
    print("   ✓ Data loaders configured")
    print()
    
    # === SETUP CALLBACKS ===
    checkpoint_dir = Path("/results/checkpoints/vad")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="vad_model_epoch{epoch:02d}_loss{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True,
    )
    
    # === SETUP TRAINER ===
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=10,
        val_check_interval=0.5,  # Validate twice per epoch
        gradient_clip_val=1.0,
    )
    
    print("   ✓ Trainer configured")
    print()
    
    # === TRAIN MODEL ===
    print("🚀 Starting training...")
    print("=" * 80)
    print()
    
    training_start = datetime.now()
    
    try:
        trainer.fit(vad_model, ckpt_path=resume_checkpoint)
        
        training_end = datetime.now()
        training_duration = (training_end - training_start).total_seconds()
        
        print()
        print("=" * 80)
        print("✅ TRAINING COMPLETED")
        print("=" * 80)
        print(f"   Duration: {training_duration / 60:.2f} minutes")
        print(f"   Best checkpoint: {checkpoint_callback.best_model_path}")
        print()
        
        # === SAVE FINAL MODEL ===
        final_model_path = checkpoint_dir / "best_vad_model.nemo"
        vad_model.save_to(str(final_model_path))
        print(f"💾 Saved final model: {final_model_path.name}")
        
        # === SAVE TRAINING SUMMARY ===
        summary = {
            "model_type": "vad_only",
            "base_model": "nvidia/vad_multilingual_marblenet",
            "training_datasets": [
                "voxconverse_dev",
                "jvs_ver1",
                "callhome_eng_70%",
            ],
            "config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            },
            "training": {
                "start_time": training_start.isoformat(),
                "end_time": training_end.isoformat(),
                "duration_minutes": training_duration / 60,
                "best_checkpoint": str(checkpoint_callback.best_model_path),
                "best_val_loss": float(checkpoint_callback.best_model_score),
            },
            "num_samples": len(train_manifest),
        }
        
        summary_path = Path("/results/training_summary_vad.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Saved training summary: {summary_path.name}")
        print()
        
        # Commit volume
        results_volume.commit()
        print("✅ Results committed to volume")
        print()
        
        return summary
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error log
        error_log = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat(),
        }
        
        with open("/results/training_error_vad.json", 'w') as f:
            json.dump(error_log, f, indent=2)
        
        results_volume.commit()
        
        raise


def prepare_vad_manifest(datasets: list, output_path: str):
    """
    Prepare VAD training manifest from RTTM files
    
    VAD manifest format:
    {"audio_filepath": "path/to/audio.wav", "offset": 0.0, "duration": 1.0, "label": "speech"}
    {"audio_filepath": "path/to/audio.wav", "offset": 1.5, "duration": 0.8, "label": "non-speech"}
    """
    import json
    import soundfile as sf
    from pathlib import Path
    import numpy as np
    
    print("   Creating VAD manifest from RTTM files...")
    
    manifest_lines = []
    
    for dataset_path in datasets:
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            print(f"      ⚠️  Dataset not found: {dataset_path}")
            continue
        
        print(f"      Processing: {dataset_path.name}")
        
        audio_dir = dataset_path / "audio"
        
        # Try different label directory names
        rttm_dir = dataset_path / "rttm"
        if not rttm_dir.exists():
            rttm_dir = dataset_path / "labels"
        
        if not rttm_dir.exists():
            print(f"         ⚠️  No RTTM/labels directory found")
            continue
        
        audio_files = list(audio_dir.glob("*.wav"))
        
        for audio_file in audio_files:
            # Find corresponding RTTM file
            rttm_file = rttm_dir / f"{audio_file.stem}.rttm"
            
            if not rttm_file.exists():
                # Try alternative naming (for callhome: audio_XXX.wav -> labels_XXX.rttm)
                file_id = audio_file.stem.replace("audio_", "")
                rttm_file = rttm_dir / f"labels_{file_id}.rttm"
            
            if not rttm_file.exists():
                continue
            
            # Get audio duration
            try:
                info = sf.info(str(audio_file))
                audio_duration = info.duration
            except:
                continue
            
            # Parse RTTM file to get speech segments
            speech_segments = []
            
            with open(rttm_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    start = float(parts[3])
                    duration = float(parts[4])
                    end = start + duration
                    
                    speech_segments.append((start, end))
            
            if len(speech_segments) == 0:
                continue
            
            # Merge overlapping segments
            speech_segments = merge_segments(speech_segments)
            
            # Generate VAD samples (speech + non-speech)
            # Strategy: sliding window with 1.0s duration, 0.5s hop
            
            window_size = 1.0
            hop_size = 0.5
            
            offset = 0.0
            while offset + window_size <= audio_duration:
                # Check if this window contains speech
                window_end = offset + window_size
                
                # Calculate overlap with speech segments
                speech_overlap = 0.0
                for seg_start, seg_end in speech_segments:
                    overlap_start = max(offset, seg_start)
                    overlap_end = min(window_end, seg_end)
                    
                    if overlap_end > overlap_start:
                        speech_overlap += (overlap_end - overlap_start)
                
                # Label as speech if >50% overlap
                # label = "speech" if speech_overlap / window_size > 0.5 else "non-speech"
                label = "speech" if speech_overlap / window_size > 0.5 else "background"

                
                manifest_lines.append({
                    "audio_filepath": str(audio_file),
                    "offset": offset,
                    "duration": window_size,
                    "label": label,
                })
                
                offset += hop_size
    
    print(f"      ✓ Generated {len(manifest_lines)} VAD samples")
    
    # Balance speech/non-speech samples
    # speech_samples = [s for s in manifest_lines if s["label"] == "speech"]
    # non_speech_samples = [s for s in manifest_lines if s["label"] == "non-speech"]
    speech_samples = [s for s in manifest_lines if s["label"] == "speech"]
    non_speech_samples = [s for s in manifest_lines if s["label"] == "background"]
    
    if len(speech_samples) == 0:
        raise RuntimeError("❌ No speech samples found. Check RTTM parsing!")

    if len(non_speech_samples) == 0:
        raise RuntimeError("❌ No background samples found. Check windowing!")


    
    print(f"      Speech: {len(speech_samples)}, Non-speech: {len(non_speech_samples)}")
    
    # Downsample majority class if needed
    if len(non_speech_samples) > len(speech_samples) * 2:
        import random
        random.seed(42)
        non_speech_samples = random.sample(non_speech_samples, len(speech_samples) * 2)
        print(f"      Balanced to: Speech={len(speech_samples)}, Non-speech={len(non_speech_samples)}")
    
    balanced_manifest = speech_samples + non_speech_samples
    
    # Shuffle
    import random
    random.seed(42)
    random.shuffle(balanced_manifest)
    
    # Write manifest
    with open(output_path, 'w') as f:
        for entry in balanced_manifest:
            f.write(json.dumps(entry) + '\n')
    
    return balanced_manifest


def merge_segments(segments):
    """Merge overlapping segments"""
    if len(segments) == 0:
        return []
    
    # Sort by start time
    segments = sorted(segments, key=lambda x: x[0])
    
    merged = [segments[0]]
    
    for current in segments[1:]:
        last = merged[-1]
        
        # Check if overlapping or adjacent
        if current[0] <= last[1]:
            # Merge
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    
    return merged


@app.local_entrypoint()
def main(
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    resume: str = None,
):
    """Main entry point"""
    
    print("\n🚀 Starting NeMo VAD finetuning...")
    print()
    
    result = finetune_vad.remote(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        resume_checkpoint=resume,
    )
    
    print("\n✅ VAD finetuning completed!")
    print(f"   Training duration: {result['training']['duration_minutes']:.2f} minutes")
    print(f"   Best validation loss: {result['training']['best_val_loss']:.4f}")
    print()
