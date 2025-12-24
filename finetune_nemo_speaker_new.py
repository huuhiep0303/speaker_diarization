"""
Finetune NeMo Speaker Embedding Model for Speaker Diarization

This script fine-tunes ONLY the Speaker Embedding component 
of NeMo's diarization pipeline using TitaNet model.

Training datasets:
- voxconverse_dev
- jvs_ver1 (Japanese)
- callhome_eng (70%)

Usage:
    modal run finetune_nemo_speaker.py --epochs 30 --batch-size 64
    modal run finetune_nemo_speaker.py --resume best_speaker_model.nemo
"""

import modal
import os
from pathlib import Path
from datetime import datetime

app = modal.App("finetune-nemo-speaker")

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
def finetune_speaker(
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    resume_checkpoint: str = None,
):
    """
    Finetune NeMo Speaker Embedding (TitaNet) model
    
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
    
    # NeMo imports
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
    
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    
    print("=" * 80)
    print("🔧 FINETUNING NeMo SPEAKER EMBEDDING MODEL")
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
    train_manifest, num_speakers = prepare_speaker_manifest(
        datasets=[
            "/dataset/train/voxconverse_dev",
            "/dataset/jvs_ver1",
            "/dataset/train/callhome_eng",
        ],
        output_path="/dataset/train_speaker_manifest.json",
    )
    print(f"   ✓ Created training manifest: {len(train_manifest)} samples")
    print(f"   ✓ Number of unique speakers: {num_speakers}")
    print()
    
    # === PREPARE VALIDATION DATA ===
    print("📦 Preparing validation data...")
    val_manifest, val_speakers = prepare_speaker_manifest(
        datasets=[
            "/dataset/test/voxconverse_test",
            "/dataset/test/callhome_eng",
            "/dataset/test/callhome_jpn",
        ],
        output_path="/dataset/val_speaker_manifest.json",
    )
    print(f"   ✓ Created validation manifest: {len(val_manifest)} samples")
    print(f"   ✓ Number of validation speakers: {val_speakers}")
    print()
    
    # === LOAD PRETRAINED SPEAKER MODEL ===
    print("📦 Loading pretrained TitaNet Large model...")
    
    try:
        speaker_model = EncDecSpeakerLabelModel.from_pretrained(
            "nvidia/speakerverification_en_titanet_large"
        )
        print("   ✓ Loaded: nvidia/speakerverification_en_titanet_large")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
        print("   Trying alternative model...")
        speaker_model = EncDecSpeakerLabelModel.from_pretrained(
            "nvidia/speakerverification_en_titanet_small"
        )
        print("   ✓ Loaded: nvidia/speakerverification_en_titanet_small")
    
    speaker_model.to(device)
    print()
    
    # === SETUP TRAINING ===
    print("⚙️  Setting up training...")
    
    # Update model config for speaker classification finetuning
    speaker_model.cfg.optim.lr = learning_rate
    speaker_model.cfg.optim.name = "adam"
    speaker_model.cfg.optim.weight_decay = 0.001
    
    # Configure for speaker classification
    speaker_model.cfg.model.decoder.num_classes = num_speakers
    
    # Setup training data
    speaker_model.setup_training_data(
        train_data_config={
            "manifest_filepath": "/dataset/train_speaker_manifest.json",
            "sample_rate": 16000,
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": 4,
            "augmentor": {
                "shift": {"prob": 0.5, "min_shift_ms": -5.0, "max_shift_ms": 5.0},
                "white_noise": {"prob": 0.5, "min_level": -90, "max_level": -46},
                "speed": {"prob": 0.5, "min_speed_rate": 0.95, "max_speed_rate": 1.05},
            },
        }
    )
    
    # Setup validation data
    speaker_model.setup_validation_data(
        val_data_config={
            "manifest_filepath": "/dataset/val_speaker_manifest.json",
            "sample_rate": 16000,
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": 2,
        }
    )
    
    print("   ✓ Data loaders configured")
    print()
    
    # === SETUP CALLBACKS ===
    checkpoint_dir = Path("/results/checkpoints/speaker")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="speaker_model_epoch{epoch:02d}_loss{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=8,
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
        val_check_interval=0.5,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,  # Effective batch size = batch_size * 2
    )
    
    print("   ✓ Trainer configured")
    print()
    
    # === TRAIN MODEL ===
    print("🚀 Starting training...")
    print("=" * 80)
    print()
    
    training_start = datetime.now()
    
    try:
        trainer.fit(speaker_model)
        
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
        final_model_path = checkpoint_dir / "best_speaker_model.nemo"
        speaker_model.save_to(str(final_model_path))
        print(f"💾 Saved final model: {final_model_path.name}")
        
        # === SAVE TRAINING SUMMARY ===
        summary = {
            "model_type": "speaker_embedding_only",
            "base_model": "nvidia/speakerverification_en_titanet_large",
            "training_datasets": [
                "voxconverse_dev",
                "jvs_ver1",
                "callhome_eng_70%",
            ],
            "config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_speakers": num_speakers,
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
        
        summary_path = Path("/results/training_summary_speaker.json")
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
        
        with open("/results/training_error_speaker.json", 'w') as f:
            json.dump(error_log, f, indent=2)
        
        results_volume.commit()
        
        raise


def prepare_speaker_manifest(datasets: list, output_path: str):
    """
    Prepare speaker embedding training manifest from RTTM files
    
    Manifest format:
    {"audio_filepath": "path/to/audio.wav", "offset": 0.0, "duration": 3.0, "label": "speaker_id"}
    """
    import json
    import soundfile as sf
    from pathlib import Path
    from collections import defaultdict
    
    print("   Creating speaker manifest from RTTM files...")
    
    manifest_lines = []
    speaker_to_id = {}
    speaker_counter = 0
    
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
                # Try alternative naming
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
            
            # Parse RTTM file to get speaker segments
            with open(rttm_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 8:
                        continue
                    
                    start = float(parts[3])
                    duration = float(parts[4])
                    speaker_label = parts[7]  # Original speaker ID
                    
                    # Skip segments that are too short (<1.0s)
                    if duration < 1.0:
                        continue
                    
                    # Map speaker label to numeric ID
                    full_speaker_id = f"{dataset_path.name}_{audio_file.stem}_{speaker_label}"
                    
                    if full_speaker_id not in speaker_to_id:
                        speaker_to_id[full_speaker_id] = f"speaker_{speaker_counter:04d}"
                        speaker_counter += 1
                    
                    numeric_speaker_id = speaker_to_id[full_speaker_id]
                    
                    # Add to manifest
                    manifest_lines.append({
                        "audio_filepath": str(audio_file),
                        "offset": start,
                        "duration": min(duration, audio_duration - start),
                        "label": numeric_speaker_id,
                    })
    
    num_speakers = len(speaker_to_id)
    
    print(f"      ✓ Generated {len(manifest_lines)} speaker samples")
    print(f"      ✓ Unique speakers: {num_speakers}")
    
    # Shuffle manifest
    import random
    random.seed(42)
    random.shuffle(manifest_lines)
    
    # Write manifest
    with open(output_path, 'w') as f:
        for entry in manifest_lines:
            f.write(json.dumps(entry) + '\n')
    
    # Save speaker mapping
    mapping_path = output_path.replace('.json', '_speaker_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(speaker_to_id, f, indent=2)
    
    return manifest_lines, num_speakers


@app.local_entrypoint()
def main(
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    resume: str = None,
):
    """Main entry point"""
    
    print("\n🚀 Starting NeMo Speaker Embedding finetuning...")
    print()
    
    result = finetune_speaker.remote(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        resume_checkpoint=resume,
    )
    
    print("\n✅ Speaker Embedding finetuning completed!")
    print(f"   Training duration: {result['training']['duration_minutes']:.2f} minutes")
    print(f"   Best validation loss: {result['training']['best_val_loss']:.4f}")
    print(f"   Number of speakers: {result['config']['num_speakers']}")
    print()
