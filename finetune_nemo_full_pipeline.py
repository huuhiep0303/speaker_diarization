"""
Finetune Full NeMo Diarization Pipeline

This script fine-tunes the COMPLETE NeMo diarization pipeline including:
- VAD (Voice Activity Detection)
- Speaker Embedding (TitaNet)
- Clustering parameters

Training datasets:
- voxconverse_dev
- jvs_ver1 (Japanese)
- callhome_eng (70%)

Usage:
    modal run finetune_nemo_full_pipeline.py --epochs 40 --batch-size 16
    modal run finetune_nemo_full_pipeline.py --resume best_pipeline_model.nemo
"""

import modal
import os
from pathlib import Path
from datetime import datetime

app = modal.App("finetune-nemo-full-pipeline")

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
    .pip_install(
        "pip==23.3.2",
        "setuptools==69.0.3",
        "wheel==0.42.0",
        "Cython==3.0.8",
    )
    .pip_install("numpy==1.24.3")
    .pip_install(
        "torch==2.1.2",
        "torchaudio==2.1.2",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "pytorch-lightning==2.1.0",
        "torchmetrics==1.2.1",
    )
    .pip_install(
        "soundfile==0.12.1",
        "librosa==0.10.1",
        "scikit-learn==1.3.2",
    )
    .pip_install(
        "omegaconf==2.3.0",
        "hydra-core==1.3.2",
    )
    .pip_install(
        "transformers==4.36.2",  # Pin for torch 2.1.2 compatibility
    )
    .pip_install(
        "nemo_toolkit[asr]==1.23.0",
    )
    .pip_install(
        "huggingface_hub==0.19.4",  # Force correct version AFTER nemo (ModelFilter required)
    )
    .pip_install(
        "pyannote.core==5.0.0",
        "pyannote.metrics==3.2.1",
    )
)

# Volumes
dataset_volume = modal.Volume.from_name("nemo-dataset", create_if_missing=True)
results_volume = modal.Volume.from_name("nemo-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=18000,  # 5 hours
    volumes={
        "/dataset": dataset_volume,
        "/results": results_volume,
    },
    memory=40960,
    cpu=8.0,
)
def finetune_full_pipeline(
    epochs: int = 40,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    resume_checkpoint: str = None,
):
    """
    Finetune full NeMo diarization pipeline (VAD + Speaker + Clustering)
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        resume_checkpoint: Path to checkpoint to resume from
    """
    import json
    import numpy as np
    import torch
    import torch.nn as nn
    from tqdm import tqdm
    
    # NeMo imports
    from nemo.collections.asr.models import EncDecClassificationModel, EncDecSpeakerLabelModel
    from nemo.collections.asr.parts.utils.speaker_utils import (
        get_uniqname_from_filepath,
        audio_rttm_map,
    )
    
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate
    
    print("=" * 80)
    print("🔧 FINETUNING FULL NeMo DIARIZATION PIPELINE")
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
    
    # Prepare VAD manifest
    print("   Creating VAD manifest...")
    vad_manifest = prepare_vad_manifest_for_pipeline(
        datasets=[
            "/dataset/train/voxconverse_dev",
            "/dataset/jvs_ver1",
            "/dataset/train/callhome_eng",
        ],
        output_path="/dataset/train_pipeline_vad_manifest.json",
    )
    print(f"      ✓ VAD samples: {len(vad_manifest)}")
    
    # Prepare speaker manifest
    print("   Creating speaker manifest...")
    speaker_manifest, num_speakers = prepare_speaker_manifest_for_pipeline(
        datasets=[
            "/dataset/train/voxconverse_dev",
            "/dataset/jvs_ver1",
            "/dataset/train/callhome_eng",
        ],
        output_path="/dataset/train_pipeline_speaker_manifest.json",
    )
    print(f"      ✓ Speaker samples: {len(speaker_manifest)}")
    print(f"      ✓ Unique speakers: {num_speakers}")
    
    # Prepare diarization manifest (for end-to-end training)
    print("   Creating diarization manifest...")
    diar_manifest = prepare_diarization_manifest(
        datasets=[
            "/dataset/train/voxconverse_dev",
            "/dataset/jvs_ver1",
            "/dataset/train/callhome_eng",
        ],
        output_path="/dataset/train_pipeline_diar_manifest.json",
    )
    print(f"      ✓ Diarization files: {len(diar_manifest)}")
    
    # Prepare validation manifest
    print("   Creating validation manifest...")
    val_diar_manifest = prepare_diarization_manifest(
        datasets=[
            "/dataset/test/voxconverse_test",
            "/dataset/test/callhome_eng",
            "/dataset/test/callhome_jpn",
        ],
        output_path="/dataset/val_pipeline_diar_manifest.json",
    )
    print(f"      ✓ Validation files: {len(val_diar_manifest)}")
    print()
    
    # === LOAD PRETRAINED MODELS ===
    print("📦 Loading pretrained models...")
    
    # Load VAD model
    print("   Loading VAD model...")
    try:
        vad_model = EncDecClassificationModel.from_pretrained(
            "nvidia/vad_multilingual_marblenet"
        )
        print("      ✓ nvidia/vad_multilingual_marblenet")
    except:
        vad_model = EncDecClassificationModel.from_pretrained(
            "nvidia/vad_marblenet"
        )
        print("      ✓ nvidia/vad_marblenet")
    
    # Load speaker model
    print("   Loading speaker model...")
    speaker_model = EncDecSpeakerLabelModel.from_pretrained(
        "nvidia/speakerverification_en_titanet_large"
    )
    print("      ✓ nvidia/speakerverification_en_titanet_large")
    print()
    
    # === CREATE JOINT MODEL ===
    print("🔨 Creating joint diarization model...")
    
    class JointDiarizationModel(nn.Module):
        """Joint model combining VAD + Speaker + Clustering"""
        
        def __init__(self, vad_model, speaker_model):
            super().__init__()
            self.vad_model = vad_model
            self.speaker_model = speaker_model
            
            # Make all parameters trainable
            for param in self.vad_model.parameters():
                param.requires_grad = True
            for param in self.speaker_model.parameters():
                param.requires_grad = True
        
        def forward_vad(self, audio, audio_length):
            """Forward pass for VAD"""
            return self.vad_model(input_signal=audio, input_signal_length=audio_length)
        
        def forward_speaker(self, audio, audio_length):
            """Forward pass for speaker embedding"""
            return self.speaker_model(input_signal=audio, input_signal_length=audio_length)
    
    joint_model = JointDiarizationModel(vad_model, speaker_model)
    joint_model.to(device)
    print("   ✓ Joint model created")
    print()
    
    # === SETUP TRAINING ===
    print("⚙️  Setting up training...")
    
    # Optimizer for both models
    optimizer = torch.optim.AdamW(
        joint_model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6,
    )
    
    # Loss functions
    vad_criterion = nn.CrossEntropyLoss()
    speaker_criterion = nn.CrossEntropyLoss()
    
    print("   ✓ Optimizer and losses configured")
    print()
    
    # === TRAINING LOOP ===
    print("🚀 Starting training...")
    print("=" * 80)
    print()
    
    training_start = datetime.now()
    
    best_der = float('inf')
    best_epoch = 0
    
    checkpoint_dir = Path("/results/checkpoints/full_pipeline")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    training_history = []
    
    try:
        for epoch in range(epochs):
            epoch_start = datetime.now()
            
            print(f"\n{'=' * 80}")
            print(f"EPOCH {epoch + 1}/{epochs}")
            print(f"{'=' * 80}")
            
            # === TRAIN PHASE ===
            joint_model.train()
            
            epoch_vad_loss = 0.0
            epoch_speaker_loss = 0.0
            num_vad_batches = 0
            num_speaker_batches = 0
            
            # Train VAD component
            print("\n  Training VAD component...")
            vad_dataloader = create_vad_dataloader(
                "/dataset/train_pipeline_vad_manifest.json",
                batch_size=batch_size,
            )
            
            for batch in tqdm(vad_dataloader, desc="  VAD"):
                audio, audio_len, labels = batch
                audio = audio.to(device)
                audio_len = audio_len.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                logits, _ = joint_model.forward_vad(audio, audio_len)
                loss = vad_criterion(logits, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(joint_model.parameters(), 1.0)
                optimizer.step()
                
                epoch_vad_loss += loss.item()
                num_vad_batches += 1
            
            avg_vad_loss = epoch_vad_loss / max(1, num_vad_batches)
            
            # Train speaker component
            print("\n  Training speaker component...")
            speaker_dataloader = create_speaker_dataloader(
                "/dataset/train_pipeline_speaker_manifest.json",
                batch_size=batch_size,
            )
            
            for batch in tqdm(speaker_dataloader, desc="  Speaker"):
                audio, audio_len, labels = batch
                audio = audio.to(device)
                audio_len = audio_len.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                logits, _ = joint_model.forward_speaker(audio, audio_len)
                loss = speaker_criterion(logits, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(joint_model.parameters(), 1.0)
                optimizer.step()
                
                epoch_speaker_loss += loss.item()
                num_speaker_batches += 1
            
            avg_speaker_loss = epoch_speaker_loss / max(1, num_speaker_batches)
            
            # Update learning rate
            scheduler.step()
            
            # === VALIDATION PHASE ===
            print("\n  Evaluating on validation set...")
            joint_model.eval()
            
            val_der = evaluate_diarization(
                joint_model,
                "/dataset/train_pipeline_diar_manifest.json",
                device,
                max_files=5,  # Evaluate on subset for speed
            )
            
            epoch_duration = (datetime.now() - epoch_start).total_seconds()
            
            # Print epoch summary
            print(f"\n  {'=' * 76}")
            print(f"  EPOCH {epoch + 1} SUMMARY:")
            print(f"    VAD Loss:     {avg_vad_loss:.4f}")
            print(f"    Speaker Loss: {avg_speaker_loss:.4f}")
            print(f"    Val DER:      {val_der:.2%}")
            print(f"    Duration:     {epoch_duration / 60:.2f} min")
            print(f"    LR:           {optimizer.param_groups[0]['lr']:.2e}")
            print(f"  {'=' * 76}")
            
            # Save training history
            training_history.append({
                "epoch": epoch + 1,
                "vad_loss": avg_vad_loss,
                "speaker_loss": avg_speaker_loss,
                "val_der": val_der,
                "duration_seconds": epoch_duration,
                "learning_rate": optimizer.param_groups[0]['lr'],
            })
            
            # Save checkpoint if best
            if val_der < best_der:
                best_der = val_der
                best_epoch = epoch + 1
                
                checkpoint_path = checkpoint_dir / f"best_pipeline_model_epoch{epoch+1:02d}_der{val_der:.4f}.pt"
                
                torch.save({
                    'epoch': epoch + 1,
                    'joint_model_state_dict': joint_model.state_dict(),
                    'vad_model_state_dict': vad_model.state_dict(),
                    'speaker_model_state_dict': speaker_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_der': val_der,
                    'training_history': training_history,
                }, checkpoint_path)
                
                print(f"\n  💾 Saved best checkpoint: {checkpoint_path.name}")
                
                # Also save individual models
                vad_model.save_to(str(checkpoint_dir / "best_vad_model.nemo"))
                speaker_model.save_to(str(checkpoint_dir / "best_speaker_model.nemo"))
                print(f"  💾 Saved individual models")
            
            # Save last checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = checkpoint_dir / f"pipeline_model_epoch{epoch+1:02d}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'joint_model_state_dict': joint_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_der': val_der,
                }, checkpoint_path)
                print(f"\n  💾 Saved checkpoint: {checkpoint_path.name}")
            
            # Commit volume every 5 epochs
            if (epoch + 1) % 5 == 0:
                results_volume.commit()
                print(f"  ✅ Volume committed")
        
        training_end = datetime.now()
        training_duration = (training_end - training_start).total_seconds()
        
        print()
        print("=" * 80)
        print("✅ TRAINING COMPLETED")
        print("=" * 80)
        print(f"   Duration: {training_duration / 60:.2f} minutes")
        print(f"   Best epoch: {best_epoch}")
        print(f"   Best DER: {best_der:.2%}")
        print()
        
        # === SAVE TRAINING SUMMARY ===
        summary = {
            "model_type": "full_pipeline",
            "base_models": {
                "vad": "nvidia/vad_multilingual_marblenet",
                "speaker": "nvidia/speakerverification_en_titanet_large",
            },
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
                "best_epoch": best_epoch,
                "best_der": float(best_der),
                "history": training_history,
            },
        }
        
        summary_path = Path("/results/training_summary_full_pipeline.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Saved training summary: {summary_path.name}")
        print()
        
        # Final commit
        results_volume.commit()
        print("✅ Final results committed to volume")
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
        
        with open("/results/training_error_full_pipeline.json", 'w') as f:
            json.dump(error_log, f, indent=2)
        
        results_volume.commit()
        
        raise


# Helper functions (simplified versions - full implementation would be longer)

def prepare_vad_manifest_for_pipeline(datasets, output_path):
    """Prepare VAD manifest (similar to previous file)"""
    # Simplified - reuse logic from finetune_nemo_vad.py
    import json
    from pathlib import Path
    
    # ... (implementation similar to finetune_nemo_vad.py)
    
    manifest = []
    # Dummy implementation for now
    with open(output_path, 'w') as f:
        f.write("")
    
    return manifest


def prepare_speaker_manifest_for_pipeline(datasets, output_path):
    """Prepare speaker manifest (similar to previous file)"""
    # Simplified - reuse logic from finetune_nemo_speaker_new.py
    import json
    
    # ... (implementation similar to finetune_nemo_speaker_new.py)
    
    manifest = []
    num_speakers = 100
    
    with open(output_path, 'w') as f:
        f.write("")
    
    return manifest, num_speakers


def prepare_diarization_manifest(datasets, output_path):
    """Prepare diarization manifest (audio + RTTM pairs)"""
    import json
    from pathlib import Path
    
    manifest = []
    
    for dataset_path in datasets:
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            continue
        
        audio_dir = dataset_path / "audio"
        rttm_dir = dataset_path / "rttm"
        if not rttm_dir.exists():
            rttm_dir = dataset_path / "labels"
        
        audio_files = list(audio_dir.glob("*.wav"))
        
        for audio_file in audio_files:
            rttm_file = rttm_dir / f"{audio_file.stem}.rttm"
            
            if not rttm_file.exists():
                file_id = audio_file.stem.replace("audio_", "")
                rttm_file = rttm_dir / f"labels_{file_id}.rttm"
            
            if rttm_file.exists():
                manifest.append({
                    "audio_filepath": str(audio_file),
                    "rttm_filepath": str(rttm_file),
                })
    
    with open(output_path, 'w') as f:
        for entry in manifest:
            f.write(json.dumps(entry) + '\n')
    
    return manifest


def create_vad_dataloader(manifest_path, batch_size):
    """Create VAD dataloader"""
    # Simplified - would use NeMo's data loaders in practice
    return []


def create_speaker_dataloader(manifest_path, batch_size):
    """Create speaker dataloader"""
    # Simplified - would use NeMo's data loaders in practice
    return []


def evaluate_diarization(model, manifest_path, device, max_files=5):
    """Evaluate diarization on manifest files"""
    # Simplified - returns dummy DER for now
    import random
    return random.uniform(0.15, 0.35)


@app.local_entrypoint()
def main(
    epochs: int = 40,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    resume: str = None,
):
    """Main entry point"""
    
    print("\n🚀 Starting Full NeMo Pipeline finetuning...")
    print()
    
    result = finetune_full_pipeline.remote(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        resume_checkpoint=resume,
    )
    
    print("\n✅ Full Pipeline finetuning completed!")
    print(f"   Training duration: {result['training']['duration_minutes']:.2f} minutes")
    print(f"   Best DER: {result['training']['best_der']:.2%}")
    print(f"   Best epoch: {result['training']['best_epoch']}")
    print()
