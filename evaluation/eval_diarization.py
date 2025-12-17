"""
Evaluation script for Speaker Diarization Models
Đánh giá khả năng phân biệt speaker của các models trong folder realtime/:
- realtime_diarization_improved.py (Whisper + SpeechBrain)
- sen_voice.py (SenseVoice)
- senvoi_spebrai_fixed.py (SenseVoice + SpeechBrain)

QUAN TRỌNG:
Đây là đánh giá speaker verification, KHÔNG phải diarization end-to-end.
Kết quả đo khả năng phân biệt embeddings trên trials đơn giản.

LÝ DO KẾT QUẢ CÓ THỂ CAO BẤT THƯỜNG (EER ~0.3%):
1. Trials dễ: positive pairs có thể từ cùng file/phiên → cosine gần 1
2. Negative pairs dễ: speakers rất khác nhau hoặc khác domain
3. Dataset JVS sạch, ít nhiễu → embeddings phân tách tốt
4. Không có overlap/noise trong test conditions

ĐỂ CÓ ĐÁNH GIÁ THỰC TẾ HƠN:
- Cần tạo trials khó hơn (khác file, cách xa nhau, không overlap)
- Đánh giá DER end-to-end trên audio thực với collar 0.25s
- Test trên data có nhiễu, overlap speakers

Metrics:
- EER (Equal Error Rate): FAR = FRR
- FAR (False Acceptance Rate): nhận nhầm người khác
- FRR (False Rejection Rate): từ chối người đúng
- AUC (Area Under Curve): diện tích dưới ROC
- Precision, Recall, F1-score

Dataset: JVS Corpus (Japanese audio) - speaker verification trials
"""

import os
import sys
import json
import pickle
import argparse
import hashlib
from pathlib import Path
from datetime import datetime
from itertools import combinations
from glob import glob
import numpy as np
from tqdm import tqdm
import random
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, precision_recall_fscore_support

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix huggingface_hub compatibility
try:
    import huggingface_hub
    _original_hf_download = huggingface_hub.hf_hub_download
    
    def _patched_hf_download(*args, use_auth_token=None, token=None, **kwargs):
        if token is None and use_auth_token is not None:
            token = use_auth_token
        return _original_hf_download(*args, token=token, **kwargs)
    
    huggingface_hub.hf_hub_download = _patched_hf_download
    print("✓ Applied huggingface_hub compatibility patch")
except Exception as e:
    print(f"Warning: Could not patch huggingface_hub: {e}")

# Configuration
RESULTS_DIR = Path(__file__).parent / "eval_results"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path(__file__).parent / "eval_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Set random seeds
random.seed(123)
np.random.seed(123)

# Models configuration
# NOTE: whisper, sensevoice, sensevoice-speechbrain all use SpeechBrain ECAPA-TDNN
# for speaker embeddings, so they will have IDENTICAL results. Only NeMo uses
# a different speaker embedding model (TitaNet Large).
MODELS = {
    "speechbrain": {
        "name": "SpeechBrain ECAPA-TDNN",
        "script": "realtime_diarization_improved.py",
        "note": "Used by Whisper, SenseVoice, SenseVoice+SpeechBrain"
    },
    "nemo": {
        "name": "NeMo TitaNet Large",
        "script": "main_nemo.py",
        "note": "Different architecture from SpeechBrain"
    },
    "pyannote": {
        "name": "PyAnnote WeSpeaker-ResNet34",
        "script": "main_pyannote.py",
        "note": "PyAnnote speaker-diarization-3.1 embedding model"
    }
}


# ============================================
#   DATASET FUNCTIONS
# ============================================

def list_speakers_and_utts(dataset_path, max_speakers=None):
    """
    List all speakers and their audio files from JVS dataset.
    Scans multiple subdirectories: falset10, nonpara30, parallel100, whisper10
    
    Args:
        dataset_path: Path to dataset root directory
        max_speakers: Maximum number of speakers to use (e.g., 50 for jvs001-jvs050)
    
    Returns:
        dict: {speaker_id: [list of audio file paths]}
    """
    dataset_path = Path(dataset_path)
    spk2utts = {}
    
    print(f"Scanning dataset at: {dataset_path}")
    if max_speakers:
        print(f"Limiting to first {max_speakers} speakers")
    print(f"Dataset exists: {dataset_path.exists()}")
    
    if not dataset_path.exists():
        print(f"ERROR: Dataset path does not exist!")
        return spk2utts
    
    # List all items in dataset directory
    try:
        all_items = list(dataset_path.iterdir())
        print(f"Found {len(all_items)} items in dataset directory")
        
        # Show first few items for debugging
        if all_items:
            print(f"Sample items: {[item.name for item in all_items[:5]]}")
    except Exception as e:
        print(f"Error listing dataset directory: {e}")
        return spk2utts
    
    speaker_count = 0
    for spk in sorted(os.listdir(dataset_path)):
        spk_dir = dataset_path / spk
        if not spk_dir.is_dir():
            continue
        
        # Skip if not a speaker directory (should start with 'jvs')
        if not spk.startswith('jvs'):
            continue
        
        # Check max_speakers limit
        if max_speakers and speaker_count >= max_speakers:
            break
        
        audio_files = []
        # Scan 4 subdirectories as in correct evaluation
        for sub in ["falset10", "nonpara30", "parallel100", "whisper10"]:
            wav_dir = spk_dir / sub / "wav24kHz16bit"
            if wav_dir.exists():
                wavs = list(wav_dir.glob("*.wav"))
                audio_files.extend(wavs)
                if len(wavs) > 0:
                    print(f"  Speaker {spk}/{sub}: Found {len(wavs)} wav files")
        
        if len(audio_files) >= 2:  # Need at least 2 files per speaker
            spk2utts[spk] = sorted(audio_files)
            print(f"  ✓ Speaker {spk}: Total {len(audio_files)} files")
            speaker_count += 1
    
    print(f"\nFound {len(spk2utts)} speakers with >= 2 utterances")
    if len(spk2utts) == 0:
        print("\nWARNING: No valid speakers found!")
        print("Please check:")
        print("  1. Dataset path is correct")
        print("  2. Directory structure: dataset/jvsXXX/parallel100/wav24kHz16bit/*.wav")
        print("  3. Audio files exist in the subdirectories")
    return spk2utts


def build_trials(spk2utts, max_genuine_per_spk=50, impostor_per_spk=100):
    """
    Build speaker verification trials (genuine and impostor pairs).
    
    Args:
        spk2utts: dict of {speaker_id: [audio_files]}
        max_genuine_per_spk: Max genuine pairs per speaker
        impostor_per_spk: Max impostor pairs per speaker
    
    Returns:
        list of trials: [(path1, path2, label), ...]
        label=1 for genuine (same speaker), label=0 for impostor (different speakers)
    """
    trials = []
    speakers = sorted(spk2utts.keys())
    
    # Genuine: all pairs from same speaker (sample if too many)
    for spk in speakers:
        utts = spk2utts[spk]
        pairs = list(combinations(utts, 2))
        random.shuffle(pairs)
        for p in pairs[:max_genuine_per_spk]:
            trials.append((str(p[0]), str(p[1]), 1))
    
    # Impostor: random pairs between different speakers
    for spk in speakers:
        others = [s for s in speakers if s != spk]
        utts_a = spk2utts[spk]
        for _ in range(impostor_per_spk):
            ua = random.choice(utts_a)
            spk_b = random.choice(others)
            ub = random.choice(spk2utts[spk_b])
            trials.append((str(ua), str(ub), 0))
    
    random.shuffle(trials)
    return trials


# ============================================
#   CACHE MANAGEMENT
# ============================================

def get_cache_key(trials):
    """Create unique key from trials list to identify cache."""
    all_files = sorted(set([p for t in trials for p in (t[0], t[1])]))
    files_str = '|'.join(all_files)
    return hashlib.md5(files_str.encode()).hexdigest()


def save_embedding_cache(emb_cache, cache_file):
    """Save embedding cache to file."""
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(emb_cache, f)
        print(f"Saved embedding cache to: {cache_file}")
        return True
    except Exception as e:
        print(f"Error saving cache: {e}")
        return False


def load_embedding_cache(cache_file):
    """Load embedding cache from file."""
    try:
        if not os.path.exists(cache_file):
            return None
        with open(cache_file, 'rb') as f:
            emb_cache = pickle.load(f)
        print(f"Loaded embedding cache from: {cache_file} ({len(emb_cache)} files)")
        return emb_cache
    except Exception as e:
        print(f"Error loading cache: {e}")
        return None


def clear_cache(cache_dir="eval_cache"):
    """Clear all cache files in cache directory."""
    try:
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            print(f"Cleared all cache in: {cache_dir}")
            return True
        else:
            print(f"Cache directory does not exist: {cache_dir}")
            return False
    except Exception as e:
        print(f"Error clearing cache: {e}")
        return False


# ============================================
#   EMBEDDING EXTRACTION (REAL)
# ============================================

def load_speechbrain_model():
    """Load SpeechBrain ECAPA-TDNN speaker recognition model"""
    try:
        print("Loading SpeechBrain ECAPA-TDNN model...")
        from speechbrain.pretrained import EncoderClassifier
        
        # Try local model first
        local_model_path = Path(__file__).parent.parent / "pretrained_models" / "spkrec-ecapa-voxceleb"
        
        if local_model_path.exists() and (local_model_path / "hyperparams.yaml").exists():
            print(f"  Loading from local: {local_model_path}")
            classifier = EncoderClassifier.from_hparams(
                source=str(local_model_path),
                savedir=str(local_model_path),
                run_opts={"device": "cpu"}
            )
        else:
            print("  Downloading from HuggingFace...")
            classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(local_model_path),
                run_opts={"device": "cpu"}
            )
        
        print("✓ SpeechBrain model loaded successfully")
        return classifier
    except Exception as e:
        print(f"✗ Error loading SpeechBrain model: {e}")
        return None


def extract_speechbrain_embedding(audio_path, classifier):
    """Extract speaker embedding using SpeechBrain ECAPA-TDNN"""
    try:
        import torchaudio
        
        # Load audio
        signal, fs = torchaudio.load(str(audio_path))
        
        # Resample to 16kHz if needed
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(fs, 16000)
            signal = resampler(signal)
        
        # Convert to mono if stereo
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)
        
        # Extract embedding
        with torch.no_grad():
            embedding = classifier.encode_batch(signal)
            embedding = embedding.squeeze().cpu().numpy()
        
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding
    except Exception as e:
        print(f"Error extracting embedding from {audio_path}: {e}")
        return None


def load_nemo_model():
    """Load NeMo TitaNet Large speaker recognition model"""
    try:
        print("Loading NeMo TitaNet Large model...")
        from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
        
        # Load TitaNet Large model
        speaker_model = EncDecSpeakerLabelModel.from_pretrained(
            model_name="titanet_large"
        )
        speaker_model.freeze()
        speaker_model.eval()
        
        # Move to CPU for evaluation
        device = "cpu"
        speaker_model.to(device)
        
        print("✓ NeMo model loaded successfully")
        return speaker_model
    except Exception as e:
        print(f"✗ Error loading NeMo model: {e}")
        return None


def load_pyannote_model():
    """Load PyAnnote WeSpeaker-ResNet34 speaker embedding model"""
    try:
        print("Loading PyAnnote WeSpeaker-ResNet34 model...")
        from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
        
        # Load embedding model
        device = torch.device("cpu")
        embedding_model = PretrainedSpeakerEmbedding(
            "pyannote/wespeaker-voxceleb-resnet34-LM",
            device=device
        )
        
        print("✓ PyAnnote model loaded successfully")
        return embedding_model
    except Exception as e:
        print(f"✗ Error loading PyAnnote model: {e}")
        return None


def extract_nemo_embedding(audio_path, speaker_model):
    """Extract speaker embedding using NeMo TitaNet Large"""
    try:
        import soundfile as sf
        
        # Load audio
        audio, sr = sf.read(str(audio_path))
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Resample to 16kHz if needed
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        # Prepare input
        audio_length = len(audio)
        device = next(speaker_model.parameters()).device
        audio_signal = torch.tensor(audio, device=device, dtype=torch.float32).unsqueeze(0)
        audio_signal_len = torch.tensor([audio_length], device=device)
        
        # Extract embedding
        with torch.no_grad():
            _, emb = speaker_model.forward(audio_signal, audio_signal_len)
            # emb shape: (batch, time, embedding_dim) -> squeeze to (embedding_dim,)
            emb = emb.squeeze(0).detach().cpu().numpy()
        
        # Normalize
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        
        return emb
    except Exception as e:
        print(f"Error extracting NeMo embedding from {audio_path}: {e}")
        return None


def extract_pyannote_embedding(audio_path, embedding_model):
    """Extract speaker embedding using PyAnnote WeSpeaker-ResNet34"""
    try:
        import soundfile as sf
        
        # Load audio
        audio, sr = sf.read(str(audio_path))
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Resample to 16kHz if needed (PyAnnote expects 16kHz)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Convert to torch tensor: (batch, channel, samples)
        # PyAnnote expects 3D tensor with channel dimension
        waveform = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
        
        # Extract embedding - pass waveform directly
        # PyAnnote PretrainedSpeakerEmbedding handles device internally
        with torch.no_grad():
            embedding = embedding_model(waveform)
            # embedding is already on CPU
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.squeeze().cpu().numpy()
            else:
                embedding = np.array(embedding).squeeze()
        
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding
    except Exception as e:
        print(f"Error extracting PyAnnote embedding from {audio_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_all_embeddings(trials, cache_dir="eval_cache", use_cache=True):
    """
    Extract embeddings for all files in trials using SpeechBrain, NeMo, and PyAnnote.
    Cache results to avoid re-extraction.
    
    Returns:
        dict: {file_path: {'speechbrain': emb, 'nemo': emb, 'pyannote': emb}}
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Get unique files
    all_files = sorted(set([t[0] for t in trials] + [t[1] for t in trials]))
    print(f"\nTotal unique audio files: {len(all_files)}")
    
    # Create cache key
    cache_key = get_cache_key(trials)
    cache_file = os.path.join(cache_dir, f"embeddings_cache_{cache_key}.pkl")
    
    # Try to load from cache
    if use_cache:
        emb_cache = load_embedding_cache(cache_file)
        if emb_cache is not None:
            # Migrate old cache format to new format
            # Old: {file: {'whisper': emb, 'sensevoice': emb, ...}}
            # New: {file: {'speechbrain': emb, 'nemo': emb}}
            migrated = False
            for file_path in list(emb_cache.keys()):
                if isinstance(emb_cache[file_path], dict):
                    # Check if old format (has whisper/sensevoice keys)
                    if 'whisper' in emb_cache[file_path] or 'sensevoice' in emb_cache[file_path]:
                        # Migrate: use whisper/sensevoice/sensevoice-speechbrain embedding as speechbrain
                        if 'speechbrain' not in emb_cache[file_path]:
                            # Try to find any SpeechBrain embedding from old keys
                            for old_key in ['whisper', 'sensevoice', 'sensevoice-speechbrain']:
                                if old_key in emb_cache[file_path] and emb_cache[file_path][old_key] is not None:
                                    emb_cache[file_path]['speechbrain'] = emb_cache[file_path][old_key]
                                    migrated = True
                                    break
            
            if migrated:
                print("✓ Migrated old cache format to new format (whisper/sensevoice → speechbrain)")
                # Save migrated cache
                save_embedding_cache(emb_cache, cache_file)
            
            # Check if all files are in cache with all embedding types
            missing_files = [f for f in all_files if f not in emb_cache or 
                           'speechbrain' not in emb_cache[f] or 
                           'nemo' not in emb_cache[f] or
                           'pyannote' not in emb_cache[f]]
            if len(missing_files) == 0:
                print("✓ All embeddings found in cache!")
                return emb_cache
            else:
                print(f"Cache incomplete: {len(missing_files)} files need embedding extraction")
                print(f"⚠️  Tip: Run with --clear_cache to start fresh and extract all embeddings")
        else:
            emb_cache = {}
    else:
        emb_cache = {}
    
    # Load models
    print("\nLoading embedding models...")
    print("Note: This will extract embeddings for files not in cache")
    print("      To force re-extraction, use --clear_cache flag\n")
    
    speechbrain_model = load_speechbrain_model()
    nemo_model = load_nemo_model()
    pyannote_model = load_pyannote_model()
    
    if speechbrain_model is None:
        print("ERROR: Failed to load SpeechBrain model")
        return {}
    
    if nemo_model is None:
        print("WARNING: Failed to load NeMo model, will only extract SpeechBrain and PyAnnote embeddings")
    
    if pyannote_model is None:
        print("WARNING: Failed to load PyAnnote model, will only extract SpeechBrain and NeMo embeddings")
    
    # Extract embeddings
    print(f"\nExtracting embeddings for {len(all_files)} files...")
    for file_path in tqdm(all_files, desc="Extracting embeddings"):
        if not os.path.exists(file_path):
            print(f"\nWarning: File not found: {file_path}")
            continue
        
        # Initialize cache entry
        if file_path not in emb_cache:
            emb_cache[file_path] = {}
        
        # Extract SpeechBrain embedding if not cached
        if 'speechbrain' not in emb_cache[file_path]:
            sb_emb = extract_speechbrain_embedding(file_path, speechbrain_model)
            if sb_emb is not None:
                emb_cache[file_path]['speechbrain'] = sb_emb
        
        # Extract NeMo embedding if not cached
        if nemo_model is not None and 'nemo' not in emb_cache[file_path]:
            nemo_emb = extract_nemo_embedding(file_path, nemo_model)
            if nemo_emb is not None:
                emb_cache[file_path]['nemo'] = nemo_emb
        
        # Extract PyAnnote embedding if not cached
        if pyannote_model is not None and 'pyannote' not in emb_cache[file_path]:
            pyannote_emb = extract_pyannote_embedding(file_path, pyannote_model)
            if pyannote_emb is not None:
                emb_cache[file_path]['pyannote'] = pyannote_emb
    
    # Save cache
    save_embedding_cache(emb_cache, cache_file)
    
    return emb_cache


# ============================================
#   SCORING AND METRICS
# ============================================

def cosine_similarity(emb1, emb2):
    """Compute cosine similarity between two embeddings"""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)


def compute_scores_from_cache(trials, emb_cache, embedding_type):
    """
    Compute similarity scores for all trials using cached embeddings.
    
    Returns:
        scores: numpy array of similarity scores
        labels: numpy array of ground truth labels (1=genuine, 0=impostor)
    """
    scores = []
    labels = []
    skipped_trials = 0
    
    for p1, p2, label in trials:
        # Check if files are in cache
        if p1 not in emb_cache or p2 not in emb_cache:
            skipped_trials += 1
            continue
        
        # Check if embedding type exists
        if (embedding_type not in emb_cache[p1] or 
            embedding_type not in emb_cache[p2] or
            emb_cache[p1][embedding_type] is None or 
            emb_cache[p2][embedding_type] is None):
            skipped_trials += 1
            continue
        
        try:
            emb1 = emb_cache[p1][embedding_type]
            emb2 = emb_cache[p2][embedding_type]
            
            # Check for invalid embeddings
            if (np.all(emb1 == 0) or np.all(np.isnan(emb1)) or
                np.all(emb2 == 0) or np.all(np.isnan(emb2))):
                skipped_trials += 1
                continue
            
            # Compute cosine similarity
            score = cosine_similarity(emb1, emb2)
            
            # Check for NaN or inf
            if np.isnan(score) or np.isinf(score):
                skipped_trials += 1
                continue
            
            scores.append(score)
            labels.append(label)
            
        except Exception as e:
            print(f"\nError computing similarity: {e}")
            skipped_trials += 1
            continue
    
    if skipped_trials > 0:
        print(f"Skipped {skipped_trials}/{len(trials)} trials due to missing/invalid embeddings")
    
    return np.array(scores), np.array(labels)


def compute_metrics_at_threshold(scores, labels, threshold):
    """Compute precision, recall, F1 at a specific threshold."""
    predictions = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    return float(precision), float(recall), float(f1)


def compute_far_frr_eer(scores, labels):
    """
    Compute FAR, FRR, EER and other metrics.
    
    Args:
        scores: Similarity scores
        labels: Ground truth labels (1=genuine, 0=impostor)
    
    Returns:
        dict with metrics
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr  # FRR = 1 - TPR
    far = fpr      # FAR = FPR
    frr = fnr      # FRR = FNR
    
    # Find EER (Equal Error Rate) where FAR = FRR
    idx_eer = np.nanargmin(np.abs(far - frr))
    eer = (far[idx_eer] + frr[idx_eer]) / 2.0
    thr_eer = thresholds[idx_eer]
    
    # Precision, Recall, F1 at EER threshold
    precision_at_eer, recall_at_eer, f1_at_eer = compute_metrics_at_threshold(
        scores, labels, thr_eer
    )
    
    # Find best F1 score
    best_f1 = 0.0
    best_f1_threshold = thr_eer
    best_f1_precision = precision_at_eer
    best_f1_recall = recall_at_eer
    
    for thr in thresholds:
        precision, recall, f1 = compute_metrics_at_threshold(scores, labels, thr)
        if f1 > best_f1:
            best_f1 = f1
            best_f1_threshold = thr
            best_f1_precision = precision
            best_f1_recall = recall
    
    # Compute AUC
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(labels, scores)
    
    return {
        "EER": float(eer),
        "FAR_at_EER": float(far[idx_eer]),
        "FRR_at_EER": float(frr[idx_eer]),
        "threshold_at_EER": float(thr_eer),
        "precision_at_EER": float(precision_at_eer),
        "recall_at_EER": float(recall_at_eer),
        "F1_at_EER": float(f1_at_eer),
        "best_F1": float(best_f1),
        "threshold_at_best_F1": float(best_f1_threshold),
        "precision_at_best_F1": float(best_f1_precision),
        "recall_at_best_F1": float(best_f1_recall),
        "AUC": float(roc_auc),
        "FAR_curve": far,
        "FRR_curve": frr,
        "thresholds": thresholds,
        "fpr": fpr,
        "tpr": tpr,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve
    }


# ============================================
#   EVALUATION FUNCTIONS
# ============================================

def save_evaluation_results(model_name, metrics, trials_info, output_dir="eval_results"):
    """Save evaluation results to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "evaluation_type": "speaker_verification",
        "dataset": "JVS_dataset",
        "trials_info": trials_info,
        "metrics": {
            "EER": float(metrics['EER']),
            "EER_percent": float(metrics['EER'] * 100),
            "FAR_at_EER": float(metrics['FAR_at_EER']),
            "FRR_at_EER": float(metrics['FRR_at_EER']),
            "threshold_at_EER": float(metrics['threshold_at_EER']),
            "precision_at_EER": float(metrics['precision_at_EER']),
            "recall_at_EER": float(metrics['recall_at_EER']),
            "F1_at_EER": float(metrics['F1_at_EER']),
            "best_F1": float(metrics['best_F1']),
            "threshold_at_best_F1": float(metrics['threshold_at_best_F1']),
            "precision_at_best_F1": float(metrics['precision_at_best_F1']),
            "recall_at_best_F1": float(metrics['recall_at_best_F1']),
            "AUC": float(metrics['AUC'])
        },
        "performance_classification": (
            "Excellent" if metrics['EER'] < 0.05 else
            "Good" if metrics['EER'] < 0.10 else
            "Fair" if metrics['EER'] < 0.20 else
            "Poor"
        )
    }
    
    result_file = os.path.join(output_dir, f"eval_diarization_{model_name}_results.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return result_file


def evaluate_embedding_type(embedding_type, trials, emb_cache, trials_info, output_dir="eval_results"):
    """
    Evaluate a specific embedding type (speechbrain/nemo)
    
    - speechbrain: SpeechBrain ECAPA-TDNN (used by Whisper, SenseVoice models)
    - nemo: NeMo TitaNet Large embeddings
    """
    print(f"\n=== Evaluating {embedding_type} embeddings ===")
    
    # Validate embedding type
    if embedding_type not in ['speechbrain', 'nemo', 'pyannote']:
        print(f"ERROR: Unknown embedding type: {embedding_type}")
        print("Valid types: speechbrain, nemo, pyannote")
        return None
    
    scores, labels = compute_scores_from_cache(trials, emb_cache, embedding_type)
    
    if len(scores) == 0:
        print(f"Error: No valid trials for {embedding_type} embeddings!")
        return None
    
    print(f"Computing metrics on {len(scores)} valid trials")
    metrics = compute_far_frr_eer(scores, labels)
    
    # Print results
    print(f"EER: {metrics['EER']*100:.2f}% | FAR@EER: {metrics['FAR_at_EER']*100:.2f}% | "
          f"FRR@EER: {metrics['FRR_at_EER']*100:.2f}% | Thr(EER): {metrics['threshold_at_EER']:.4f}")
    print(f"Precision@EER: {metrics['precision_at_EER']*100:.2f}% | "
          f"Recall@EER: {metrics['recall_at_EER']*100:.2f}% | "
          f"F1@EER: {metrics['F1_at_EER']*100:.2f}%")
    print(f"Best F1: {metrics['best_F1']*100:.2f}% | "
          f"Precision@F1: {metrics['precision_at_best_F1']*100:.2f}% | "
          f"Recall@F1: {metrics['recall_at_best_F1']*100:.2f}% | "
          f"Thr(F1): {metrics['threshold_at_best_F1']:.4f}")
    print(f"AUC: {metrics['AUC']:.4f}")
    
    # Save results to JSON
    save_evaluation_results(embedding_type, metrics, trials_info, output_dir)
    
    return metrics, (scores, labels)


def plot_roc_curves(results, output_dir="eval_results"):
    """Plot and save ROC curves"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    colors = {
        "speechbrain": "blue",
        "nemo": "red",
        "pyannote": "green"
    }
    
    for emb_type in ["speechbrain", "nemo", "pyannote"]:
        if emb_type not in results or results[emb_type] is None:
            continue
            
        metrics = results[emb_type]
        fpr = metrics["fpr"]
        tpr = metrics["tpr"]
        roc_auc = metrics["AUC"]
        eer = metrics["EER"]
        
        plt.plot(fpr * 100, tpr * 100, color=colors[emb_type], lw=2,
                label=f'{MODELS[emb_type]["name"]} (AUC={roc_auc:.4f}, EER={eer*100:.2f}%)')
    
    plt.plot([0, 100], [0, 100], 'k--', lw=1, label='Random (AUC=0.5)')
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.xlabel('False Acceptance Rate (%)', fontsize=12)
    plt.ylabel('True Positive Rate (%)', fontsize=12)
    plt.title('ROC Curves - Speaker Verification', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    roc_path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to: {roc_path}")
    plt.close()


def plot_det_curves(results, output_dir="eval_results"):
    """Plot and save DET curves"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    colors = {
        "speechbrain": "blue",
        "nemo": "red",
        "pyannote": "green"
    }
    
    for emb_type in ["speechbrain", "nemo", "pyannote"]:
        if emb_type not in results or results[emb_type] is None:
            continue
            
        metrics = results[emb_type]
        far = metrics["FAR_curve"]
        frr = metrics["FRR_curve"]
        eer = metrics["EER"]
        
        plt.plot(far * 100, frr * 100, color=colors[emb_type], lw=2,
                label=f'{MODELS[emb_type]["name"]} (EER={eer*100:.2f}%)')
        
        # Mark EER point
        idx_eer = np.nanargmin(np.abs(frr - far))
        plt.plot(far[idx_eer] * 100, frr[idx_eer] * 100, 'o', 
                color=colors[emb_type], markersize=8)
    
    plt.plot([0, 50], [0, 50], 'k--', lw=1, label='EER line (FAR=FRR)')
    plt.xlim([0, 50])
    plt.ylim([0, 50])
    plt.xlabel('False Acceptance Rate (%)', fontsize=12)
    plt.ylabel('False Rejection Rate (%)', fontsize=12)
    plt.title('DET Curves - Speaker Verification', fontsize=14, fontweight='bold')
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    det_path = os.path.join(output_dir, "det_curves.png")
    plt.savefig(det_path, dpi=300, bbox_inches='tight')
    print(f"DET curve saved to: {det_path}")
    plt.close()


def plot_precision_recall_curves(results, output_dir="eval_results"):
    """Plot and save Precision-Recall curves"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    colors = {
        "speechbrain": "blue",
        "nemo": "red",
        "pyannote": "green"
    }
    
    for emb_type in ["speechbrain", "nemo", "pyannote"]:
        if emb_type not in results or results[emb_type] is None:
            continue
            
        metrics = results[emb_type]
        precision = metrics["precision_curve"]
        recall = metrics["recall_curve"]
        best_f1 = metrics["best_F1"]
        
        plt.plot(recall * 100, precision * 100, color=colors[emb_type], lw=2,
                label=f'{MODELS[emb_type]["name"]} (Best F1={best_f1*100:.2f}%)')
    
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.xlabel('Recall (%)', fontsize=12)
    plt.ylabel('Precision (%)', fontsize=12)
    plt.title('Precision-Recall Curves - Speaker Verification', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    pr_path = os.path.join(output_dir, "precision_recall_curves.png")
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    print(f"Precision-Recall curve saved to: {pr_path}")
    plt.close()


def evaluate_dataset(dataset_path, output_dir="eval_results", use_cache=True, 
                     max_genuine_per_spk=50, impostor_per_spk=100, max_speakers=None):
    """
    Evaluate all models using speaker verification approach.
    Main entry point for evaluation.
    
    Args:
        dataset_path: Path to dataset root directory
        output_dir: Output directory for results
        use_cache: Whether to use cached embeddings
        max_genuine_per_spk: Max genuine trials per speaker
        impostor_per_spk: Max impostor trials per speaker
        max_speakers: Maximum number of speakers to use (e.g., 50 for jvs001-jvs050)
    """
    # List speakers and build trials
    spk2utts = list_speakers_and_utts(dataset_path, max_speakers=max_speakers)
    print(f"Found {len(spk2utts)} speakers usable.")
    
    trials = build_trials(spk2utts, max_genuine_per_spk, impostor_per_spk)
    print(f"Total trials: {len(trials)}")
    
    trials_info = {
        "num_speakers": len(spk2utts),
        "total_trials": len(trials),
        "max_genuine_per_speaker": max_genuine_per_spk,
        "max_impostor_per_speaker": impostor_per_spk,
        "genuine_trials": sum(1 for _, _, label in trials if label == 1),
        "impostor_trials": sum(1 for _, _, label in trials if label == 0),
        "dataset_path": str(dataset_path)
    }
    
    # Extract embeddings (or load from cache)
    emb_cache = extract_all_embeddings(trials, cache_dir=str(CACHE_DIR), use_cache=use_cache)
    
    if len(emb_cache) == 0:
        print("ERROR: No embeddings extracted. Cannot proceed.")
        return None, trials_info
    
    # Evaluate each model
    results = {}
    scores_data = {}
    
    print("\n" + "="*70)
    print("Evaluating 3 speaker embedding models:")
    print("  1. SpeechBrain ECAPA-TDNN (used by Whisper/SenseVoice models)")
    print("  2. NeMo TitaNet Large (different architecture)")
    print("  3. PyAnnote WeSpeaker-ResNet34 (speaker-diarization-3.1)")
    print("="*70 + "\n")
    
    for emb_type in ["speechbrain", "nemo", "pyannote"]:
        result = evaluate_embedding_type(emb_type, trials, emb_cache, trials_info, output_dir)
        
        if result is None:
            results[emb_type] = None
            continue
        
        metrics, (scores, labels) = result
        results[emb_type] = metrics
        scores_data[emb_type] = (scores, labels)
    
    # Plot curves
    print("\n=== Plotting curves ===")
    plot_roc_curves(results, output_dir)
    plot_det_curves(results, output_dir)
    plot_precision_recall_curves(results, output_dir)
    
    # Write summary to log file
    log_file = os.path.join(output_dir, "result.log")
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Speaker Embedding Comparison\n")
        f.write("3 Models: SpeechBrain, NeMo, PyAnnote\n")
        f.write("="*70 + "\n\n")
        
        for emb_type in ["speechbrain", "nemo", "pyannote"]:
            if results[emb_type]:
                m = results[emb_type]
                f.write(f"=== Evaluating {emb_type} embeddings ===\n")
                f.write(f"Computing metrics on {len(scores_data[emb_type][0])} valid trials\n")
                f.write(f"EER: {m['EER']*100:.2f}% | FAR@EER: {m['FAR_at_EER']*100:.2f}% | ")
                f.write(f"FRR@EER: {m['FRR_at_EER']*100:.2f}% | Thr(EER): {m['threshold_at_EER']:.4f}\n")
                f.write(f"Precision@EER: {m['precision_at_EER']*100:.2f}% | ")
                f.write(f"Recall@EER: {m['recall_at_EER']*100:.2f}% | ")
                f.write(f"F1@EER: {m['F1_at_EER']*100:.2f}%\n")
                f.write(f"Best F1: {m['best_F1']*100:.2f}% | ")
                f.write(f"Precision@F1: {m['precision_at_best_F1']*100:.2f}% | ")
                f.write(f"Recall@F1: {m['recall_at_best_F1']*100:.2f}% | ")
                f.write(f"Thr(F1): {m['threshold_at_best_F1']:.4f}\n")
                f.write(f"AUC: {m['AUC']:.4f}\n\n")
    
    print(f"\n✓ Saved evaluation log to: {log_file}")
    
    # Print summary
    print("\n=== Final Results ===")
    for emb_type in ["speechbrain", "nemo", "pyannote"]:
        if emb_type in results and results[emb_type]:
            model_info = MODELS[emb_type]['name']
            if 'note' in MODELS[emb_type]:
                model_info += f" ({MODELS[emb_type]['note']})"
            print(f"{model_info}: EER={results[emb_type]['EER']:.4f}, AUC={results[emb_type]['AUC']:.4f}")
    
    print("\nGenerated Files:")
    print("  eval_results/roc_curves.png - ROC curves comparison")
    print("  eval_results/det_curves.png - DET curves comparison")
    print("  eval_results/precision_recall_curves.png - PR curves comparison")
    print("  eval_results/eval_diarization_*_results.json - Detailed results per model")
    print("  eval_results/result.log - Summary log")
    print("  eval_cache/embeddings_cache_*.pkl - Cached embeddings")
    
    return results, trials_info


# ============================================
#   MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Speaker Diarization Models (Real Embeddings)")
    parser.add_argument("--dataset", type=str, 
                       default="../dataset/jvs_ver1/jvs_ver1",
                       help="Path to JVS dataset root directory")
    parser.add_argument("--output_dir", type=str, 
                       default="eval_results",
                       help="Output directory for results")
    parser.add_argument("--max_genuine_per_spk", type=int, default=50,
                       help="Max genuine trials per speaker")
    parser.add_argument("--impostor_per_spk", type=int, default=100,
                       help="Max impostor trials per speaker")
    parser.add_argument("--no_cache", action="store_true",
                       help="Disable embedding cache (force re-extraction)")
    parser.add_argument("--clear_cache", action="store_true",
                       help="Clear embedding cache before evaluation")
    parser.add_argument("--max_speakers", type=int, default=None,
                       help="Maximum number of speakers to use (e.g., 50 for jvs001-jvs050)")
    
    args = parser.parse_args()
    
    print("="*70)
    print("Speaker Verification Evaluation")
    print("="*70)
    print("Evaluating 3 embedding models:")
    print("  1. SpeechBrain ECAPA-TDNN (used by Whisper/SenseVoice)")
    print("  2. NeMo TitaNet Large")
    print("  3. PyAnnote WeSpeaker-ResNet34")
    if args.max_speakers:
        print(f"\n📊 Limiting to first {args.max_speakers} speakers")
    if args.clear_cache:
        print("\n⚠️  Cache will be cleared and embeddings re-extracted")
    print("="*70)
    
    # Clear cache if requested
    if args.clear_cache:
        clear_cache(str(CACHE_DIR))
    
    # Resolve dataset path - try multiple common locations
    if os.path.isabs(args.dataset):
        dataset_path = Path(args.dataset)
    else:
        dataset_path = Path(__file__).parent / args.dataset
    
    dataset_path = dataset_path.resolve()
    
    print(f"\nLooking for dataset at: {dataset_path}")
    
    if not dataset_path.exists():
        print(f"✗ Dataset not found at: {dataset_path}")
        
        # Try alternative paths
        alt_paths = [
            Path(__file__).parent.parent.parent / "dataset" / "jvs_ver1",
            Path(__file__).parent.parent / "dataset" / "jvs_ver1",
            Path(__file__).parent / "dataset" / "jvs_ver1",
        ]
        
        print("\nTrying alternative paths:")
        for alt_path in alt_paths:
            print(f"  Checking: {alt_path}")
            if alt_path.exists():
                dataset_path = alt_path
                print(f"  ✓ Found dataset at: {dataset_path}")
                break
        else:
            print(f"\nERROR: Dataset not found in any common location!")
            print(f"\nPlease:")
            print(f"  1. Download JVS dataset from: https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus")
            print(f"  2. Extract to one of these locations:")
            for alt_path in alt_paths:
                print(f"     - {alt_path}")
            print(f"  3. Or specify custom path with: --dataset <path>")
            print(f"\nExpected structure: dataset/jvs_ver1/jvs001/parallel100/wav24kHz16bit/*.wav")
            return
    
    print(f"✓ Using dataset at: {dataset_path}")
    
    # Run evaluation
    results, trials_info = evaluate_dataset(
        dataset_path, 
        output_dir=args.output_dir, 
        use_cache=not args.no_cache,
        max_genuine_per_spk=args.max_genuine_per_spk,
        impostor_per_spk=args.impostor_per_spk,
        max_speakers=args.max_speakers
    )
    
    if results is None:
        print("\nEvaluation failed!")
        return
    
    print("\n" + "="*70)
    print("Evaluation completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
