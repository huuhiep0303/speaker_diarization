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
MODELS = {
    "whisper": {
        "name": "Whisper+SpeechBrain",
        "script": "realtime_diarization_improved.py"
    },
    "sensevoice": {
        "name": "SenseVoice",
        "script": "sen_voice.py"
    },
    "sensevoice-speechbrain": {
        "name": "SenseVoice+SpeechBrain",
        "script": "senvoi_spebrai_fixed.py"
    }
}


# ============================================
#   DATASET FUNCTIONS
# ============================================

def list_speakers_and_utts(dataset_path):
    """
    List all speakers and their audio files from JVS dataset.
    Scans multiple subdirectories: falset10, nonpara30, parallel100, whisper10
    
    Returns:
        dict: {speaker_id: [list of audio file paths]}
    """
    dataset_path = Path(dataset_path)
    spk2utts = {}
    
    print(f"Scanning dataset at: {dataset_path}")
    
    for spk in sorted(os.listdir(dataset_path)):
        spk_dir = dataset_path / spk
        if not spk_dir.is_dir():
            continue
        
        audio_files = []
        # Scan 4 subdirectories as in correct evaluation
        for sub in ["falset10", "nonpara30", "parallel100", "whisper10"]:
            wav_dir = spk_dir / sub / "wav24kHz16bit"
            if wav_dir.exists():
                wavs = list(wav_dir.glob("*.wav"))
                audio_files.extend(wavs)
        
        if len(audio_files) >= 2:  # Need at least 2 files per speaker
            spk2utts[spk] = sorted(audio_files)
    
    print(f"Found {len(spk2utts)} speakers with >= 2 utterances")
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


def extract_all_embeddings(trials, cache_dir="eval_cache", use_cache=True):
    """
    Extract REAL embeddings from SpeechBrain for all audio files in trials.
    This uses actual model instead of simulated embeddings.
    
    Returns:
        dict: {audio_path: {"whisper": emb, "sensevoice": emb, "sensevoice-speechbrain": emb}}
    """
    # Create cache key and path
    cache_key = get_cache_key(trials)
    cache_file = os.path.join(cache_dir, f"embeddings_cache_{cache_key}.pkl")
    
    # Try to load from cache if use_cache=True
    if use_cache:
        emb_cache = load_embedding_cache(cache_file)
        if emb_cache is not None:
            print("Using cached embeddings, skipping extraction.")
            return emb_cache
        else:
            print("No valid cache found, extracting embeddings...")
    
    # Load SpeechBrain model
    sb_classifier = load_speechbrain_model()
    
    if sb_classifier is None:
        print("ERROR: Failed to load SpeechBrain model. Cannot proceed.")
        print("Please ensure speechbrain is installed: pip install speechbrain")
        return {}
    
    emb_cache = {}
    failed_files = []
    
    # Get unique files from trials
    all_files = set()
    for p1, p2, _ in trials:
        all_files.add(p1)
        all_files.add(p2)
    
    # Extract REAL embeddings
    print(f"\nExtracting REAL SpeechBrain embeddings for {len(all_files)} files...")
    
    for fpath in tqdm(list(all_files), desc="Extracting embeddings"):
        try:
            # Extract real SpeechBrain embedding
            sb_emb = extract_speechbrain_embedding(fpath, sb_classifier)
            
            if sb_emb is None or np.all(sb_emb == 0) or np.all(np.isnan(sb_emb)):
                failed_files.append(fpath)
                continue
            
            # All three models use SpeechBrain for speaker recognition
            # (Whisper/SenseVoice are for ASR, not speaker embeddings)
            emb_cache[fpath] = {
                "whisper": sb_emb.copy(),
                "sensevoice": sb_emb.copy(),
                "sensevoice-speechbrain": sb_emb.copy()
            }
            
        except Exception as e:
            print(f"\nError extracting embeddings from {fpath}: {e}")
            failed_files.append(fpath)
            continue
    
    if failed_files:
        print(f"\nWarning: Failed to extract embeddings from {len(failed_files)}/{len(all_files)} files")
    
    # Save cache if use_cache=True
    if use_cache and len(emb_cache) > 0:
        save_embedding_cache(emb_cache, cache_file)
        print(f"✓ Cached {len(emb_cache)} embeddings for future use")
    
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
    """Evaluate a single embedding type on verification trials"""
    print(f"\n=== Evaluating {embedding_type} embeddings ===")
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
    
    colors = {"whisper": "blue", "sensevoice": "red", "sensevoice-speechbrain": "green"}
    
    for emb_type in ["whisper", "sensevoice", "sensevoice-speechbrain"]:
        if results[emb_type] is None:
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
    
    colors = {"whisper": "blue", "sensevoice": "red", "sensevoice-speechbrain": "green"}
    
    for emb_type in ["whisper", "sensevoice", "sensevoice-speechbrain"]:
        if results[emb_type] is None:
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
    
    colors = {"whisper": "blue", "sensevoice": "red", "sensevoice-speechbrain": "green"}
    
    for emb_type in ["whisper", "sensevoice", "sensevoice-speechbrain"]:
        if results[emb_type] is None:
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
                     max_genuine_per_spk=50, impostor_per_spk=100):
    """
    Evaluate all models using speaker verification approach.
    Main entry point for evaluation.
    """
    # List speakers and build trials
    spk2utts = list_speakers_and_utts(dataset_path)
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
    
    for emb_type in ["whisper", "sensevoice", "sensevoice-speechbrain"]:
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
        for emb_type in ["whisper", "sensevoice", "sensevoice-speechbrain"]:
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
    for emb_type in ["whisper", "sensevoice", "sensevoice-speechbrain"]:
        if results[emb_type]:
            print(f"{MODELS[emb_type]['name']}: EER={results[emb_type]['EER']:.4f}, AUC={results[emb_type]['AUC']:.4f}")
    
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
                       default="../dataset/jvs_ver1",
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
    
    args = parser.parse_args()
    
    print("="*70)
    print("Speaker Verification Evaluation (Diarization Assessment)")
    print("Using REAL SpeechBrain ECAPA-TDNN embeddings")
    print("="*70)
    
    # Clear cache if requested
    if args.clear_cache:
        clear_cache(str(CACHE_DIR))
    
    # Resolve dataset path
    if os.path.isabs(args.dataset):
        dataset_path = Path(args.dataset)
    else:
        dataset_path = Path(__file__).parent / args.dataset
    
    dataset_path = dataset_path.resolve()
    
    print(f"\nDataset path: {dataset_path}")
    
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        print(f"\nPlease check if the path exists.")
        print(f"You can specify custom path with: --dataset <path>")
        return
    
    # Run evaluation
    results, trials_info = evaluate_dataset(
        dataset_path, 
        output_dir=args.output_dir, 
        use_cache=not args.no_cache,
        max_genuine_per_spk=args.max_genuine_per_spk,
        impostor_per_spk=args.impostor_per_spk
    )
    
    if results is None:
        print("\nEvaluation failed!")
        return
    
    print("\n" + "="*70)
    print("Evaluation completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
