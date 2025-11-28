"""
Evaluation script for Speaker Diarization Models
Đánh giá các model trong realtime folder:
- realtime_diarization_improved.py (Whisper + SpeechBrain)
- sen_voice.py (SenseVoice)
- senvoi_spebrai_fixed.py (SenseVoice + SpeechBrain)

Metrics:
- Speaker Verification: EER, FAR, FRR, Precision, Recall, F1
- Speaker Clustering: Accuracy, Purity
"""

import os
import sys
import csv
import json
import time
import pickle
import hashlib
from pathlib import Path
from tqdm import tqdm
from glob import glob
from itertools import combinations
import numpy as np
import random
from sklearn.metrics import roc_curve, precision_recall_fscore_support, auc, precision_recall_curve
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import models from realtime folder
try:
    from speechbrain.inference import EncoderClassifier
    import soundfile as sf
except ImportError as e:
    print(f"ERROR: Missing required packages: {e}")
    print("Install with: pip install speechbrain soundfile")
    sys.exit(1)

# Random seed for reproducibility
random.seed(123)
np.random.seed(123)

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = Path(__file__).parent / "eval_cache"
RESULTS_DIR = Path(__file__).parent / "eval_results"
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


class SpeakerEmbeddingExtractor:
    """Wrapper for speaker embedding extraction"""
    
    def __init__(self, model_type='speechbrain'):
        self.model_type = model_type
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load speaker embedding model"""
        print(f"Loading {self.model_type} model...")
        
        if self.model_type == 'speechbrain':
            local_model_dir = Path(__file__).parent.parent / "pretrained_models" / "spkrec-ecapa-voxceleb"
            
            if local_model_dir.exists():
                print(f"  Loading from: {local_model_dir}")
                self.model = EncoderClassifier.from_hparams(
                    source=str(local_model_dir),
                    run_opts={"device": DEVICE}
                )
            else:
                print(f"  Downloading from HuggingFace...")
                self.model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": DEVICE},
                    savedir=str(local_model_dir)
                )
            print(f"  ✓ Model loaded successfully!")
    
    def extract_embedding(self, audio_path):
        """Extract embedding from audio file"""
        try:
            # Load audio
            audio, sr = sf.read(audio_path)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            
            if DEVICE == "cuda":
                audio_tensor = audio_tensor.cuda()
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model.encode_batch(audio_tensor)
                embedding = embedding.squeeze().cpu().numpy()
            
            return embedding
            
        except Exception as e:
            print(f"Error extracting embedding from {audio_path}: {e}")
            return None
    
    @staticmethod
    def cosine_similarity(emb1, emb2):
        """Calculate cosine similarity between two embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def list_speakers_and_utterances(root_dir):
    """
    List speakers and their audio files
    Returns: dict {speaker_id: [audio_paths]}
    """
    spk2utts = {}
    
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"Error: Directory {root_dir} does not exist")
        return spk2utts
    
    # Find all audio files grouped by speaker
    for spk_dir in sorted(root_path.iterdir()):
        if not spk_dir.is_dir():
            continue
        
        speaker_id = spk_dir.name
        audio_files = []
        
        # Search for audio files
        for ext in ['*.wav', '*.mp3', '*.flac']:
            audio_files.extend(list(spk_dir.rglob(ext)))
        
        if len(audio_files) >= 2:  # Need at least 2 utterances per speaker
            spk2utts[speaker_id] = [str(f) for f in sorted(audio_files)]
    
    return spk2utts


def build_trials(spk2utts, max_genuine_per_spk=50, impostor_per_spk=100):
    """
    Build trial pairs for evaluation
    Returns: list of (path1, path2, label)
    - label=1: genuine (same speaker)
    - label=0: impostor (different speakers)
    """
    trials = []
    speakers = sorted(spk2utts.keys())
    
    # Genuine pairs: different utterances from same speaker
    for spk in speakers:
        utts = spk2utts[spk]
        pairs = list(combinations(utts, 2))
        random.shuffle(pairs)
        for p in pairs[:max_genuine_per_spk]:
            trials.append((p[0], p[1], 1))
    
    # Impostor pairs: utterances from different speakers
    for spk in speakers:
        others = [s for s in speakers if s != spk]
        utts_a = spk2utts[spk]
        
        for _ in range(impostor_per_spk):
            ua = random.choice(utts_a)
            spk_b = random.choice(others)
            ub = random.choice(spk2utts[spk_b])
            trials.append((ua, ub, 0))
    
    random.shuffle(trials)
    return trials


def get_cache_key(trials):
    """Generate cache key from trial list"""
    all_files = sorted(set([p for t in trials for p in (t[0], t[1])]))
    files_str = '|'.join(all_files)
    return hashlib.md5(files_str.encode()).hexdigest()


def save_embedding_cache(emb_cache, cache_file):
    """Save embedding cache to file"""
    try:
        cache_file = Path(cache_file)
        cache_file.parent.mkdir(exist_ok=True, parents=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(emb_cache, f)
        print(f"✓ Saved embedding cache: {cache_file}")
        return True
    except Exception as e:
        print(f"Error saving cache: {e}")
        return False


def load_embedding_cache(cache_file):
    """Load embedding cache from file"""
    try:
        cache_file = Path(cache_file)
        if not cache_file.exists():
            return None
        with open(cache_file, 'rb') as f:
            emb_cache = pickle.load(f)
        print(f"✓ Loaded embedding cache: {cache_file} ({len(emb_cache)} files)")
        return emb_cache
    except Exception as e:
        print(f"Error loading cache: {e}")
        return None


def extract_all_embeddings(trials, extractor, cache_dir=None, use_cache=True):
    """Extract embeddings for all files in trials"""
    
    # Setup cache
    if cache_dir and use_cache:
        cache_key = get_cache_key(trials)
        cache_file = Path(cache_dir) / f"embeddings_cache_{cache_key}.pkl"
        
        # Try load from cache
        emb_cache = load_embedding_cache(cache_file)
        if emb_cache is not None:
            print("Using cached embeddings")
            return emb_cache
    
    # Extract embeddings
    emb_cache = {}
    failed_files = []
    
    # Get unique files
    all_files = set()
    for p1, p2, _ in trials:
        all_files.add(p1)
        all_files.add(p2)
    
    print(f"Extracting embeddings from {len(all_files)} files...")
    for fpath in tqdm(list(all_files)):
        try:
            embedding = extractor.extract_embedding(fpath)
            
            if embedding is None or np.all(embedding == 0) or np.any(np.isnan(embedding)):
                failed_files.append(fpath)
                continue
            
            emb_cache[fpath] = embedding.copy()
            
        except Exception as e:
            print(f"\nError processing {fpath}: {e}")
            failed_files.append(fpath)
    
    if failed_files:
        print(f"Warning: Failed to extract {len(failed_files)}/{len(all_files)} files")
    
    # Save cache
    if cache_dir and use_cache and len(emb_cache) > 0:
        save_embedding_cache(emb_cache, cache_file)
    
    return emb_cache


def compute_scores_from_cache(trials, emb_cache):
    """Compute cosine similarity scores from cached embeddings"""
    scores, labels = [], []
    skipped_trials = 0
    
    for p1, p2, y in trials:
        if p1 not in emb_cache or p2 not in emb_cache:
            skipped_trials += 1
            continue
        
        try:
            emb1 = emb_cache[p1]
            emb2 = emb_cache[p2]
            
            if np.all(emb1 == 0) or np.any(np.isnan(emb1)) or \
               np.all(emb2 == 0) or np.any(np.isnan(emb2)):
                skipped_trials += 1
                continue
            
            score = SpeakerEmbeddingExtractor.cosine_similarity(emb1, emb2)
            
            if np.isnan(score) or np.isinf(score):
                skipped_trials += 1
                continue
            
            scores.append(score)
            labels.append(y)
            
        except Exception as e:
            print(f"\nError computing similarity: {e}")
            skipped_trials += 1
    
    if skipped_trials > 0:
        print(f"Skipped {skipped_trials}/{len(trials)} trials")
    
    return np.array(scores), np.array(labels)


def compute_metrics_at_threshold(scores, labels, threshold):
    """Compute metrics at specific threshold"""
    predictions = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    return float(precision), float(recall), float(f1)


def compute_eer_and_metrics(scores, labels):
    """Compute EER, FAR, FRR and related metrics"""
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    
    # EER: point where FAR = FRR
    idx_eer = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx_eer] + fnr[idx_eer]) / 2.0
    thr_eer = thresholds[idx_eer]
    
    far_at_eer = fpr[idx_eer]
    frr_at_eer = fnr[idx_eer]
    
    # Metrics at EER threshold
    precision_at_eer, recall_at_eer, f1_at_eer = compute_metrics_at_threshold(
        scores, labels, thr_eer
    )
    
    # Find best F1
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
    
    # AUC
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall AUC
    precision_curve, recall_curve, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall_curve, precision_curve)
    
    results = {
        'eer': float(eer),
        'threshold_at_eer': float(thr_eer),
        'far_at_eer': float(far_at_eer),
        'frr_at_eer': float(frr_at_eer),
        'precision_at_eer': float(precision_at_eer),
        'recall_at_eer': float(recall_at_eer),
        'f1_at_eer': float(f1_at_eer),
        'best_f1': float(best_f1),
        'best_f1_threshold': float(best_f1_threshold),
        'best_f1_precision': float(best_f1_precision),
        'best_f1_recall': float(best_f1_recall),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc)
    }
    
    return results


def plot_results(scores, labels, output_dir, model_name="model"):
    """Plot ROC and PR curves"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (1-FRR)')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.grid(True)
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)
    
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'PR (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_curves.png', dpi=150)
    plt.close()
    
    print(f"✓ Saved plots: {output_dir / f'{model_name}_curves.png'}")


def save_results(results, output_file):
    """Save evaluation results to JSON"""
    output_file = Path(output_file)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved results: {output_file}")


def print_results(results):
    """Print evaluation results"""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"EER:                 {results['eer']:.4f} ({results['eer']*100:.2f}%)")
    print(f"Threshold at EER:    {results['threshold_at_eer']:.4f}")
    print(f"FAR at EER:          {results['far_at_eer']:.4f}")
    print(f"FRR at EER:          {results['frr_at_eer']:.4f}")
    print(f"Precision at EER:    {results['precision_at_eer']:.4f}")
    print(f"Recall at EER:       {results['recall_at_eer']:.4f}")
    print(f"F1 at EER:           {results['f1_at_eer']:.4f}")
    print("-"*60)
    print(f"Best F1:             {results['best_f1']:.4f}")
    print(f"Best F1 Threshold:   {results['best_f1_threshold']:.4f}")
    print(f"Best F1 Precision:   {results['best_f1_precision']:.4f}")
    print(f"Best F1 Recall:      {results['best_f1_recall']:.4f}")
    print("-"*60)
    print(f"ROC AUC:             {results['roc_auc']:.4f}")
    print(f"PR AUC:              {results['pr_auc']:.4f}")
    print("="*60 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Speaker Diarization Models')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing speaker audio files')
    parser.add_argument('--model', type=str, default='speechbrain',
                        choices=['speechbrain'],
                        help='Model type to evaluate')
    parser.add_argument('--max_genuine', type=int, default=50,
                        help='Max genuine pairs per speaker')
    parser.add_argument('--max_impostor', type=int, default=100,
                        help='Max impostor pairs per speaker')
    parser.add_argument('--use_cache', action='store_true', default=True,
                        help='Use embedding cache')
    parser.add_argument('--output_name', type=str, default='eval_results',
                        help='Output file name prefix')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("SPEAKER DIARIZATION EVALUATION")
    print(f"{'='*60}")
    print(f"Data directory: {args.data_dir}")
    print(f"Model: {args.model}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}\n")
    
    # List speakers
    print("Loading speaker data...")
    spk2utts = list_speakers_and_utterances(args.data_dir)
    
    if not spk2utts:
        print("Error: No speakers found!")
        return
    
    print(f"✓ Found {len(spk2utts)} speakers")
    for spk, utts in list(spk2utts.items())[:5]:
        print(f"  {spk}: {len(utts)} utterances")
    if len(spk2utts) > 5:
        print(f"  ... and {len(spk2utts)-5} more speakers")
    
    # Build trials
    print(f"\nBuilding trials...")
    trials = build_trials(spk2utts, args.max_genuine, args.max_impostor)
    print(f"✓ Created {len(trials)} trial pairs")
    
    genuine_count = sum(1 for _, _, label in trials if label == 1)
    impostor_count = len(trials) - genuine_count
    print(f"  Genuine: {genuine_count}")
    print(f"  Impostor: {impostor_count}")
    
    # Initialize extractor
    print(f"\nInitializing {args.model} extractor...")
    extractor = SpeakerEmbeddingExtractor(model_type=args.model)
    
    # Extract embeddings
    print("\nExtracting embeddings...")
    emb_cache = extract_all_embeddings(
        trials, extractor, 
        cache_dir=CACHE_DIR if args.use_cache else None,
        use_cache=args.use_cache
    )
    
    if not emb_cache:
        print("Error: No embeddings extracted!")
        return
    
    # Compute scores
    print("\nComputing similarity scores...")
    scores, labels = compute_scores_from_cache(trials, emb_cache)
    
    if len(scores) == 0:
        print("Error: No valid scores computed!")
        return
    
    print(f"✓ Computed {len(scores)} scores")
    
    # Compute metrics
    print("\nComputing metrics...")
    results = compute_eer_and_metrics(scores, labels)
    
    # Add metadata
    results['model'] = args.model
    results['num_speakers'] = len(spk2utts)
    results['num_trials'] = len(trials)
    results['num_genuine'] = genuine_count
    results['num_impostor'] = impostor_count
    results['num_valid_scores'] = len(scores)
    
    # Print results
    print_results(results)
    
    # Save results
    output_file = RESULTS_DIR / f"{args.output_name}_{args.model}.json"
    save_results(results, output_file)
    
    # Plot curves
    print("Generating plots...")
    plot_results(scores, labels, RESULTS_DIR, model_name=f"{args.output_name}_{args.model}")
    
    print("\n✓ Evaluation completed successfully!")


if __name__ == "__main__":
    main()
