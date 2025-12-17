"""
NeMo Speaker Diarization - Spectral Clustering Evaluation
==========================================================

Đánh giá Spectral Clustering cho NeMo TitaNet speaker diarization.

Spectral Clustering:
- Sử dụng graph-based approach
- Phù hợp cho non-convex clusters
- Affinity matrix từ cosine similarity
- Eigen decomposition cho clustering
- Auto-detect số clusters dựa trên silhouette score

Metrics đánh giá:
1. Clustering Quality:
   - Silhouette Score: Độ phân tách giữa clusters [-1, 1]
   - Davies-Bouldin Index: Độ compact và separation [0, ∞)
   - Calinski-Harabasz Score: Variance ratio [0, ∞)
   
2. Purity & Coverage:
   - Purity: % segments trong dominant cluster
   - Coverage: % duration covered by dominant speaker
   - Perfect Clustering Rate: % files với 1 cluster + 100% purity
   
3. Speaker Count Accuracy:
   - Accuracy: % files với số speaker chính xác (=1 cho JVS)
   - Mean Absolute Error: Sai số trung bình
   - Over/Under-segmentation rates
   
4. Distance Metrics:
   - Intra-cluster: Độ compact của clusters
   - Inter-cluster: Độ phân tách giữa clusters
   - Separation Ratio: inter/intra (higher is better)

Dataset: JVS Corpus (Japanese single-speaker audio)
Target: Mỗi file nên có 1 cluster với 100% purity
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Dependencies check
try:
    from sklearn.metrics import (
        silhouette_score, 
        davies_bouldin_score, 
        calinski_harabasz_score
    )
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.spatial.distance import cdist
    import torch
    import soundfile as sf
except ImportError as e:
    print("ERROR: Missing required packages!")
    print("Install with: pip install scikit-learn scipy torch soundfile")
    print(f"Details: {e}")
    sys.exit(1)

try:
    from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
except ImportError:
    print("ERROR: NeMo not installed!")
    print("Install with: pip install nemo_toolkit[asr]")
    sys.exit(1)

# Configuration
RESULTS_DIR = Path(__file__).parent / "eval_results"
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================
#   NEMO SPECTRAL CLUSTERING DIARIZATION
# ============================================

class NeMoSpectralDiarization:
    """
    NeMo TitaNet + Spectral Clustering cho speaker diarization
    Tối ưu hóa cho single-speaker detection
    """
    
    def __init__(self, 
                 pretrained_model="titanet_large",
                 window_length_sec=2.0,
                 shift_length_sec=1.0,
                 affinity_threshold=0.9,
                 spectral_gamma=1.0,
                 max_speakers=2):
        """
        Parameters
        ----------
        pretrained_model : str
            NeMo model name (titanet_large recommended)
        window_length_sec : float
            Độ dài mỗi segment
        shift_length_sec : float
            Shift giữa segments
        affinity_threshold : float
            Ngưỡng để sparse affinity matrix (0.0-1.0)
            Higher = stricter clustering
        spectral_gamma : float
            RBF kernel gamma (unused if using cosine similarity)
        max_speakers : int
            Số speakers tối đa để test
        """
        self.window_length_sec = window_length_sec
        self.shift_length_sec = shift_length_sec
        self.affinity_threshold = affinity_threshold
        self.spectral_gamma = spectral_gamma
        self.max_speakers = max_speakers
        self.sample_rate = 16000
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🚀 Loading NeMo {pretrained_model} on {self.device}...")
        print(f"   Affinity threshold: {affinity_threshold}")
        print(f"   Max speakers: {max_speakers}")
        
        self.speaker_model = EncDecSpeakerLabelModel.from_pretrained(
            model_name=pretrained_model
        )
        self.speaker_model.freeze()
        self.speaker_model.eval()
        self.speaker_model.to(self.device)
        print(f"✅ Model loaded successfully")
    
    def segment_audio(self, audio: np.ndarray) -> List[Tuple[np.ndarray, float, float]]:
        """Segment audio thành overlapping windows"""
        window_samples = int(self.window_length_sec * self.sample_rate)
        shift_samples = int(self.shift_length_sec * self.sample_rate)
        
        segments = []
        start = 0
        
        while start < len(audio):
            end = min(start + window_samples, len(audio))
            segment = audio[start:end]
            
            if len(segment) >= self.sample_rate * 0.5:  # Min 0.5s
                segments.append((
                    segment, 
                    start / self.sample_rate, 
                    end / self.sample_rate
                ))
            
            start += shift_samples
            if end >= len(audio):
                break
        
        return segments
    
    def extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Extract L2-normalized speaker embedding"""
        audio_signal = torch.tensor(audio, device=self.device, dtype=torch.float32).unsqueeze(0)
        audio_signal_len = torch.tensor([len(audio)], device=self.device)
        
        with torch.no_grad():
            _, emb = self.speaker_model.forward(audio_signal, audio_signal_len)
            emb = emb.squeeze(0).detach().cpu().numpy()
        
        # L2 normalize
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
        return emb_norm
    
    def cluster_spectral(self, embeddings: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Spectral clustering với auto-detection của số clusters
        
        Strategy:
        1. Compute cosine similarity affinity matrix
        2. Apply threshold để làm sparse
        3. Test từ 1 đến max_speakers clusters
        4. Chọn số clusters cho silhouette score cao nhất
        """
        if len(embeddings) == 1:
            return np.array([0]), 1
        
        # Compute affinity matrix từ cosine similarity
        affinity = cosine_similarity(embeddings)
        
        # Apply threshold để sparse matrix
        # Điều này giúp tách các clusters rõ ràng hơn
        affinity[affinity < self.affinity_threshold] = 0
        
        # Ensure diagonal is 1
        np.fill_diagonal(affinity, 1.0)
        
        # Try different numbers of clusters
        best_n = 1
        best_score = -np.inf
        best_labels = np.zeros(len(embeddings), dtype=int)
        
        for n_clusters in range(1, min(self.max_speakers + 1, len(embeddings))):
            if n_clusters == 1:
                labels = np.zeros(len(embeddings), dtype=int)
                score = 0.0  # Single cluster always gives 0 score
            else:
                try:
                    # Spectral clustering với precomputed affinity
                    clusterer = SpectralClustering(
                        n_clusters=n_clusters,
                        affinity='precomputed',
                        assign_labels='kmeans',
                        random_state=42
                    )
                    labels = clusterer.fit_predict(affinity)
                    
                    # Compute silhouette score
                    # Higher is better
                    score = silhouette_score(embeddings, labels, metric='cosine')
                    
                except Exception as e:
                    # If clustering fails, skip
                    continue
            
            # Prefer fewer clusters if scores are similar
            # Penalize more clusters slightly
            adjusted_score = score - 0.05 * (n_clusters - 1)
            
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_n = n_clusters
                best_labels = labels
        
        return best_labels, best_n
    
    def process_audio(self, audio_file: str) -> Dict:
        """Process audio file và return diarization results"""
        # Load audio
        audio, sr = sf.read(audio_file)
        
        # Resample if needed
        if sr != self.sample_rate:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            except ImportError:
                raise ImportError("librosa required for resampling. Install with: pip install librosa")
        
        # Convert to mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Segment and extract embeddings
        segments = self.segment_audio(audio)
        embeddings = []
        time_ranges = []
        
        for segment_audio, start_time, end_time in segments:
            emb = self.extract_embedding(segment_audio)
            embeddings.append(emb)
            time_ranges.append((start_time, end_time))
        
        embeddings = np.array(embeddings)
        
        # Cluster with spectral clustering
        labels, n_clusters = self.cluster_spectral(embeddings)
        
        # Convert to speaker IDs
        speaker_labels = [f"speaker_{label}" for label in labels]
        
        return {
            'embeddings': embeddings,
            'cluster_labels': labels,
            'speaker_labels': speaker_labels,
            'time_ranges': time_ranges,
            'num_speakers': n_clusters,
            'num_segments': len(embeddings)
        }


# ============================================
#   DATASET LOADING
# ============================================

def load_jvs_dataset(dataset_root: str, max_speakers: Optional[int] = None) -> List[Dict]:
    """Load JVS dataset"""
    dataset_root = Path(dataset_root)
    dataset = []
    
    print(f"📂 Loading JVS dataset from: {dataset_root}")
    
    speaker_dirs = sorted([d for d in dataset_root.iterdir() 
                          if d.is_dir() and d.name.startswith('jvs')])
    
    if max_speakers:
        speaker_dirs = speaker_dirs[:max_speakers]
    
    for spk_dir in speaker_dirs:
        speaker_id = spk_dir.name
        
        for sub_type in ["parallel100", "nonpara30", "falset10", "whisper10"]:
            wav_dir = spk_dir / sub_type / "wav24kHz16bit"
            
            if not wav_dir.exists():
                continue
            
            for wav_file in sorted(wav_dir.glob("*.wav")):
                dataset.append({
                    'audio_file': str(wav_file),
                    'speaker_id': speaker_id,
                    'utterance_type': sub_type,
                    'filename': wav_file.name
                })
    
    print(f"✓ Loaded {len(dataset)} files from {len(speaker_dirs)} speakers")
    return dataset


# ============================================
#   EVALUATION METRICS
# ============================================

def evaluate_clustering_quality(embeddings: np.ndarray, labels: np.ndarray) -> Dict:
    """Evaluate clustering quality metrics"""
    n_clusters = len(np.unique(labels))
    
    if n_clusters < 2 or len(embeddings) < 2:
        return {
            'silhouette_score': None,
            'davies_bouldin_index': None,
            'calinski_harabasz_score': None,
            'n_clusters': n_clusters,
            'n_samples': len(embeddings)
        }
    
    try:
        sil = silhouette_score(embeddings, labels, metric='cosine')
    except:
        sil = None
    
    try:
        db = davies_bouldin_score(embeddings, labels)
    except:
        db = None
    
    try:
        ch = calinski_harabasz_score(embeddings, labels)
    except:
        ch = None
    
    return {
        'silhouette_score': float(sil) if sil is not None else None,
        'davies_bouldin_index': float(db) if db is not None else None,
        'calinski_harabasz_score': float(ch) if ch is not None else None,
        'n_clusters': int(n_clusters),
        'n_samples': int(len(embeddings))
    }


def compute_purity_coverage(embeddings: np.ndarray, labels: np.ndarray) -> Dict:
    """Compute purity and coverage for single-speaker audio"""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Count samples per cluster
    cluster_counts = {label: int(np.sum(labels == label)) for label in unique_labels}
    
    # Dominant cluster
    dominant_cluster = max(cluster_counts, key=cluster_counts.get)
    dominant_count = cluster_counts[dominant_cluster]
    total_count = len(labels)
    
    purity = dominant_count / total_count if total_count > 0 else 0.0
    coverage = purity  # Same for uniform segments
    
    return {
        'purity': float(purity),
        'coverage': float(coverage),
        'n_clusters': int(n_clusters),
        'dominant_cluster_size': int(dominant_count),
        'total_segments': int(total_count),
        'is_perfect': bool(n_clusters == 1 and purity == 1.0)
    }


def compute_distances(embeddings: np.ndarray, labels: np.ndarray) -> Dict:
    """Compute intra and inter cluster distances"""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Intra-cluster distances
    intra_dists = []
    for label in unique_labels:
        cluster_embs = embeddings[labels == label]
        if len(cluster_embs) < 2:
            continue
        centroid = cluster_embs.mean(axis=0)
        dists = cdist(cluster_embs, [centroid], metric='cosine').flatten()
        intra_dists.append(float(dists.mean()))
    
    avg_intra = np.mean(intra_dists) if intra_dists else 0.0
    
    # Inter-cluster distances
    inter_dist = None
    if n_clusters >= 2:
        centroids = []
        for label in unique_labels:
            cluster_embs = embeddings[labels == label]
            centroids.append(cluster_embs.mean(axis=0))
        
        centroids = np.array(centroids)
        pairwise = cdist(centroids, centroids, metric='cosine')
        
        inter_dists = []
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                inter_dists.append(pairwise[i, j])
        
        inter_dist = float(np.mean(inter_dists)) if inter_dists else None
    
    return {
        'avg_intra_cluster_distance': float(avg_intra),
        'avg_inter_cluster_distance': inter_dist,
        'separation_ratio': float(inter_dist / avg_intra) if inter_dist and avg_intra > 0 else None
    }


def evaluate_speaker_count(predicted: int, true: int = 1) -> Dict:
    """Evaluate speaker count accuracy"""
    error = predicted - true
    
    return {
        'true_count': int(true),
        'predicted_count': int(predicted),
        'error': int(error),
        'absolute_error': int(abs(error)),
        'is_correct': bool(predicted == true),
        'over_segmentation': bool(predicted > true),
        'under_segmentation': bool(predicted < true)
    }


# ============================================
#   EVALUATION PIPELINE
# ============================================

def evaluate_single_audio(diarizer: NeMoSpectralDiarization,
                         audio_file: str,
                         true_speaker_id: str) -> Dict:
    """Evaluate single audio file"""
    try:
        result = diarizer.process_audio(audio_file)
        
        embeddings = result['embeddings']
        labels = result['cluster_labels']
        num_speakers = result['num_speakers']
        
        # Metrics
        clustering_metrics = evaluate_clustering_quality(embeddings, labels)
        purity_metrics = compute_purity_coverage(embeddings, labels)
        distance_metrics = compute_distances(embeddings, labels)
        count_metrics = evaluate_speaker_count(num_speakers, true=1)
        
        return {
            'success': True,
            'audio_file': audio_file,
            'true_speaker': true_speaker_id,
            'clustering_quality': clustering_metrics,
            'purity_coverage': purity_metrics,
            'distances': distance_metrics,
            'speaker_count': count_metrics
        }
        
    except Exception as e:
        return {
            'success': False,
            'audio_file': audio_file,
            'true_speaker': true_speaker_id,
            'error': str(e)
        }


def aggregate_results(results: List[Dict]) -> Dict:
    """Aggregate evaluation results"""
    successful = [r for r in results if r['success']]
    n_total = len(results)
    n_success = len(successful)
    
    if n_success == 0:
        return {'n_total': n_total, 'n_success': 0, 'success_rate': 0.0}
    
    # Clustering quality
    sil_scores = [r['clustering_quality']['silhouette_score'] 
                 for r in successful if r['clustering_quality']['silhouette_score'] is not None]
    db_indices = [r['clustering_quality']['davies_bouldin_index'] 
                 for r in successful if r['clustering_quality']['davies_bouldin_index'] is not None]
    ch_scores = [r['clustering_quality']['calinski_harabasz_score'] 
                for r in successful if r['clustering_quality']['calinski_harabasz_score'] is not None]
    
    # Purity
    purities = [r['purity_coverage']['purity'] for r in successful]
    coverages = [r['purity_coverage']['coverage'] for r in successful]
    perfect_count = sum(1 for r in successful if r['purity_coverage']['is_perfect'])
    
    # Speaker count
    count_correct = sum(1 for r in successful if r['speaker_count']['is_correct'])
    abs_errors = [r['speaker_count']['absolute_error'] for r in successful]
    over_seg = sum(1 for r in successful if r['speaker_count']['over_segmentation'])
    
    # Distances
    intra_dists = [r['distances']['avg_intra_cluster_distance'] for r in successful]
    inter_dists = [r['distances']['avg_inter_cluster_distance'] 
                  for r in successful if r['distances']['avg_inter_cluster_distance'] is not None]
    sep_ratios = [r['distances']['separation_ratio'] 
                 for r in successful if r['distances']['separation_ratio'] is not None]
    
    return {
        'n_total': n_total,
        'n_success': n_success,
        'success_rate': n_success / n_total,
        
        'clustering_quality': {
            'silhouette_score': {
                'mean': float(np.mean(sil_scores)) if sil_scores else None,
                'std': float(np.std(sil_scores)) if sil_scores else None,
                'median': float(np.median(sil_scores)) if sil_scores else None,
            },
            'davies_bouldin_index': {
                'mean': float(np.mean(db_indices)) if db_indices else None,
                'std': float(np.std(db_indices)) if db_indices else None,
                'median': float(np.median(db_indices)) if db_indices else None,
            },
            'calinski_harabasz_score': {
                'mean': float(np.mean(ch_scores)) if ch_scores else None,
                'std': float(np.std(ch_scores)) if ch_scores else None,
                'median': float(np.median(ch_scores)) if ch_scores else None,
            }
        },
        
        'purity_coverage': {
            'mean_purity': float(np.mean(purities)),
            'std_purity': float(np.std(purities)),
            'median_purity': float(np.median(purities)),
            'mean_coverage': float(np.mean(coverages)),
            'perfect_clustering_rate': perfect_count / n_success,
        },
        
        'speaker_count_accuracy': {
            'accuracy': count_correct / n_success,
            'mean_absolute_error': float(np.mean(abs_errors)),
            'median_absolute_error': float(np.median(abs_errors)),
            'over_segmentation_rate': over_seg / n_success,
        },
        
        'distances': {
            'mean_intra_cluster': float(np.mean(intra_dists)),
            'mean_inter_cluster': float(np.mean(inter_dists)) if inter_dists else None,
            'mean_separation_ratio': float(np.mean(sep_ratios)) if sep_ratios else None,
        }
    }


# ============================================
#   MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="NeMo Spectral Clustering Evaluation for Speaker Diarization"
    )
    parser.add_argument("--dataset", type=str,
                       default="../dataset/jvs_ver1/jvs_ver1",
                       help="JVS dataset root path")
    parser.add_argument("--max_speakers", type=int, default=None,
                       help="Max speakers to evaluate (None = all)")
    parser.add_argument("--max_files_per_speaker", type=int, default=None,
                       help="Max files per speaker (None = all)")
    parser.add_argument("--model", type=str, default="titanet_large",
                       help="NeMo pretrained model")
    parser.add_argument("--affinity_threshold", type=float, default=0.9,
                       help="Affinity threshold for sparse matrix (0.0-1.0)")
    parser.add_argument("--max_speakers_detect", type=int, default=2,
                       help="Max speakers to detect in clustering")
    parser.add_argument("--output_dir", type=str, default="eval_results",
                       help="Output directory")
    
    args = parser.parse_args()
    
    print("="*80)
    print("NeMo Spectral Clustering Evaluation")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Affinity threshold: {args.affinity_threshold}")
    print(f"Max speakers detect: {args.max_speakers_detect}")
    print("="*80)
    
    # Load dataset
    dataset_path = Path(__file__).parent / args.dataset if not os.path.isabs(args.dataset) else Path(args.dataset)
    dataset_path = dataset_path.resolve()
    
    if not dataset_path.exists():
        print(f"❌ ERROR: Dataset not found at {dataset_path}")
        return
    
    dataset = load_jvs_dataset(dataset_path, max_speakers=args.max_speakers)
    
    if len(dataset) == 0:
        print("❌ ERROR: No audio files found!")
        return
    
    # Filter dataset
    if args.max_files_per_speaker:
        speaker_files = {}
        for item in dataset:
            spk = item['speaker_id']
            if spk not in speaker_files:
                speaker_files[spk] = []
            if len(speaker_files[spk]) < args.max_files_per_speaker:
                speaker_files[spk].append(item)
        
        eval_dataset = []
        for spk in sorted(speaker_files.keys()):
            eval_dataset.extend(speaker_files[spk])
        
        print(f"📊 Evaluating {len(eval_dataset)} files from {len(speaker_files)} speakers")
    else:
        eval_dataset = dataset
        print(f"📊 Evaluating all {len(eval_dataset)} files")
    
    print()
    
    # Initialize diarizer
    diarizer = NeMoSpectralDiarization(
        pretrained_model=args.model,
        affinity_threshold=args.affinity_threshold,
        max_speakers=args.max_speakers_detect
    )
    print()
    
    # Evaluate
    results = []
    for item in tqdm(eval_dataset, desc="Processing"):
        result = evaluate_single_audio(
            diarizer=diarizer,
            audio_file=item['audio_file'],
            true_speaker_id=item['speaker_id']
        )
        results.append(result)
    
    # Aggregate
    print("\n📊 Aggregating results...")
    aggregated = aggregate_results(results)
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    print(f"\nSuccess: {aggregated['n_success']}/{aggregated['n_total']} ({aggregated['success_rate']*100:.1f}%)")
    
    # Clustering quality
    print("\n--- Clustering Quality ---")
    cq = aggregated['clustering_quality']
    if cq['silhouette_score']['mean'] is not None:
        print(f"Silhouette Score: {cq['silhouette_score']['mean']:.4f} ± {cq['silhouette_score']['std']:.4f}")
        print(f"  Median: {cq['silhouette_score']['median']:.4f}")
    if cq['davies_bouldin_index']['mean'] is not None:
        print(f"Davies-Bouldin: {cq['davies_bouldin_index']['mean']:.4f} ± {cq['davies_bouldin_index']['std']:.4f}")
    if cq['calinski_harabasz_score']['mean'] is not None:
        print(f"Calinski-Harabasz: {cq['calinski_harabasz_score']['mean']:.2f} ± {cq['calinski_harabasz_score']['std']:.2f}")
    
    # Purity
    print("\n--- Purity & Coverage ---")
    pc = aggregated['purity_coverage']
    print(f"Mean Purity: {pc['mean_purity']*100:.2f}% ± {pc['std_purity']*100:.2f}%")
    print(f"  Median: {pc['median_purity']*100:.2f}%")
    print(f"Perfect Clustering Rate: {pc['perfect_clustering_rate']*100:.2f}%")
    
    # Speaker count
    print("\n--- Speaker Count Accuracy ---")
    sca = aggregated['speaker_count_accuracy']
    print(f"Accuracy: {sca['accuracy']*100:.2f}% (correct = 1 speaker)")
    print(f"Mean Error: {sca['mean_absolute_error']:.2f} speakers")
    print(f"Over-segmentation: {sca['over_segmentation_rate']*100:.2f}%")
    
    # Distances
    print("\n--- Distance Metrics ---")
    dm = aggregated['distances']
    print(f"Intra-cluster: {dm['mean_intra_cluster']:.4f} (lower = more compact)")
    if dm['mean_inter_cluster'] is not None:
        print(f"Inter-cluster: {dm['mean_inter_cluster']:.4f} (higher = more separated)")
    if dm['mean_separation_ratio'] is not None:
        print(f"Separation ratio: {dm['mean_separation_ratio']:.2f} (higher = better)")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    config_name = f"spectral_aff{args.affinity_threshold:.2f}"
    
    # Detailed
    detail_file = output_dir / f"spectral_detailed_{config_name}_{timestamp}.json"
    with open(detail_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Detailed results: {detail_file}")
    
    # Summary
    summary_file = output_dir / f"spectral_summary_{config_name}_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)
    print(f"✓ Summary: {summary_file}")
    
    # Log
    log_file = output_dir / f"spectral_log_{config_name}_{timestamp}.txt"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("NeMo Spectral Clustering Evaluation\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Affinity threshold: {args.affinity_threshold}\n")
        f.write(f"Max speakers detect: {args.max_speakers_detect}\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Date: {datetime.now().isoformat()}\n\n")
        
        f.write(f"Success: {aggregated['n_success']}/{aggregated['n_total']}\n\n")
        
        f.write("Clustering Quality:\n")
        if cq['silhouette_score']['mean'] is not None:
            f.write(f"  Silhouette: {cq['silhouette_score']['mean']:.4f} ± {cq['silhouette_score']['std']:.4f}\n")
        if cq['davies_bouldin_index']['mean'] is not None:
            f.write(f"  Davies-Bouldin: {cq['davies_bouldin_index']['mean']:.4f} ± {cq['davies_bouldin_index']['std']:.4f}\n")
        if cq['calinski_harabasz_score']['mean'] is not None:
            f.write(f"  Calinski-Harabasz: {cq['calinski_harabasz_score']['mean']:.2f} ± {cq['calinski_harabasz_score']['std']:.2f}\n")
        
        f.write(f"\nPurity & Coverage:\n")
        f.write(f"  Mean Purity: {pc['mean_purity']*100:.2f}%\n")
        f.write(f"  Perfect Rate: {pc['perfect_clustering_rate']*100:.2f}%\n")
        
        f.write(f"\nSpeaker Count:\n")
        f.write(f"  Accuracy: {sca['accuracy']*100:.2f}%\n")
        f.write(f"  Mean Error: {sca['mean_absolute_error']:.2f}\n")
        f.write(f"  Over-seg: {sca['over_segmentation_rate']*100:.2f}%\n")
        
        f.write(f"\nDistances:\n")
        f.write(f"  Intra: {dm['mean_intra_cluster']:.4f}\n")
        if dm['mean_inter_cluster'] is not None:
            f.write(f"  Inter: {dm['mean_inter_cluster']:.4f}\n")
        if dm['mean_separation_ratio'] is not None:
            f.write(f"  Ratio: {dm['mean_separation_ratio']:.2f}\n")
    
    print(f"✓ Log: {log_file}")
    
    print("\n" + "="*80)
    print("✅ Evaluation completed!")
    print("="*80)


if __name__ == "__main__":
    main()
