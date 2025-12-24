"""
NeMo Speaker Diarization với Custom VAD Model

Script này integrate VAD model đã finetune vào NeMo diarization pipeline.
Sử dụng đầy đủ pipeline: VAD → Speaker Embeddings → Clustering

Usage:
    python infer_diarization.py --audio audio.wav --vad-model best_vad_model.nemo --num-speakers 2
    python infer_diarization.py --audio audio.wav --vad-model best_vad_model.nemo  # Auto-detect speakers
"""

import argparse
import torch
import soundfile as sf
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import List, Tuple, Optional
from collections import Counter

# NeMo imports
from nemo.collections.asr.models import EncDecClassificationModel, EncDecSpeakerLabelModel
from sklearn.cluster import SpectralClustering


def run_diarization_inference(
    audio_path: str,
    vad_model_path: str,
    speaker_model: str = "titanet_large",
    num_speakers: Optional[int] = None,
    vad_threshold: float = 0.5,
    vad_onset: float = 0.5,
    vad_offset: float = 0.3,
    min_speech_duration: float = 0.1,
    min_silence_duration: float = 0.3,
    embedding_window: float = 1.5,
    embedding_shift: float = 0.75,
    output_dir: str = "diar_output"
):
    """
    Run full speaker diarization with custom VAD model
    
    Args:
        audio_path: Path to input audio file
        vad_model_path: Path to finetuned VAD .nemo model
        speaker_model: Speaker embedding model name
        num_speakers: Number of speakers (optional, auto-detect if None)
        vad_threshold: VAD decision threshold
        vad_onset: VAD onset threshold
        vad_offset: VAD offset threshold
        min_speech_duration: Minimum speech segment duration
        min_silence_duration: Minimum silence to split segments
        embedding_window: Window size for embeddings
        embedding_shift: Shift for embedding windows
        output_dir: Output directory
    
    Returns:
        dict with results
    """
    
    audio_path = Path(audio_path)
    vad_model_path = Path(vad_model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 80)
    print("🎙️  NeMo Speaker Diarization (Custom VAD)")
    print("=" * 80)
    print(f"Audio: {audio_path.name}")
    print(f"VAD Model: {vad_model_path.name}")
    print(f"Speaker Model: {speaker_model}")
    print(f"Num Speakers: {num_speakers if num_speakers else 'Auto-detect'}")
    print(f"VAD Threshold: {vad_threshold}")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}\n")
    
    # Step 1: Load Audio
    print("=" * 80)
    print("STEP 1: Load Audio")
    print("=" * 80)
    audio, sr = sf.read(str(audio_path))
    
    if len(audio.shape) > 1:
        print(f"   Converting stereo to mono...")
        audio = audio.mean(axis=1)
    
    if sr != 16000:
        import librosa
        print(f"   Resampling from {sr}Hz to 16000Hz...")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    audio_duration = len(audio) / sr
    print(f"✅ Audio loaded: {audio_duration:.2f}s, {sr}Hz")
    print()
    
    # Step 2: Run VAD with Sliding Windows
    print("=" * 80)
    print("STEP 2: Voice Activity Detection (VAD)")
    print("=" * 80)
    print(f"   Loading VAD model: {vad_model_path.name}")
    
    vad_model = EncDecClassificationModel.restore_from(str(vad_model_path))
    vad_model.to(device)
    vad_model.eval()
    
    print(f"   ✅ VAD model loaded")
    
    # VAD inference with sliding windows (match training setup)
    vad_window = 1.0  # 1.0s window (như khi training)
    vad_hop = 0.5     # 0.5s hop (như khi training)
    
    print(f"   Running VAD with sliding windows (window={vad_window}s, hop={vad_hop}s)...")
    
    vad_segments = run_vad_sliding_window(
        audio=audio,
        sr=sr,
        vad_model=vad_model,
        device=device,
        window_size=vad_window,
        hop_size=vad_hop,
        threshold=vad_threshold,
        onset=vad_onset,
        offset=vad_offset,
        min_duration=min_speech_duration,
        min_silence=min_silence_duration,
    )
    
    print(f"✅ VAD completed: {len(vad_segments)} speech segments detected")
    
    if len(vad_segments) == 0:
        print("❌ No speech detected!")
        return {
            "audio_path": str(audio_path),
            "audio_duration": audio_duration,
            "segments": [],
            "speaker_labels": [],
        }
    
    # Print VAD segments
    total_speech = sum(end - start for start, end in vad_segments)
    print(f"   Total speech: {total_speech:.2f}s ({total_speech/audio_duration*100:.1f}%)")
    print(f"   Segments preview:")
    for i, (start, end) in enumerate(vad_segments[:5]):
        print(f"     #{i+1}: {start:.2f}s - {end:.2f}s ({end-start:.2f}s)")
    if len(vad_segments) > 5:
        print(f"     ... and {len(vad_segments) - 5} more")
    print()
    
    # Step 3: Create Embedding Windows
    print("=" * 80)
    print("STEP 3: Create Embedding Windows")
    print("=" * 80)
    
    embedding_segments, segment_map = create_embedding_windows(
        vad_segments=vad_segments,
        window_size=embedding_window,
        shift=embedding_shift,
    )
    
    print(f"✅ Created {len(embedding_segments)} embedding windows from {len(vad_segments)} VAD segments")
    print()
    
    # Step 4: Extract Speaker Embeddings
    print("=" * 80)
    print("STEP 4: Extract Speaker Embeddings")
    print("=" * 80)
    print(f"   Loading speaker model: {speaker_model}")
    
    speaker_model_obj = EncDecSpeakerLabelModel.from_pretrained(model_name=speaker_model)
    speaker_model_obj.freeze()
    speaker_model_obj.eval()
    speaker_model_obj.to(device)
    
    print(f"   ✅ Speaker model loaded")
    print(f"   Extracting embeddings for {len(embedding_segments)} windows...")
    
    embeddings = extract_embeddings(
        audio=audio,
        sr=sr,
        segments=embedding_segments,
        speaker_model=speaker_model_obj,
        device=device,
    )
    
    print(f"✅ Embeddings extracted: shape {embeddings.shape}")
    print()
    
    # Step 5: Speaker Clustering
    print("=" * 80)
    print("STEP 5: Speaker Clustering")
    print("=" * 80)
    
    if num_speakers is None:
        # Auto-detect number of speakers
        num_speakers = estimate_num_speakers(embeddings, max_speakers=8)
        print(f"   Auto-detected: {num_speakers} speakers")
    else:
        print(f"   Using specified: {num_speakers} speakers")
    
    if len(embeddings) < num_speakers:
        print(f"   ⚠️  Not enough segments for {num_speakers} speakers, using {len(embeddings)} clusters")
        num_speakers = len(embeddings)
    
    if num_speakers == 1:
        window_labels = [0] * len(embedding_segments)
        print(f"   Single speaker detected")
    else:
        window_labels = cluster_embeddings(
            embeddings=embeddings,
            num_clusters=num_speakers,
        )
        print(f"   ✅ Clustering completed")
    
    # Map window labels back to VAD segments
    segment_labels = map_labels_to_segments(
        window_labels=window_labels,
        segment_map=segment_map,
        num_vad_segments=len(vad_segments),
    )
    
    unique_speakers = len(set(segment_labels))
    print(f"✅ Final result: {unique_speakers} speakers detected")
    print()
    
    # Step 6: Generate RTTM
    print("=" * 80)
    print("STEP 6: Generate RTTM Output")
    print("=" * 80)
    
    rttm_lines = []
    for i, (start, end) in enumerate(vad_segments):
        duration = end - start
        speaker_id = f"speaker_{segment_labels[i]}"
        line = f"SPEAKER {audio_path.stem} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker_id} <NA> <NA>"
        rttm_lines.append(line)
    
    rttm_content = "\n".join(rttm_lines)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rttm_path = output_dir / f"{audio_path.stem}_{timestamp}.rttm"
    
    with open(rttm_path, 'w') as f:
        f.write(rttm_content)
    
    print(f"✅ RTTM saved: {rttm_path}")
    print()
    
    # Summary
    print("=" * 80)
    print("📈 SUMMARY")
    print("=" * 80)
    print(f"Audio duration:     {audio_duration:.2f}s")
    print(f"Speech segments:    {len(vad_segments)}")
    print(f"Total speech:       {total_speech:.2f}s ({total_speech/audio_duration*100:.1f}%)")
    print(f"Speakers detected:  {unique_speakers}")
    
    for spk_id in sorted(set(segment_labels)):
        spk_segments = [i for i, lbl in enumerate(segment_labels) if lbl == spk_id]
        spk_duration = sum(vad_segments[i][1] - vad_segments[i][0] for i in spk_segments)
        print(f"  speaker_{spk_id}: {len(spk_segments)} segments, {spk_duration:.2f}s")
    
    print("=" * 80)
    print()
    
    # Print segments
    print("📋 SEGMENTS:")
    print(f"{'#':<4} {'Start':<10} {'End':<10} {'Duration':<10} {'Speaker':<12}")
    print("-" * 50)
    for i, (start, end) in enumerate(vad_segments[:20]):
        duration = end - start
        speaker_id = f"speaker_{segment_labels[i]}"
        print(f"{i+1:<4} {start:<10.3f} {end:<10.3f} {duration:<10.3f} {speaker_id:<12}")
    
    if len(vad_segments) > 20:
        print(f"... and {len(vad_segments) - 20} more segments")
    
    print()
    
    return {
        "audio_path": str(audio_path),
        "audio_duration": audio_duration,
        "segments": vad_segments,
        "speaker_labels": segment_labels,
        "rttm_path": str(rttm_path),
    }


def run_vad_sliding_window(
    audio: np.ndarray,
    sr: int,
    vad_model,
    device: torch.device,
    window_size: float = 1.0,
    hop_size: float = 0.5,
    threshold: float = 0.5,
    onset: float = 0.5,
    offset: float = 0.3,
    min_duration: float = 0.1,
    min_silence: float = 0.3,
) -> List[Tuple[float, float]]:
    """
    Run VAD with sliding windows
    
    Args:
        audio: Audio signal
        sr: Sample rate
        vad_model: VAD model
        device: Device
        window_size: Window size in seconds
        hop_size: Hop size in seconds
        threshold: Decision threshold
        onset: Onset threshold
        offset: Offset threshold
        min_duration: Minimum speech duration
        min_silence: Minimum silence duration
    
    Returns:
        List of (start, end) speech segments
    """
    
    window_samples = int(window_size * sr)
    hop_samples = int(hop_size * sr)
    
    # Generate windows
    num_windows = int(np.ceil((len(audio) - window_samples) / hop_samples)) + 1
    
    speech_probs = []
    window_times = []
    
    print(f"      Processing {num_windows} windows...")
    
    with torch.no_grad():
        for i in range(num_windows):
            start_sample = i * hop_samples
            end_sample = start_sample + window_samples
            
            if end_sample > len(audio):
                # Pad last window
                window = np.pad(audio[start_sample:], (0, end_sample - len(audio)), mode='constant')
            else:
                window = audio[start_sample:end_sample]
            
            # Ensure 1D
            if len(window.shape) > 1:
                window = window.mean(axis=1)
            
            # Convert to tensor
            window_tensor = torch.from_numpy(window).float().unsqueeze(0).to(device)
            window_len = torch.tensor([len(window)]).to(device)
            
            # Get prediction
            logits = vad_model(input_signal=window_tensor, input_signal_length=window_len)
            probs = torch.softmax(logits, dim=-1)
            
            # Extract speech probability (class 1 = speech)
            if probs.dim() == 3:
                speech_prob = probs[0, -1, 1].item()
            elif probs.dim() == 2:
                speech_prob = probs[0, 1].item()
            else:
                speech_prob = probs[-1].item() if len(probs.shape) > 0 else 0.0
            
            speech_probs.append(speech_prob)
            window_times.append(i * hop_size)
            
            if (i + 1) % 100 == 0:
                print(f"         {i+1}/{num_windows} windows processed")
    
    speech_probs = np.array(speech_probs)
    
    print(f"      Speech probability range: [{speech_probs.min():.3f}, {speech_probs.max():.3f}]")
    print(f"      Mean: {speech_probs.mean():.3f}")
    
    # Convert probabilities to segments
    segments = []
    in_speech = False
    segment_start = 0.0
    
    for i, prob in enumerate(speech_probs):
        time = window_times[i]
        
        if prob >= onset:
            if not in_speech:
                in_speech = True
                segment_start = time
        else:
            if in_speech and prob < offset:
                in_speech = False
                segments.append((segment_start, time + window_size))
    
    # Close last segment
    if in_speech:
        segments.append((segment_start, len(audio) / sr))
    
    print(f"      Raw segments: {len(segments)}")
    
    # Filter short segments
    segments = [(s, e) for s, e in segments if (e - s) >= min_duration]
    print(f"      After filtering (min {min_duration}s): {len(segments)}")
    
    # Merge close segments
    if len(segments) > 0:
        merged = [segments[0]]
        for current in segments[1:]:
            last = merged[-1]
            gap = current[0] - last[1]
            
            if gap < min_silence:
                merged[-1] = (last[0], current[1])
            else:
                merged.append(current)
        
        segments = merged
        print(f"      After merging (min silence {min_silence}s): {len(segments)}")
    
    return segments


def create_embedding_windows(
    vad_segments: List[Tuple[float, float]],
    window_size: float,
    shift: float,
) -> Tuple[List[Tuple[float, float]], List[int]]:
    """
    Create sliding windows for embedding extraction
    
    Args:
        vad_segments: VAD segments
        window_size: Window size
        shift: Window shift
    
    Returns:
        (embedding_windows, segment_map)
    """
    
    embedding_windows = []
    segment_map = []  # Maps each window to its parent VAD segment
    
    for seg_idx, (vad_start, vad_end) in enumerate(vad_segments):
        vad_duration = vad_end - vad_start
        
        if vad_duration < window_size:
            # Too short, use as is
            embedding_windows.append((vad_start, vad_end))
            segment_map.append(seg_idx)
        else:
            # Apply sliding window
            current = vad_start
            while current + window_size <= vad_end:
                embedding_windows.append((current, current + window_size))
                segment_map.append(seg_idx)
                current += shift
            
            # Add last window if needed
            if current < vad_end:
                embedding_windows.append((vad_end - window_size, vad_end))
                segment_map.append(seg_idx)
    
    return embedding_windows, segment_map


def extract_embeddings(
    audio: np.ndarray,
    sr: int,
    segments: List[Tuple[float, float]],
    speaker_model,
    device: torch.device,
) -> np.ndarray:
    """
    Extract speaker embeddings
    
    Args:
        audio: Audio signal
        sr: Sample rate
        segments: Segments to extract embeddings from
        speaker_model: Speaker model
        device: Device
    
    Returns:
        Embeddings array (n_segments, embedding_dim)
    """
    
    embeddings = []
    
    with torch.no_grad():
        for i, (start, end) in enumerate(segments):
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment_audio = audio[start_sample:end_sample]
            
            # Ensure 1D
            if len(segment_audio.shape) > 1:
                segment_audio = segment_audio.mean(axis=1)
            
            # Skip if too short
            if len(segment_audio) < sr * 0.1:
                embeddings.append(np.zeros(192))  # Placeholder
                continue
            
            # Convert to tensor
            segment_tensor = torch.from_numpy(segment_audio).float().unsqueeze(0).to(device)
            segment_len = torch.tensor([len(segment_audio)]).to(device)
            
            # Extract embedding
            _, emb = speaker_model(
                input_signal=segment_tensor,
                input_signal_length=segment_len
            )
            emb = emb.squeeze(0).cpu().numpy()
            
            embeddings.append(emb)
            
            if (i + 1) % 50 == 0:
                print(f"      {i+1}/{len(segments)} embeddings extracted")
    
    return np.array(embeddings)


def cluster_embeddings(
    embeddings: np.ndarray,
    num_clusters: int,
    affinity_threshold: float = 0.15,
) -> List[int]:
    """
    Cluster embeddings using spectral clustering
    
    Args:
        embeddings: Embeddings array
        num_clusters: Number of clusters
        affinity_threshold: Affinity threshold
    
    Returns:
        Cluster labels
    """
    
    # Normalize embeddings
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Compute cosine similarity
    affinity = np.dot(embeddings_norm, embeddings_norm.T)
    
    # Apply threshold
    affinity = np.where(affinity > affinity_threshold, affinity, 0)
    
    # Ensure symmetry
    affinity = (affinity + affinity.T) / 2
    affinity = np.maximum(affinity, 0)
    
    # Spectral clustering
    clustering = SpectralClustering(
        n_clusters=num_clusters,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=42
    )
    
    labels = clustering.fit_predict(affinity)
    
    return labels.tolist()


def map_labels_to_segments(
    window_labels: List[int],
    segment_map: List[int],
    num_vad_segments: int,
) -> List[int]:
    """
    Map window labels to VAD segments using majority voting
    
    Args:
        window_labels: Labels for embedding windows
        segment_map: Maps window index to VAD segment index
        num_vad_segments: Number of VAD segments
    
    Returns:
        Labels for VAD segments
    """
    
    segment_labels = []
    
    for seg_idx in range(num_vad_segments):
        # Get all window labels for this segment
        window_indices = [i for i, s in enumerate(segment_map) if s == seg_idx]
        labels_for_segment = [window_labels[i] for i in window_indices]
        
        # Majority vote
        if labels_for_segment:
            most_common = Counter(labels_for_segment).most_common(1)[0][0]
            segment_labels.append(most_common)
        else:
            segment_labels.append(0)
    
    return segment_labels


def estimate_num_speakers(
    embeddings: np.ndarray,
    max_speakers: int = 8,
) -> int:
    """
    Estimate number of speakers using eigenvalue gap
    
    Args:
        embeddings: Embeddings array
        max_speakers: Maximum number of speakers
    
    Returns:
        Estimated number of speakers
    """
    
    # Simple heuristic: use spectral clustering with different k values
    # and pick k with best silhouette score
    
    if len(embeddings) <= 2:
        return 1
    
    # For simplicity, use a heuristic based on embedding diversity
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    affinity = np.dot(embeddings_norm, embeddings_norm.T)
    
    # Check pairwise similarities
    mean_similarity = affinity.mean()
    
    # Heuristic: if mean similarity is high, likely fewer speakers
    if mean_similarity > 0.7:
        return 2
    elif mean_similarity > 0.5:
        return 3
    else:
        return min(4, max(2, len(embeddings) // 10))


def main():
    parser = argparse.ArgumentParser(description="NeMo Speaker Diarization with Custom VAD")
    
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to input audio file"
    )
    
    parser.add_argument(
        "--vad-model",
        type=str,
        default="best_vad_model.nemo",
        help="Path to finetuned VAD .nemo model"
    )
    
    parser.add_argument(
        "--speaker-model",
        type=str,
        default="titanet_large",
        help="Speaker embedding model name (default: titanet_large)"
    )
    
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Number of speakers (optional, auto-detect if not specified)"
    )
    
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="VAD threshold (default: 0.5)"
    )
    
    parser.add_argument(
        "--vad-onset",
        type=float,
        default=0.5,
        help="VAD onset threshold (default: 0.5)"
    )
    
    parser.add_argument(
        "--vad-offset",
        type=float,
        default=0.3,
        help="VAD offset threshold (default: 0.3)"
    )
    
    parser.add_argument(
        "--min-speech",
        type=float,
        default=0.1,
        help="Minimum speech duration (default: 0.1s)"
    )
    
    parser.add_argument(
        "--min-silence",
        type=float,
        default=0.3,
        help="Minimum silence duration (default: 0.3s)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="diar_output",
        help="Output directory (default: diar_output)"
    )
    
    args = parser.parse_args()
    
    try:
        result = run_diarization_inference(
            audio_path=args.audio,
            vad_model_path=args.vad_model,
            speaker_model=args.speaker_model,
            num_speakers=args.num_speakers,
            vad_threshold=args.vad_threshold,
            vad_onset=args.vad_onset,
            vad_offset=args.vad_offset,
            min_speech_duration=args.min_speech,
            min_silence_duration=args.min_silence,
            output_dir=args.output_dir,
        )
        
        print("✅ SUCCESS!\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}\n")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
