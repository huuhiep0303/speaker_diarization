"""
Clustering Diarization Pipeline - Step-by-Step Implementation
==============================================================
Mô phỏng chi tiết từng bước của NeMo clustering diarization pipeline:
1. Audio Input
2. Voice Activity Detection (VAD)
3. Speech Segmentation
4. Speaker Embeddings Extraction
5. Affinity Matrix Construction
6. Laplacian Matrix
7. Eigenvectors Computation
8. Spectral Clustering

Mỗi bước sẽ log chi tiết output và lưu vào file .txt

Usage:
    python clustering_diarize.py audio.wav --output-log clustering_log.txt
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import numpy as np
import soundfile as sf
import torch
import warnings

# NeMo imports
from nemo.collections.asr.models import EncDecClassificationModel, EncDecSpeakerLabelModel
from nemo.collections.asr.parts.utils.vad_utils import (
    generate_vad_segment_table,
    vad_construct_pyannote_object_per_file,
)

# Clustering imports
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import cdist, squareform
from scipy.linalg import eigh
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")


class ClusteringDiarizationPipeline:
    """
    Step-by-step clustering diarization pipeline with detailed logging
    """
    
    def __init__(
        self,
        vad_model: str = "vad_multilingual_marblenet",
        speaker_model: str = "titanet_large",
        device: str = "cpu",
        log_file: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize pipeline
        
        Parameters
        ----------
        vad_model : str
            VAD model name
        speaker_model : str
            Speaker embedding model name
        device : str
            Device to run on
        log_file : str, optional
            Path to log file
        verbose : bool
            Print to console
        """
        self.device = device
        self.verbose = verbose
        self.log_file = log_file
        self.log_buffer = []
        
        # Store intermediate results for evaluation
        self.last_embeddings = None
        self.last_labels = None
        self.last_affinity = None
        self.last_laplacian = None
        self.last_eigenvalues = None
        self.last_eigenvectors = None
        
        # Parameters
        self.sample_rate = 16000
        self.vad_window = 0.025  # 25ms
        self.vad_shift = 0.01    # 10ms
        self.vad_onset = 0.3     # Lowered from 0.5 to detect more speech
        self.vad_offset = 0.3    # Lowered from 0.5
        self.vad_min_duration_on = 0.1   # Lowered from 0.2 to catch shorter segments
        self.vad_min_duration_off = 0.1  # Lowered from 0.2
        
        self.emb_window = 1.5    # 1.5s
        self.emb_shift = 0.75    # 0.75s
        
        self.max_num_speakers = 8
        self.max_rp_threshold = 0.15
        
        self.log("="*80)
        self.log("CLUSTERING DIARIZATION PIPELINE - INITIALIZATION")
        self.log("="*80)
        self.log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Device: {device}")
        self.log(f"VAD Model: {vad_model}")
        self.log(f"Speaker Model: {speaker_model}")
        self.log("")
        
        # Load models
        self.log(">>> Step 0: Loading Models")
        self.log("-" * 60)
        
        self.log("Loading VAD model...")
        self.vad_model = EncDecClassificationModel.from_pretrained(
            model_name=vad_model
        )
        self.vad_model.freeze()
        self.vad_model.eval()
        self.vad_model.to(device)
        self.log(f"✓ VAD model loaded: {vad_model}")
        
        self.log("Loading Speaker Embedding model...")
        self.speaker_model = EncDecSpeakerLabelModel.from_pretrained(
            model_name=speaker_model
        )
        self.speaker_model.freeze()
        self.speaker_model.eval()
        self.speaker_model.to(device)
        self.log(f"✓ Speaker model loaded: {speaker_model}")
        
        self.log("")
    
    def log(self, message: str):
        """Log message to buffer and optionally print"""
        self.log_buffer.append(message)
        if self.verbose:
            print(message)
    
    def save_log(self):
        """Save log buffer to file"""
        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.log_buffer))
            
            self.log(f"\n💾 Log saved to: {log_path}")
    
    def process_audio(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None
    ) -> List[Dict]:
        """
        Run complete diarization pipeline with detailed logging
        
        Parameters
        ----------
        audio_path : str
            Path to audio file
        num_speakers : int, optional
            Oracle number of speakers
        
        Returns
        -------
        List[Dict]
            Diarization segments
        """
        audio_path = Path(audio_path)
        
        self.log("="*80)
        self.log("STARTING DIARIZATION PIPELINE")
        self.log("="*80)
        self.log(f"Audio: {audio_path.name}")
        self.log(f"Full Path: {audio_path.absolute()}")
        self.log("")
        
        # Step 1: Load Audio
        self.log(">>> Step 1: Audio Input")
        self.log("-" * 60)
        audio, sr = sf.read(str(audio_path))
        
        # Convert stereo to mono if needed
        if len(audio.shape) > 1:
            self.log(f"⚠ Audio is multi-channel with shape {audio.shape}, converting to mono")
            audio = np.mean(audio, axis=1)
        
        duration = len(audio) / sr
        self.log(f"Sample Rate: {sr} Hz")
        self.log(f"Duration: {duration:.2f} seconds")
        self.log(f"Samples: {len(audio)}")
        self.log(f"Audio Shape: {audio.shape}")
        
        # Resample if needed
        if sr != self.sample_rate:
            self.log(f"⚠ Resampling from {sr} Hz to {self.sample_rate} Hz")
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        self.log(f"✓ Audio loaded successfully")
        self.log("")
        
        # Step 2: Voice Activity Detection
        self.log(">>> Step 2: Voice Activity Detection (VAD)")
        self.log("-" * 60)
        vad_segments = self._run_vad(audio, sr)
        self.log(f"✓ VAD completed: {len(vad_segments)} speech segments detected")
        self.log("")
        
        # Step 3: Speech Segmentation
        self.log(">>> Step 3: Speech Segmentation")
        self.log("-" * 60)
        speech_segments = self._create_segments(vad_segments, duration)
        self.log(f"✓ Created {len(speech_segments)} segments for embedding extraction")
        self.log("")
        
        # Step 4: Speaker Embeddings
        self.log(">>> Step 4: Speaker Embeddings Extraction")
        self.log("-" * 60)
        embeddings = self._extract_embeddings(audio, sr, speech_segments)
        
        if len(embeddings) == 0:
            self.log("❌ No embeddings extracted, cannot continue pipeline")
            self.log("")
            # Store empty results
            self.last_embeddings = embeddings
            self.last_labels = np.array([])
            self.last_affinity = None
            self.last_laplacian = None
            self.last_eigenvalues = None
            self.last_eigenvectors = None
            return []
        
        self.log(f"✓ Extracted {len(embeddings)} embeddings")
        self.log(f"Embedding dimension: {embeddings.shape[1]}")
        self.log("")
        
        # Step 5: Affinity Matrix
        self.log(">>> Step 5: Affinity Matrix Construction")
        self.log("-" * 60)
        affinity_matrix = self._compute_affinity_matrix(embeddings)
        self.log(f"✓ Affinity matrix computed: {affinity_matrix.shape}")
        self.log("")
        
        # Step 6: Laplacian Matrix
        self.log(">>> Step 6: Laplacian Matrix")
        self.log("-" * 60)
        laplacian = self._compute_laplacian(affinity_matrix)
        self.log(f"✓ Laplacian matrix computed: {laplacian.shape}")
        self.log("")
        
        # Step 7: Eigenvectors
        self.log(">>> Step 7: Eigenvectors Computation")
        self.log("-" * 60)
        eigenvalues, eigenvectors = self._compute_eigenvectors(laplacian, num_speakers)
        self.log(f"✓ Computed eigenvectors: {eigenvectors.shape}")
        self.log("")
        
        # Step 8: Clustering
        self.log(">>> Step 8: Spectral Clustering")
        self.log("-" * 60)
        labels = self._perform_clustering(eigenvectors, num_speakers)
        self.log(f"✓ Clustering completed")
        self.log("")
        
        # Store intermediate results for evaluation
        self.last_embeddings = embeddings
        self.last_labels = labels
        self.last_affinity = affinity_matrix
        self.last_laplacian = laplacian
        self.last_eigenvalues = eigenvalues
        self.last_eigenvectors = eigenvectors
        
        # Generate final segments
        self.log(">>> Step 9: Generate Final Segments")
        self.log("-" * 60)
        diar_segments = self._generate_segments(speech_segments, labels)
        self.log(f"✓ Generated {len(diar_segments)} diarization segments")
        
        # Summary
        speakers = set(seg['speaker'] for seg in diar_segments)
        self.log("")
        self.log("="*80)
        self.log("PIPELINE COMPLETED")
        self.log("="*80)
        self.log(f"Total segments: {len(diar_segments)}")
        self.log(f"Number of speakers detected: {len(speakers)}")
        self.log(f"Speakers: {sorted(speakers)}")
        self.log("="*80)
        
        return diar_segments
    
    def _run_vad(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """
        Run VAD on audio
        
        Returns
        -------
        List[Tuple[float, float]]
            List of (start, end) speech segments
        """
        self.log(f"VAD Parameters:")
        self.log(f"  Window: {self.vad_window}s")
        self.log(f"  Shift: {self.vad_shift}s")
        self.log(f"  Onset: {self.vad_onset}")
        self.log(f"  Offset: {self.vad_offset}")
        self.log(f"  Min duration ON: {self.vad_min_duration_on}s")
        self.log(f"  Min duration OFF: {self.vad_min_duration_off}s")
        self.log("")
        
        # Ensure audio is 1D
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Process audio in sliding windows for frame-level VAD
        frame_len = int(self.vad_window * sr)
        frame_shift = int(self.vad_shift * sr)
        
        num_frames = int(np.ceil((len(audio) - frame_len) / frame_shift)) + 1
        self.log(f"Processing {num_frames} frames (window={self.vad_window}s, shift={self.vad_shift}s)")
        
        speech_probs = []
        
        self.log("Running VAD inference on sliding windows...")
        with torch.no_grad():
            for i in range(num_frames):
                start = i * frame_shift
                end = start + frame_len
                
                # Handle last frame
                if end > len(audio):
                    frame = np.pad(audio[start:], (0, end - len(audio)), mode='constant')
                else:
                    frame = audio[start:end]
                
                # Convert to tensor
                frame_tensor = torch.from_numpy(frame).float().unsqueeze(0).to(self.device)
                frame_len_tensor = torch.tensor([len(frame)]).to(self.device)
                
                # Get VAD prediction
                logits = self.vad_model(input_signal=frame_tensor, input_signal_length=frame_len_tensor)[0]
                probs = torch.softmax(logits, dim=-1)
                
                # Extract speech probability (class 1)
                if len(probs.shape) >= 1:
                    speech_prob = probs[-1].item() if len(probs.shape) == 1 else probs[0, -1].item()
                else:
                    speech_prob = 0.0
                
                speech_probs.append(speech_prob)
        
        speech_probs = np.array(speech_probs)
        
        self.log(f"VAD output shape: {speech_probs.shape}")
        self.log(f"Speech probability range: [{speech_probs.min():.4f}, {speech_probs.max():.4f}]")
        self.log(f"Mean speech probability: {speech_probs.mean():.4f}")
        
        # Apply thresholds
        speech_frames = speech_probs > self.vad_onset
        
        self.log(f"Frames above onset threshold: {speech_frames.sum()}/{len(speech_frames)}")
        
        # Convert to segments
        segments = []
        in_speech = False
        start_frame = 0
        
        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_speech:
                start_frame = i
                in_speech = True
            elif not is_speech and in_speech:
                # Check offset threshold
                if speech_probs[i] < self.vad_offset:
                    start_time = start_frame * self.vad_shift
                    end_time = i * self.vad_shift
                    duration = end_time - start_time
                    
                    if duration >= self.vad_min_duration_on:
                        segments.append((start_time, end_time))
                    
                    in_speech = False
        
        # Handle last segment
        if in_speech:
            start_time = start_frame * self.vad_shift
            end_time = len(speech_frames) * self.vad_shift
            duration = end_time - start_time
            if duration >= self.vad_min_duration_on:
                segments.append((start_time, end_time))
        
        # Merge close segments
        merged_segments = []
        if segments:
            current_start, current_end = segments[0]
            
            for start, end in segments[1:]:
                gap = start - current_end
                if gap < self.vad_min_duration_off:
                    # Merge
                    current_end = end
                else:
                    merged_segments.append((current_start, current_end))
                    current_start, current_end = start, end
            
            merged_segments.append((current_start, current_end))
        
        self.log(f"Speech segments (before merge): {len(segments)}")
        self.log(f"Speech segments (after merge): {len(merged_segments)}")
        
        # Log first few segments
        for i, (start, end) in enumerate(merged_segments[:10]):
            self.log(f"  Segment {i+1}: [{start:.2f}s - {end:.2f}s] (duration: {end-start:.2f}s)")
        
        if len(merged_segments) > 10:
            self.log(f"  ... and {len(merged_segments) - 10} more segments")
        
        total_speech = sum(end - start for start, end in merged_segments)
        self.log(f"Total speech time: {total_speech:.2f}s")
        
        return merged_segments
    
    def _create_segments(
        self,
        vad_segments: List[Tuple[float, float]],
        audio_duration: float
    ) -> List[Tuple[float, float]]:
        """
        Create sliding window segments from VAD output
        
        Returns
        -------
        List[Tuple[float, float]]
            Segments for embedding extraction
        """
        self.log(f"Segmentation Parameters:")
        self.log(f"  Window: {self.emb_window}s")
        self.log(f"  Shift: {self.emb_shift}s")
        self.log("")
        
        segments = []
        
        for vad_start, vad_end in vad_segments:
            vad_duration = vad_end - vad_start
            
            if vad_duration < self.emb_window:
                # Segment too short, use as is
                segments.append((vad_start, vad_end))
            else:
                # Apply sliding window
                current = vad_start
                while current + self.emb_window <= vad_end:
                    segments.append((current, current + self.emb_window))
                    current += self.emb_shift
                
                # Add last segment if needed
                if current < vad_end:
                    segments.append((vad_end - self.emb_window, vad_end))
        
        self.log(f"Created {len(segments)} segments from {len(vad_segments)} VAD segments")
        
        # Log statistics
        if segments:
            segment_durations = [end - start for start, end in segments]
            self.log(f"Segment duration range: [{min(segment_durations):.2f}s, {max(segment_durations):.2f}s]")
            self.log(f"Mean segment duration: {np.mean(segment_durations):.2f}s")
            
            # Log first few segments
            for i, (start, end) in enumerate(segments[:10]):
                self.log(f"  Segment {i+1}: [{start:.2f}s - {end:.2f}s]")
            
            if len(segments) > 10:
                self.log(f"  ... and {len(segments) - 10} more segments")
        else:
            self.log("⚠️ No segments created!")
        
        return segments
    
    def _extract_embeddings(
        self,
        audio: np.ndarray,
        sr: int,
        segments: List[Tuple[float, float]]
    ) -> np.ndarray:
        """
        Extract speaker embeddings for each segment
        
        Returns
        -------
        np.ndarray
            Embeddings matrix (n_segments, embedding_dim)
        """
        self.log(f"Extracting embeddings for {len(segments)} segments...")
        self.log("")
        
        if len(segments) == 0:
            self.log("⚠️ No segments to process, returning empty embeddings")
            return np.array([])
        
        embeddings = []
        
        for i, (start, end) in enumerate(segments):
            # Extract audio segment
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment_audio = audio[start_sample:end_sample]
            
            # Ensure segment is 1D
            if len(segment_audio.shape) > 1:
                segment_audio = np.mean(segment_audio, axis=1)
            
            # Convert to tensor
            segment_tensor = torch.from_numpy(segment_audio).float().unsqueeze(0).to(self.device)
            segment_len = torch.tensor([len(segment_audio)]).to(self.device)
            
            # Extract embedding
            with torch.no_grad():
                _, emb = self.speaker_model(
                    input_signal=segment_tensor,
                    input_signal_length=segment_len
                )
                emb = emb.squeeze(0).cpu().numpy()
            
            embeddings.append(emb)
            
            if (i + 1) % 20 == 0 or (i + 1) == len(segments):
                self.log(f"  Processed {i+1}/{len(segments)} segments")
        
        embeddings = np.array(embeddings)
        
        if len(embeddings) == 0:
            self.log("⚠️ No embeddings extracted")
            return embeddings
        
        self.log("")
        self.log(f"Embeddings shape: {embeddings.shape}")
        self.log(f"Embedding dimension: {embeddings.shape[1]}")
        self.log(f"Embedding norm range: [{np.linalg.norm(embeddings, axis=1).min():.4f}, {np.linalg.norm(embeddings, axis=1).max():.4f}]")
        
        return embeddings
    
    def _compute_affinity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute affinity matrix from embeddings using cosine similarity
        
        Returns
        -------
        np.ndarray
            Affinity matrix (n_segments, n_segments)
        """
        self.log("Computing cosine similarity matrix...")
        
        # Normalize embeddings
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute cosine similarity
        affinity = np.dot(embeddings_norm, embeddings_norm.T)
        
        self.log(f"Affinity matrix shape: {affinity.shape}")
        self.log(f"Affinity range: [{affinity.min():.4f}, {affinity.max():.4f}]")
        self.log(f"Mean affinity: {affinity.mean():.4f}")
        self.log(f"Diagonal mean: {np.diag(affinity).mean():.4f}")
        
        # Apply threshold-based refinement (optional)
        self.log(f"Applying affinity refinement with threshold: {self.max_rp_threshold}")
        affinity_refined = np.where(affinity > self.max_rp_threshold, affinity, 0)
        
        sparsity = (affinity_refined == 0).sum() / affinity_refined.size
        self.log(f"Sparsity after refinement: {sparsity:.2%}")
        
        # Ensure symmetry
        affinity_refined = (affinity_refined + affinity_refined.T) / 2
        
        return affinity_refined
    
    def _compute_laplacian(self, affinity: np.ndarray) -> np.ndarray:
        """
        Compute normalized Laplacian matrix
        
        Returns
        -------
        np.ndarray
            Laplacian matrix
        """
        self.log("Computing Laplacian matrix...")
        
        # Degree matrix
        degree = np.sum(affinity, axis=1)
        self.log(f"Degree range: [{degree.min():.4f}, {degree.max():.4f}]")
        self.log(f"Mean degree: {degree.mean():.4f}")
        
        # Avoid division by zero
        degree_inv_sqrt = np.where(degree > 1e-8, 1.0 / np.sqrt(degree), 0)
        
        # Normalized Laplacian: L = D^(-1/2) * A * D^(-1/2)
        D_inv_sqrt = np.diag(degree_inv_sqrt)
        laplacian = D_inv_sqrt @ affinity @ D_inv_sqrt
        
        self.log(f"Laplacian shape: {laplacian.shape}")
        self.log(f"Laplacian range: [{laplacian.min():.4f}, {laplacian.max():.4f}]")
        self.log(f"Laplacian mean: {laplacian.mean():.4f}")
        
        return laplacian
    
    def _compute_eigenvectors(
        self,
        laplacian: np.ndarray,
        num_speakers: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvectors of Laplacian matrix
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (eigenvalues, eigenvectors)
        """
        self.log("Computing eigendecomposition...")
        
        # Compute all eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigh(laplacian)
        
        # Sort by eigenvalue (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        self.log(f"Computed {len(eigenvalues)} eigenvalues")
        self.log(f"Eigenvalue range: [{eigenvalues.min():.6f}, {eigenvalues.max():.6f}]")
        
        # Log top eigenvalues
        self.log("Top 10 eigenvalues:")
        for i in range(min(10, len(eigenvalues))):
            self.log(f"  λ_{i+1} = {eigenvalues[i]:.6f}")
        
        # Estimate number of speakers from eigengap
        if num_speakers is None:
            num_speakers = self._estimate_num_speakers(eigenvalues)
        
        self.log(f"Using {num_speakers} eigenvectors for clustering")
        
        # Select top k eigenvectors
        selected_eigenvectors = eigenvectors[:, :num_speakers]
        
        self.log(f"Selected eigenvectors shape: {selected_eigenvectors.shape}")
        
        return eigenvalues, selected_eigenvectors
    
    def _estimate_num_speakers(self, eigenvalues: np.ndarray) -> int:
        """
        Estimate number of speakers from eigenvalue spectrum
        
        Returns
        -------
        int
            Estimated number of speakers
        """
        self.log("Estimating number of speakers from eigengap...")
        
        # Compute eigengaps
        eigengaps = np.diff(eigenvalues[:self.max_num_speakers])
        
        self.log("Eigengaps:")
        for i, gap in enumerate(eigengaps):
            self.log(f"  Gap {i+1}-{i+2}: {gap:.6f}")
        
        # Find largest eigengap
        max_gap_idx = np.argmax(eigengaps)
        num_speakers = max_gap_idx + 1
        
        self.log(f"Largest eigengap at position {max_gap_idx + 1}")
        self.log(f"Estimated number of speakers: {num_speakers}")
        
        # Apply constraints
        num_speakers = max(1, min(num_speakers, self.max_num_speakers))
        
        return num_speakers
    
    def _perform_clustering(
        self,
        eigenvectors: np.ndarray,
        num_speakers: Optional[int] = None
    ) -> np.ndarray:
        """
        Perform k-means clustering on eigenvectors
        
        Returns
        -------
        np.ndarray
            Cluster labels
        """
        if num_speakers is None:
            num_speakers = eigenvectors.shape[1]
        
        self.log(f"Performing k-means clustering with k={num_speakers}...")
        
        # Normalize eigenvectors (row-wise)
        eigenvectors_norm = normalize(eigenvectors, axis=1, norm='l2')
        
        self.log(f"Normalized eigenvectors shape: {eigenvectors_norm.shape}")
        
        # K-means clustering
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=num_speakers, random_state=42, n_init=10)
        labels = kmeans.fit_predict(eigenvectors_norm)
        
        self.log(f"Clustering completed")
        self.log(f"Labels shape: {labels.shape}")
        self.log(f"Unique labels: {np.unique(labels)}")
        
        # Count per cluster
        for i in range(num_speakers):
            count = np.sum(labels == i)
            self.log(f"  Cluster {i}: {count} segments ({count/len(labels)*100:.1f}%)")
        
        return labels
    
    def _generate_segments(
        self,
        segments: List[Tuple[float, float]],
        labels: np.ndarray
    ) -> List[Dict]:
        """
        Generate final diarization segments with speaker labels
        
        Returns
        -------
        List[Dict]
            Diarization segments
        """
        self.log("Generating final segments...")
        
        # Create segments with labels
        labeled_segments = []
        for (start, end), label in zip(segments, labels):
            labeled_segments.append({
                "start": start,
                "end": end,
                "speaker": f"speaker_{label}"
            })
        
        # Sort by start time
        labeled_segments.sort(key=lambda x: x['start'])
        
        # Merge consecutive segments with same speaker
        merged = []
        if labeled_segments:
            current = labeled_segments[0].copy()
            
            for seg in labeled_segments[1:]:
                if seg['speaker'] == current['speaker'] and seg['start'] - current['end'] < 0.5:
                    # Merge
                    current['end'] = seg['end']
                else:
                    merged.append(current)
                    current = seg.copy()
            
            merged.append(current)
        
        self.log(f"Segments before merge: {len(labeled_segments)}")
        self.log(f"Segments after merge: {len(merged)}")
        
        # Log first few segments
        for i, seg in enumerate(merged[:10]):
            self.log(f"  [{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['speaker']}")
        
        if len(merged) > 10:
            self.log(f"  ... and {len(merged) - 10} more segments")
        
        return merged


def main():
    parser = argparse.ArgumentParser(
        description="Clustering Diarization Pipeline with Detailed Logging"
    )
    parser.add_argument("audio_path", type=str, help="Path to audio file")
    parser.add_argument("--vad-model", type=str, default="vad_multilingual_marblenet",
                        help="VAD model name")
    parser.add_argument("--speaker-model", type=str, default="titanet_large",
                        help="Speaker embedding model name")
    parser.add_argument("--num-speakers", type=int, default=None,
                        help="Oracle number of speakers")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    parser.add_argument("--output-log", type=str, default=None,
                        help="Output log file path")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Output JSON file for segments")
    parser.add_argument("--no-verbose", action='store_true',
                        help="Disable console output")
    
    args = parser.parse_args()
    
    # Determine device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Default log file
    if args.output_log is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_name = Path(args.audio_path).stem
        args.output_log = f"clustering_log_{audio_name}_{timestamp}.txt"
    
    # Initialize pipeline
    pipeline = ClusteringDiarizationPipeline(
        vad_model=args.vad_model,
        speaker_model=args.speaker_model,
        device=device,
        log_file=args.output_log,
        verbose=not args.no_verbose
    )
    
    # Run pipeline
    segments = pipeline.process_audio(
        args.audio_path,
        num_speakers=args.num_speakers
    )
    
    # Save log
    pipeline.save_log()
    
    # Save segments to JSON if requested
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Segments saved to: {output_path}")
    
    print(f"\n✅ Pipeline completed!")
    print(f"   Log file: {args.output_log}")
    if args.output_json:
        print(f"   Segments: {args.output_json}")


if __name__ == "__main__":
    main()
