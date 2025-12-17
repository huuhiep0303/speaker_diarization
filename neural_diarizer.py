"""
Neural Diarizer Pipeline - NeMo ClusteringDiarizer with Detailed Logging
=========================================================================
Pipeline sử dụng NeMo ClusteringDiarizer để thực hiện speaker diarization.

Components:
- VAD (Voice Activity Detection): MarbleNet multilingual
- Speaker Embeddings: TitaNet Large
- Clustering: Spectral clustering with affinity refinement

Pipeline Steps with Logging:
1. Audio Input & Validation
2. Manifest Creation
3. VAD Processing
4. Speech Segmentation
5. Speaker Embeddings Extraction
6. Clustering (Affinity Matrix → Laplacian → Eigenvectors → K-means)
7. Post-processing & Results

Usage:
    from neural_diarizer import NeuralDiarizer
    
    diarizer = NeuralDiarizer(config_path="diar_infer_config.yaml", log_file="pipeline.log")
    results = diarizer.diarize_audio("audio.wav")
    
    # Results format:
    # [
    #     {"start": 0.0, "end": 2.5, "speaker": "speaker_0"},
    #     {"start": 2.5, "end": 5.0, "speaker": "speaker_1"},
    #     ...
    # ]
"""

import os
import json
import yaml
import tempfile
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import soundfile as sf
import torch
from omegaconf import OmegaConf

# NeMo imports
from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.parts.utils.manifest_utils import create_manifest

warnings.filterwarnings("ignore")


@dataclass
class DiarizationSegment:
    """Represents a speaker diarization segment"""
    start: float
    end: float
    speaker: str
    
    def to_dict(self) -> Dict:
        return {
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker
        }
    
    def __repr__(self):
        return f"[{self.start:.2f}s - {self.end:.2f}s] {self.speaker}"


class NeuralDiarizer:
    """
    Neural Speaker Diarization using NeMo ClusteringDiarizer with Detailed Logging
    
    Pipeline steps:
    1. Audio Input & Validation
    2. Manifest Creation
    3. VAD Processing
    4. Speech Segmentation
    5. Speaker Embeddings Extraction
    6. Clustering (Affinity → Laplacian → Eigenvectors → K-means)
    7. Post-processing & Results
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        output_dir: str = "diar_output",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_num_speakers: int = 8,
        oracle_num_speakers: Optional[int] = None,
        log_file: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize Neural Diarizer
        
        Parameters
        ----------
        config_path : str, optional
            Path to diarization config YAML file
        output_dir : str
            Directory to save diarization outputs
        device : str
            Device to run on ("cuda" or "cpu")
        max_num_speakers : int
            Maximum number of speakers to detect
        oracle_num_speakers : int, optional
            If provided, use this exact number of speakers
        log_file : str, optional
            Path to log file for detailed pipeline logging
        verbose : bool
            Enable verbose logging
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.verbose = verbose
        self.log_file = log_file
        self.log_buffer = []
        
        # Log initialization
        self.log("="*80)
        self.log("NEURAL DIARIZER INITIALIZATION - NeMo ClusteringDiarizer")
        self.log("="*80)
        self.log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Device: {device}")
        self.log(f"Output Directory: {self.output_dir}")
        self.log(f"Max Speakers: {max_num_speakers}")
        if oracle_num_speakers:
            self.log(f"Oracle Speakers: {oracle_num_speakers}")
        self.log("")
        
        # Load or create config
        if config_path and os.path.exists(config_path):
            self.config = OmegaConf.load(config_path)
            self.log(f"📋 Loaded config from: {config_path}")
        else:
            self.config = self._create_default_config()
            self.log("📋 Using default configuration")
        
        # Override config parameters
        self.config.device = device
        self.config.diarizer.out_dir = str(self.output_dir)
        self.config.verbose = verbose
        
        # Set clustering parameters
        if oracle_num_speakers:
            self.config.diarizer.clustering.parameters.oracle_num_speakers = True
            self.config.diarizer.oracle_num_speakers = oracle_num_speakers
        else:
            self.config.diarizer.clustering.parameters.oracle_num_speakers = False
            self.config.diarizer.clustering.parameters.max_num_speakers = max_num_speakers
        
        # Log configuration details
        self.log("\n>>> Configuration Details")
        self.log("-" * 60)
        self.log(f"VAD Model: {self.config.diarizer.vad.model_path}")
        self.log(f"  Window: {self.config.diarizer.vad.parameters.window_length_in_sec}s")
        self.log(f"  Shift: {self.config.diarizer.vad.parameters.shift_length_in_sec}s")
        self.log(f"  Onset: {self.config.diarizer.vad.parameters.onset}")
        self.log(f"  Offset: {self.config.diarizer.vad.parameters.offset}")
        self.log(f"  Min Duration ON: {self.config.diarizer.vad.parameters.min_duration_on}s")
        self.log(f"  Min Duration OFF: {self.config.diarizer.vad.parameters.min_duration_off}s")
        self.log("")
        self.log(f"Speaker Embedding Model: {self.config.diarizer.speaker_embeddings.model_path}")
        self.log(f"  Window: {self.config.diarizer.speaker_embeddings.parameters.window_length_in_sec}s")
        self.log(f"  Shift: {self.config.diarizer.speaker_embeddings.parameters.shift_length_in_sec}s")
        self.log("")
        self.log(f"Clustering Parameters:")
        self.log(f"  Max Speakers: {self.config.diarizer.clustering.parameters.max_num_speakers}")
        self.log(f"  Max RP Threshold: {self.config.diarizer.clustering.parameters.max_rp_threshold}")
        self.log(f"  Enhanced Count Threshold: {self.config.diarizer.clustering.parameters.enhanced_count_thres}")
        self.log("")
        
        # Initialize diarizer model (lazy loading)
        self.diarizer = None
        
        self.log("✅ Neural Diarizer initialized!")
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
            
            self.log(f"\n💾 Pipeline log saved to: {log_path}")
    
    def _create_default_config(self) -> OmegaConf:
        """Create default diarization configuration"""
        config = {
            'device': 'cpu',
            'num_workers': 0,
            'batch_size': 32,
            'sample_rate': 16000,
            'verbose': True,
            'diarizer': {
                'manifest_filepath': None,
                'out_dir': 'diar_output',
                'oracle_vad': False,
                'oracle_num_speakers': False,
                'collar': 0.25,
                'ignore_overlap': True,
                'vad': {
                    'model_path': 'vad_multilingual_marblenet',
                    'external_vad_manifest': None,
                    'parameters': {
                        'window_length_in_sec': 0.025,
                        'shift_length_in_sec': 0.01,
                        'smoothing': 'median',
                        'overlap': 0.5,
                        'normalize_audio': True,
                        'onset': 0.5,
                        'offset': 0.5,
                        'pad_onset': 0.05,
                        'pad_offset': 0.05,
                        'min_duration_on': 0.2,
                        'min_duration_off': 0.2,
                        'filter_speech_first': True
                    }
                },
                'speaker_embeddings': {
                    'model_path': 'titanet_large',
                    'parameters': {
                        'window_length_in_sec': 1.5,
                        'shift_length_in_sec': 0.75,
                        'multiscale_weights': [1.0],
                        'save_embeddings': False
                    }
                },
                'clustering': {
                    'parameters': {
                        'oracle_num_speakers': False,
                        'max_num_speakers': 8,
                        'enhanced_count_thres': 80,
                        'max_rp_threshold': 0.15,
                        'sparse_search_volume': 30,
                        'maj_vote_spk_count': False
                    }
                }
            }
        }
        return OmegaConf.create(config)
    
    def _ensure_diarizer_loaded(self):
        """Lazy load the diarizer model"""
        if self.diarizer is None:
            self.log(">>> Step 0: Loading NeMo Models")
            self.log("-" * 60)
            self.log("Loading NeMo ClusteringDiarizer models...")
            self.log("  - VAD Model: Loading...")
            self.log("  - Speaker Embedding Model: Loading...")
            self.diarizer = ClusteringDiarizer(cfg=self.config)
            self.log("✅ All models loaded successfully!")
            self.log("")
    
    def diarize_audio(
        self,
        audio_path: Union[str, Path],
        num_speakers: Optional[int] = None
    ) -> List[Dict]:
        """
        Perform speaker diarization on audio file with detailed logging
        
        Parameters
        ----------
        audio_path : str or Path
            Path to audio file (WAV format recommended)
        num_speakers : int, optional
            If provided, use this exact number of speakers (oracle mode)
        
        Returns
        -------
        List[Dict]
            List of diarization segments with format:
            [{"start": 0.0, "end": 2.5, "speaker": "speaker_0"}, ...]
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        self.log("="*80)
        self.log("STARTING NEURAL DIARIZATION PIPELINE")
        self.log("="*80)
        self.log(f"Audio File: {audio_path.name}")
        self.log(f"Full Path: {audio_path.absolute()}")
        self.log("")
        
        # Step 1: Audio Input
        self.log(">>> Step 1: Audio Input & Validation")
        self.log("-" * 60)
        audio_data, sample_rate = sf.read(str(audio_path))
        duration = len(audio_data) / sample_rate
        self.log(f"Sample Rate: {sample_rate} Hz")
        self.log(f"Duration: {duration:.2f} seconds")
        self.log(f"Total Samples: {len(audio_data)}")
        self.log(f"Audio Shape: {audio_data.shape}")
        self.log("✅ Audio loaded and validated")
        self.log("")
        
        # Override oracle_num_speakers if provided
        if num_speakers:
            self.config.diarizer.clustering.parameters.oracle_num_speakers = True
            self.config.diarizer.oracle_num_speakers = num_speakers
            self.log(f"🎯 Using Oracle Mode: {num_speakers} speakers")
            self.log("")
        
        # Step 2: Create manifest file
        self.log(">>> Step 2: Manifest Creation")
        self.log("-" * 60)
        manifest_path = self.output_dir / "input_manifest.json"
        self._create_manifest(audio_path, manifest_path, num_speakers)
        
        # Update config with manifest path
        self.config.diarizer.manifest_filepath = str(manifest_path)
        
        # Ensure model is loaded
        self._ensure_diarizer_loaded()
        
        # Step 3-6: Run NeMo diarization pipeline
        self.log(">>> Step 3-6: Running NeMo Diarization Pipeline")
        self.log("-" * 60)
        self.log("This includes:")
        self.log("  Step 3: Voice Activity Detection (VAD)")
        self.log("  Step 4: Speech Segmentation")
        self.log("  Step 5: Speaker Embeddings Extraction")
        self.log("  Step 6: Clustering (Affinity → Laplacian → Eigenvectors → K-means)")
        self.log("")
        self.log("Starting NeMo pipeline execution...")
        self.log("")
        
        # Capture NeMo's internal processing
        self.diarizer.diarize()
        
        self.log("")
        self.log("✅ NeMo pipeline completed!")
        self.log("")
        
        # Step 7: Parse results
        self.log(">>> Step 7: Post-processing & Results")
        self.log("-" * 60)
        results = self._parse_rttm_results(audio_path.stem)
        
        if len(results) == 0:
            self.log("⚠️  No segments found in RTTM output")
        else:
            speakers = set(seg['speaker'] for seg in results)
            self.log(f"Total Segments: {len(results)}")
            self.log(f"Number of Speakers Detected: {len(speakers)}")
            self.log(f"Speaker IDs: {sorted(speakers)}")
            self.log("")
            
            # Log segment statistics
            segment_durations = [seg['end'] - seg['start'] for seg in results]
            total_speech = sum(segment_durations)
            self.log(f"Speech Statistics:")
            self.log(f"  Total Speech Time: {total_speech:.2f}s")
            self.log(f"  Speech Ratio: {total_speech/duration*100:.1f}%")
            self.log(f"  Avg Segment Duration: {sum(segment_durations)/len(segment_durations):.2f}s")
            self.log(f"  Min Segment Duration: {min(segment_durations):.2f}s")
            self.log(f"  Max Segment Duration: {max(segment_durations):.2f}s")
            self.log("")
            
            # Log per-speaker statistics
            self.log("Per-Speaker Statistics:")
            speaker_stats = {}
            for seg in results:
                spk = seg['speaker']
                if spk not in speaker_stats:
                    speaker_stats[spk] = []
                speaker_stats[spk].append(seg['end'] - seg['start'])
            
            for spk in sorted(speaker_stats.keys()):
                durations = speaker_stats[spk]
                total_time = sum(durations)
                self.log(f"  {spk}:")
                self.log(f"    Segments: {len(durations)}")
                self.log(f"    Total Time: {total_time:.2f}s ({total_time/duration*100:.1f}% of audio)")
                self.log(f"    Avg Duration: {total_time/len(durations):.2f}s")
            self.log("")
            
            # Log first few segments
            self.log("First 10 segments:")
            for i, seg in enumerate(results[:10]):
                self.log(f"  [{seg['start']:6.2f}s - {seg['end']:6.2f}s] {seg['speaker']} ({seg['end']-seg['start']:.2f}s)")
            
            if len(results) > 10:
                self.log(f"  ... and {len(results) - 10} more segments")
        
        self.log("")
        self.log("="*80)
        self.log("PIPELINE COMPLETED SUCCESSFULLY")
        self.log("="*80)
        self.log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("")
        
        # Save log
        self.save_log()
        
        return results
    
    def diarize_batch(
        self,
        audio_paths: List[Union[str, Path]],
        num_speakers: Optional[int] = None
    ) -> Dict[str, List[Dict]]:
        """
        Perform speaker diarization on multiple audio files
        
        Parameters
        ----------
        audio_paths : List[str or Path]
            List of paths to audio files
        num_speakers : int, optional
            If provided, use this exact number of speakers for all files
        
        Returns
        -------
        Dict[str, List[Dict]]
            Dictionary mapping audio filename to diarization results
        """
        if self.verbose:
            print(f"\n🎤 Batch diarization: {len(audio_paths)} files")
        
        # Override oracle mode if num_speakers provided
        if num_speakers:
            self.config.diarizer.clustering.parameters.oracle_num_speakers = True
            self.config.diarizer.oracle_num_speakers = num_speakers
        
        # Create manifest for all files (after setting num_speakers)
        manifest_path = self.output_dir / "batch_manifest.json"
        self._create_batch_manifest(audio_paths, manifest_path, num_speakers)
        
        # Update config
        self.config.diarizer.manifest_filepath = str(manifest_path)
        
        # Ensure model is loaded
        self._ensure_diarizer_loaded()
        
        # Run diarization
        if self.verbose:
            print("🔄 Running batch diarization...")
        
        self.diarizer.diarize()
        
        if self.verbose:
            print("✅ Batch diarization completed!")
        
        # Parse results for each file
        results = {}
        for audio_path in audio_paths:
            audio_path = Path(audio_path)
            file_results = self._parse_rttm_results(audio_path.stem)
            results[audio_path.name] = file_results
        
        return results
    
    def _create_manifest(self, audio_path: Path, manifest_path: Path, num_speakers: Optional[int] = None):
        """Create NeMo manifest file for single audio"""
        # Get audio duration
        audio_data, sample_rate = sf.read(str(audio_path))
        duration = len(audio_data) / sample_rate
        
        manifest_entry = {
            "audio_filepath": str(audio_path.absolute()),
            "offset": 0,
            "duration": duration,
            "label": "infer",
            "text": "-",
            "num_speakers": num_speakers,  # Set to actual value or None
            "rttm_filepath": None,
            "uem_filepath": None
        }
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest_entry, f)
            f.write('\n')
        
        self.log(f"Manifest created: {manifest_path}")
        self.log(f"  Audio: {audio_path.absolute()}")
        self.log(f"  Duration: {duration:.2f}s")
        self.log(f"  Num Speakers: {num_speakers if num_speakers else 'auto-detect'}")
        self.log("✅ Manifest created successfully")
        self.log("")
    
    def _create_batch_manifest(self, audio_paths: List[Path], manifest_path: Path, num_speakers: Optional[int] = None):
        """Create NeMo manifest file for multiple audio files"""
        with open(manifest_path, 'w') as f:
            for audio_path in audio_paths:
                audio_path = Path(audio_path)
                audio_data, sample_rate = sf.read(str(audio_path))
                duration = len(audio_data) / sample_rate
                
                manifest_entry = {
                    "audio_filepath": str(audio_path.absolute()),
                    "offset": 0,
                    "duration": duration,
                    "label": "infer",
                    "text": "-",
                    "num_speakers": num_speakers,  # Set to actual value or None
                    "rttm_filepath": None,
                    "uem_filepath": None
                }
                
                json.dump(manifest_entry, f)
                f.write('\n')
    
    def _parse_rttm_results(self, audio_stem: str) -> List[Dict]:
        """
        Parse RTTM output file to extract diarization segments
        
        RTTM format:
        SPEAKER <file-id> 1 <start> <duration> <NA> <NA> <speaker-id> <NA> <NA>
        """
        rttm_path = self.output_dir / "pred_rttms" / f"{audio_stem}.rttm"
        
        self.log(f"Parsing RTTM file: {rttm_path}")
        
        if not rttm_path.exists():
            self.log(f"⚠️  RTTM file not found: {rttm_path}")
            return []
        
        segments = []
        with open(rttm_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8 and parts[0] == "SPEAKER":
                    start = float(parts[3])
                    duration = float(parts[4])
                    speaker_id = parts[7]
                    
                    segments.append({
                        "start": start,
                        "end": start + duration,
                        "speaker": speaker_id
                    })
        
        # Sort by start time
        segments.sort(key=lambda x: x['start'])
        
        self.log(f"✅ Parsed {len(segments)} segments from RTTM")
        
        return segments
    
    def save_results(self, results: List[Dict], output_path: Union[str, Path]):
        """Save diarization results to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"💾 Results saved to: {output_path}")
    
    def print_segments(self, segments: List[Dict]):
        """Pretty print diarization segments"""
        print("\n" + "="*60)
        print("DIARIZATION RESULTS")
        print("="*60)
        
        speakers = {}
        for seg in segments:
            speaker = seg['speaker']
            if speaker not in speakers:
                speakers[speaker] = []
            speakers[speaker].append(seg)
        
        print(f"\n👥 Number of speakers: {len(speakers)}")
        print(f"📊 Total segments: {len(segments)}")
        
        for speaker_id in sorted(speakers.keys()):
            speaker_segs = speakers[speaker_id]
            total_duration = sum(s['end'] - s['start'] for s in speaker_segs)
            print(f"\n{speaker_id}: {len(speaker_segs)} segments, {total_duration:.2f}s total")
            
            for seg in speaker_segs[:5]:  # Show first 5 segments
                print(f"  [{seg['start']:6.2f}s - {seg['end']:6.2f}s] ({seg['end']-seg['start']:.2f}s)")
            
            if len(speaker_segs) > 5:
                print(f"  ... and {len(speaker_segs) - 5} more segments")
        
        print("\n" + "="*60 + "\n")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Neural Speaker Diarization with NeMo")
    parser.add_argument("audio_path", type=str, help="Path to audio file")
    parser.add_argument("--config", type=str, default="diar_infer_config.yaml",
                        help="Path to diarization config")
    parser.add_argument("--output-dir", type=str, default="diar_output",
                        help="Output directory")
    parser.add_argument("--max-speakers", type=int, default=8,
                        help="Maximum number of speakers")
    parser.add_argument("--num-speakers", type=int, default=None,
                        help="Exact number of speakers (oracle mode)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--output-log", type=str, default=None,
                        help="Save detailed log to file")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Determine device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Default log file
    if args.output_log is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_name = Path(args.audio_path).stem
        args.output_log = f"neural_diarizer_log_{audio_name}_{timestamp}.txt"
    
    # Initialize diarizer
    config_path = args.config if os.path.exists(args.config) else None
    diarizer = NeuralDiarizer(
        config_path=config_path,
        output_dir=args.output_dir,
        device=device,
        max_num_speakers=args.max_speakers,
        oracle_num_speakers=args.num_speakers,
        log_file=args.output_log,
        verbose=True
    )
    
    # Run diarization
    results = diarizer.diarize_audio(
        args.audio_path,
        num_speakers=args.num_speakers
    )
    
    # Print results
    diarizer.print_segments(results)
    
    # Save to JSON if requested
    if args.output_json:
        diarizer.save_results(results, args.output_json)
    
    print(f"\n✅ Done!")
    print(f"   Results: {args.output_dir}")
    print(f"   Log: {args.output_log}")
    if args.output_json:
        print(f"   JSON: {args.output_json}")


if __name__ == "__main__":
    main()
