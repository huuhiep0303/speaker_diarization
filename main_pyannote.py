"""
Real-time Speaker Diarization with PyAnnote
Implementation of pyannote.audio pipeline for speaker diarization
Based on pyannote/speaker-diarization-3.1 model

Features:
- Speaker diarization using pyannote.audio
- Processes single audio files or batch processing
- Outputs RTTM format for evaluation
- Extracts speaker embeddings for verification
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

try:
    from pyannote.audio import Pipeline
    from pyannote.audio import Model
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
    from pyannote.core import Segment, Annotation
    import soundfile as sf
except ImportError as e:
    print("ERROR: Missing required packages. Please install:")
    print("  pip install pyannote.audio soundfile")
    print(f"\nOriginal error: {e}")
    sys.exit(1)


class PyAnnoteDiarization:
    """
    PyAnnote Speaker Diarization System
    
    Uses pyannote.audio pipeline for speaker diarization and embedding extraction.
    """
    
    def __init__(self, 
                 device: str = None,
                 token: Optional[str] = None,
                 num_speakers: Optional[int] = None,
                 min_speakers: Optional[int] = None,
                 max_speakers: Optional[int] = None):
        """
        Initialize PyAnnote Diarization System
        
        Parameters
        ----------
        device : str, optional
            Device to use ('cuda' or 'cpu'). Auto-detected if None.
        token : str, optional
            HuggingFace authentication token (required for pyannote models)
        num_speakers : int, optional
            Exact number of speakers (if known)
        min_speakers : int, optional
            Minimum number of speakers
        max_speakers : int, optional
            Maximum number of speakers
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.token = token
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        
        print(f"🚀 Initializing PyAnnote Speaker Diarization...")
        print(f"   Device: {self.device}")
        
        # Load diarization pipeline
        try:
            print("   Loading pyannote/speaker-diarization-3.1 pipeline...")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=self.token
            )
            
            # Move pipeline to device
            if self.device == "cuda":
                self.pipeline.to(torch.device("cuda"))
            
            print("✅ Diarization pipeline loaded successfully!")
        except Exception as e:
            print(f"✗ Error loading diarization pipeline: {e}")
            print("\nNote: You may need to:")
            print("  1. Accept pyannote/segmentation-3.0 user conditions")
            print("  2. Accept pyannote/speaker-diarization-3.1 user conditions")
            print("  3. Create access token at hf.co/settings/tokens")
            raise
        
        # Load embedding model for speaker verification
        try:
            print("   Loading speaker embedding model...")
            self.embedding_model = PretrainedSpeakerEmbedding(
                "pyannote/wespeaker-voxceleb-resnet34-LM",
                device=torch.device(self.device)
            )
            print("✅ Embedding model loaded successfully!")
        except Exception as e:
            print(f"⚠️  Warning: Could not load embedding model: {e}")
            print("   Speaker embedding extraction will not be available.")
            self.embedding_model = None
    
    def diarize_file(self, audio_path: str) -> Annotation:
        """
        Perform speaker diarization on an audio file
        
        Parameters
        ----------
        audio_path : str
            Path to audio file
        
        Returns
        -------
        Annotation
            PyAnnote Annotation object with speaker segments
        """
        print(f"\n🎤 Processing: {audio_path}")
        
        # Set up diarization parameters
        diarization_params = {}
        if self.num_speakers is not None:
            diarization_params["num_speakers"] = self.num_speakers
        elif self.min_speakers is not None or self.max_speakers is not None:
            if self.min_speakers is not None:
                diarization_params["min_speakers"] = self.min_speakers
            if self.max_speakers is not None:
                diarization_params["max_speakers"] = self.max_speakers
        
        # Run diarization
        try:
            # Load audio using soundfile to avoid AudioDecoder issues
            waveform, sample_rate = sf.read(audio_path)
            
            # Convert to mono if stereo
            if len(waveform.shape) > 1:
                waveform = waveform.mean(axis=1)
            
            # Convert to torch tensor and prepare audio dict
            waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
            
            # Create audio dictionary for pipeline
            audio_dict = {
                "waveform": waveform_tensor,
                "sample_rate": sample_rate
            }
            
            # Run diarization with audio dict
            diarization_output = self.pipeline(audio_dict, **diarization_params)
            
            # Extract Annotation from DiarizeOutput
            # DiarizeOutput has 'speaker_diarization' attribute containing the Annotation
            if hasattr(diarization_output, 'speaker_diarization'):
                diarization = diarization_output.speaker_diarization
            elif hasattr(diarization_output, 'itertracks'):
                # Already an Annotation
                diarization = diarization_output
            else:
                # Fallback - shouldn't happen
                diarization = diarization_output
            
            # Print results
            speakers = set()
            segment_count = 0
            for segment, track, speaker in diarization.itertracks(yield_label=True):
                speakers.add(speaker)
                segment_count += 1
            
            print(f"✓ Found {len(speakers)} speakers in {segment_count} segments")
            
            return diarization
        
        except Exception as e:
            print(f"✗ Error during diarization: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def extract_embedding(self, audio_path: str, segment: Optional[Segment] = None) -> np.ndarray:
        """
        Extract speaker embedding from audio file or segment
        
        Parameters
        ----------
        audio_path : str
            Path to audio file
        segment : Segment, optional
            Specific segment to extract embedding from. If None, uses full file.
        
        Returns
        -------
        np.ndarray
            Speaker embedding vector (normalized)
        """
        if self.embedding_model is None:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            # Load audio
            audio, sample_rate = sf.read(audio_path)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Extract segment if specified
            if segment is not None:
                start_sample = int(segment.start * sample_rate)
                end_sample = int(segment.end * sample_rate)
                audio = audio[start_sample:end_sample]
            
            # Convert to torch tensor
            waveform = torch.from_numpy(audio).float().unsqueeze(0)
            
            # Move to device
            if self.device == "cuda":
                waveform = waveform.cuda()
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.embedding_model(waveform)
                embedding = embedding.squeeze().cpu().numpy()
            
            # Normalize
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            return embedding
        
        except Exception as e:
            print(f"✗ Error extracting embedding: {e}")
            return None
    
    def save_rttm(self, diarization, output_path: str, audio_uri: str = "audio"):
        """
        Save diarization results to RTTM format
        
        Parameters
        ----------
        diarization : Annotation or DiarizeOutput
            PyAnnote Annotation object or DiarizeOutput
        output_path : str
            Path to output RTTM file
        audio_uri : str
            URI/filename to use in RTTM file
        """
        try:
            # Extract Annotation if it's DiarizeOutput
            if hasattr(diarization, 'speaker_diarization'):
                annotation = diarization.speaker_diarization
            elif hasattr(diarization, 'itertracks'):
                annotation = diarization
            else:
                annotation = diarization
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                for segment, track, speaker in annotation.itertracks(yield_label=True):
                    # RTTM format: SPEAKER <file-id> 1 <start> <duration> <NA> <NA> <speaker-id> <NA> <NA>
                    f.write(f"SPEAKER {audio_uri} 1 {segment.start:.3f} {segment.duration:.3f} "
                           f"<NA> <NA> {speaker} <NA> <NA>\n")
            
            print(f"✓ Saved RTTM to: {output_path}")
        
        except Exception as e:
            print(f"✗ Error saving RTTM: {e}")
    
    def save_json(self, diarization, output_path: str, audio_path: str = None):
        """
        Save diarization results to JSON format
        
        Parameters
        ----------
        diarization : Annotation or DiarizeOutput
            PyAnnote Annotation object or DiarizeOutput
        output_path : str
            Path to output JSON file
        audio_path : str, optional
            Original audio file path
        """
        try:
            # Extract Annotation if it's DiarizeOutput
            if hasattr(diarization, 'speaker_diarization'):
                annotation = diarization.speaker_diarization
            elif hasattr(diarization, 'itertracks'):
                annotation = diarization
            else:
                annotation = diarization
            
            # Build result structure
            segments = []
            for segment, track, speaker in annotation.itertracks(yield_label=True):
                segments.append({
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "duration": float(segment.duration),
                    "speaker": speaker
                })
            
            # Get speaker statistics
            speakers = {}
            for seg in segments:
                spk = seg["speaker"]
                if spk not in speakers:
                    speakers[spk] = {"count": 0, "total_duration": 0.0}
                speakers[spk]["count"] += 1
                speakers[spk]["total_duration"] += seg["duration"]
            
            result = {
                "audio_file": audio_path if audio_path else "unknown",
                "timestamp": datetime.now().isoformat(),
                "num_speakers": len(speakers),
                "speakers": speakers,
                "segments": segments,
                "total_segments": len(segments)
            }
            
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Saved JSON to: {output_path}")
        
        except Exception as e:
            print(f"✗ Error saving JSON: {e}")
    
    def print_results(self, diarization):
        """
        Print diarization results to console
        
        Parameters
        ----------
        diarization : Annotation or DiarizeOutput
            PyAnnote Annotation object or DiarizeOutput
        """
        # Extract Annotation if it's DiarizeOutput
        if hasattr(diarization, 'speaker_diarization'):
            annotation = diarization.speaker_diarization
        elif hasattr(diarization, 'itertracks'):
            annotation = diarization
        else:
            annotation = diarization
        
        print("\n" + "="*70)
        print("Speaker Diarization Results:")
        print("="*70)
        
        for segment, track, speaker in annotation.itertracks(yield_label=True):
            print(f"{segment.start:>7.2f}s - {segment.end:>7.2f}s : {speaker}")
        
        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="PyAnnote Speaker Diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file diarization
  python main_pyannote.py --audio_file path/to/audio.wav
  
  # Specify number of speakers
  python main_pyannote.py --audio_file audio.wav --num_speakers 2
  
  # Save to RTTM format
  python main_pyannote.py --audio_file audio.wav --output_rttm output.rttm
  
  # Use HuggingFace token
  python main_pyannote.py --audio_file audio.wav --token YOUR_HF_TOKEN
        """
    )
    
    parser.add_argument("--audio_file", type=str, required=True,
                       help="Path to audio file for diarization")
    parser.add_argument("--output_json", type=str, default=None,
                       help="Output JSON file path (default: auto-generated)")
    parser.add_argument("--output_rttm", type=str, default=None,
                       help="Output RTTM file path (optional)")
    parser.add_argument("--num_speakers", type=int, default=None,
                       help="Exact number of speakers (if known)")
    parser.add_argument("--min_speakers", type=int, default=None,
                       help="Minimum number of speakers")
    parser.add_argument("--max_speakers", type=int, default=None,
                       help="Maximum number of speakers")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu, default: auto-detect)")
    parser.add_argument("--token", type=str, default=None,
                       help="HuggingFace authentication token")
    
    args = parser.parse_args()
    
    # Validate audio file
    if not os.path.exists(args.audio_file):
        print(f"✗ Error: Audio file not found: {args.audio_file}")
        return
    
    # Initialize diarization system
    try:
        diarizer = PyAnnoteDiarization(
            device=args.device,
            token=args.token,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers
        )
    except Exception as e:
        print(f"\n✗ Failed to initialize PyAnnote: {e}")
        return
    
    # Run diarization
    try:
        diarization = diarizer.diarize_file(args.audio_file)
    except Exception as e:
        print(f"\n✗ Diarization failed: {e}")
        return
    
    # Print results
    diarizer.print_results(diarization)
    
    # Save results
    audio_basename = Path(args.audio_file).stem
    
    # Save JSON
    if args.output_json:
        json_path = args.output_json
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = f"pyannote_output_{audio_basename}_{timestamp}.json"
    
    diarizer.save_json(diarization, json_path, args.audio_file)
    
    # Save RTTM if requested
    if args.output_rttm:
        diarizer.save_rttm(diarization, args.output_rttm, audio_basename)
    
    print("\n✅ Diarization completed successfully!")


if __name__ == "__main__":
    main()
