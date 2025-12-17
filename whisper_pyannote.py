"""
Fusion Speaker Diarization: Whisper + PyAnnote
Kết hợp:
- Whisper (faster-whisper) cho ASR
- PyAnnote WeSpeaker-ResNet34 cho speaker embedding

Usage:
    python whisper_pyannote.py --audio_file path/to/audio.wav
"""

import os
import sys
import json
import argparse
import tempfile
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import soundfile as sf

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
    print(f"⚠️  Warning: Could not patch huggingface_hub: {e}")

try:
    from faster_whisper import WhisperModel
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
except ImportError as e:
    print("ERROR: Missing required packages. Please install:")
    print("  pip install faster-whisper pyannote.audio torch soundfile")
    print(f"\nOriginal error: {e}")
    sys.exit(1)

# Configuration
SAMPLE_RATE = 16000
WHISPER_MODEL = "small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# Speaker settings
SIMILARITY_THRESHOLD = 0.60
EMBEDDING_UPDATE_WEIGHT = 0.3
MAX_SPEAKERS = 10
MIN_DURATION_FOR_UPDATE = 2.0
MIN_AUDIO_LENGTH = 8000  # 0.5s @ 16kHz

# Output
OUTPUT_DIR = "."


class SpeakerManager:
    """Quản lý nhận diện người nói với PyAnnote embeddings"""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.speakers = []  # List of embeddings
        self.counts = []    # Count per speaker
        self.next_id = 0
    
    def get_embedding(self, audio_f32, sample_rate=16000):
        """Trích xuất embedding từ audio sử dụng PyAnnote"""
        if len(audio_f32) < MIN_AUDIO_LENGTH:
            # Pad by repeating audio
            pad_len = MIN_AUDIO_LENGTH - len(audio_f32)
            audio_f32 = np.concatenate([audio_f32, audio_f32[:pad_len]])
        
        # Convert to tensor: (batch, channel, samples)
        # PyAnnote expects 3D tensor with channel dimension
        waveform = torch.from_numpy(audio_f32).float().unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
        
        # Extract embedding - PyAnnote handles device internally
        with torch.no_grad():
            emb = self.embedding_model(waveform)
            if isinstance(emb, torch.Tensor):
                emb = emb.squeeze().cpu().numpy()
            else:
                emb = np.array(emb).squeeze()
        
        # Normalize
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb
    
    def identify(self, audio_f32, duration=0.0, sample_rate=16000):
        """Nhận diện hoặc đăng ký người nói mới"""
        try:
            emb = self.get_embedding(audio_f32, sample_rate)
            
            # Nếu chưa có ai
            if not self.speakers:
                self.speakers.append(emb)
                self.counts.append(1)
                self.next_id = 1
                return "spk_01"
            
            # So sánh với các speaker đã có
            sims = [np.dot(emb, spk) for spk in self.speakers]
            max_sim = max(sims)
            idx = np.argmax(sims)
            
            # Nếu đủ tương đồng
            if max_sim >= SIMILARITY_THRESHOLD:
                # Update embedding với EMA
                if duration >= MIN_DURATION_FOR_UPDATE:
                    self.speakers[idx] = (
                        (1 - EMBEDDING_UPDATE_WEIGHT) * self.speakers[idx] +
                        EMBEDDING_UPDATE_WEIGHT * emb
                    )
                self.counts[idx] += 1
                return f"spk_{idx+1:02d}"
            
            # Tạo speaker mới
            if len(self.speakers) < MAX_SPEAKERS:
                self.speakers.append(emb)
                self.counts.append(1)
                self.next_id += 1
                return f"spk_{self.next_id:02d}"
            
            # Nếu quá MAX_SPEAKERS, gán vào người tương đồng nhất
            self.counts[idx] += 1
            return f"spk_{idx+1:02d}"
            
        except Exception as e:
            print(f"⚠️  Error in speaker identification: {e}")
            return "spk_???"


class WhisperPyAnnoteDiarization:
    """Fusion system: Whisper ASR + PyAnnote Speaker Embedding"""
    
    def __init__(self):
        print(f"🔄 Initializing Whisper + PyAnnote Fusion System")
        print(f"   Device: {DEVICE}")
        
        # Load Whisper ASR
        print(f"📥 Loading Whisper {WHISPER_MODEL}...")
        self.whisper_model = WhisperModel(
            WHISPER_MODEL,
            device=DEVICE,
            compute_type=COMPUTE_TYPE
        )
        print("✓ Whisper loaded")
        
        # Load PyAnnote embedding model
        print("📥 Loading PyAnnote WeSpeaker-ResNet34...")
        self.embedding_model = PretrainedSpeakerEmbedding(
            "pyannote/wespeaker-voxceleb-resnet34-LM",
            device=torch.device(DEVICE)
        )
        print("✓ PyAnnote embedding model loaded")
        
        # Initialize speaker manager
        self.speaker_mgr = SpeakerManager(self.embedding_model)
        
        print("✅ Fusion system ready!")
    
    def process_audio(self, audio_path):
        """
        Process audio file: transcription + speaker diarization
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            dict: Results with segments containing text, speaker, timestamps
        """
        print(f"\n🎤 Processing: {audio_path}")
        
        # Load audio
        audio, sr = sf.read(audio_path)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Resample to 16kHz if needed
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE
        
        duration = len(audio) / sr
        print(f"   Duration: {duration:.2f}s")
        print(f"   Sample rate: {sr}Hz")
        
        # Transcription with Whisper
        print("\n📝 Running Whisper transcription...")
        segments, info = self.whisper_model.transcribe(
            audio_path,
            language=None,  # None for auto-detection
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 500,
            }
        )
        
        print(f"   Detected language: {info.language} (probability: {info.language_probability:.2f})")
        
        # Process segments
        results = {
            "audio_file": os.path.basename(audio_path),
            "timestamp": datetime.now().isoformat(),
            "model": {
                "asr": f"faster-whisper-{WHISPER_MODEL}",
                "speaker": "pyannote/wespeaker-voxceleb-resnet34-LM"
            },
            "device": DEVICE,
            "language": info.language,
            "segments": []
        }
        
        print("\n🔍 Processing segments with speaker diarization...")
        for segment in segments:
            start_time = segment.start
            end_time = segment.end
            text = segment.text.strip()
            
            if not text:
                continue
            
            # Extract audio segment
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = audio[start_sample:end_sample]
            
            # Speaker identification
            segment_duration = end_time - start_time
            speaker_id = self.speaker_mgr.identify(
                segment_audio,
                duration=segment_duration,
                sample_rate=sr
            )
            
            # Save segment result
            segment_data = {
                "start": round(start_time, 2),
                "end": round(end_time, 2),
                "duration": round(segment_duration, 2),
                "text": text,
                "speaker": speaker_id
            }
            results["segments"].append(segment_data)
            
            print(f"   [{start_time:.2f}s - {end_time:.2f}s] {speaker_id}: {text}")
        
        # Speaker statistics
        speaker_stats = {}
        for seg in results["segments"]:
            spk = seg["speaker"]
            if spk not in speaker_stats:
                speaker_stats[spk] = {"count": 0, "duration": 0.0, "text_length": 0}
            speaker_stats[spk]["count"] += 1
            speaker_stats[spk]["duration"] += seg["duration"]
            speaker_stats[spk]["text_length"] += len(seg["text"])
        
        results["speaker_stats"] = speaker_stats
        results["total_segments"] = len(results["segments"])
        results["total_speakers"] = len(speaker_stats)
        
        return results
    
    def save_json(self, results, output_path):
        """Save results to JSON file"""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n✅ Results saved to: {output_path}")
        except Exception as e:
            print(f"❌ Error saving JSON: {e}")
    
    def print_summary(self, results):
        """Print summary statistics"""
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Total segments: {results['total_segments']}")
        print(f"Total speakers: {results['total_speakers']}")
        print(f"Language: {results.get('language', 'unknown')}")
        
        if results.get("speaker_stats"):
            print("\nSpeaker Statistics:")
            for spk, stats in sorted(results["speaker_stats"].items()):
                print(f"  {spk}: {stats['count']} segments, "
                      f"{stats['duration']:.1f}s total, "
                      f"{stats['text_length']} chars")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Whisper + PyAnnote Fusion Diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single audio file
  python whisper_pyannote.py --audio_file audio.wav
  
  # Specify output file
  python whisper_pyannote.py --audio_file audio.wav --output result.json
        """
    )
    
    parser.add_argument("--audio_file", type=str, required=True,
                       help="Path to audio file")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Validate audio file
    if not os.path.exists(args.audio_file):
        print(f"✗ Error: Audio file not found: {args.audio_file}")
        return
    
    # Initialize system
    try:
        system = WhisperPyAnnoteDiarization()
    except Exception as e:
        print(f"\n✗ Failed to initialize system: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Process audio
    try:
        results = system.process_audio(args.audio_file)
    except Exception as e:
        print(f"\n✗ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print summary
    system.print_summary(results)
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        audio_basename = Path(args.audio_file).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"whisper_pyannote_{audio_basename}_{timestamp}.json"
    
    system.save_json(results, output_path)
    
    print("\n✅ Processing completed successfully!")


if __name__ == "__main__":
    main()
