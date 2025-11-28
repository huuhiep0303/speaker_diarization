"""
Real-time Speaker Diarization with Whisper Transcription
Improved version with:
- VAD (Silero) for silence detection
- Dual-buffer strategy (partial + full transcription)
- faster-whisper for better performance
- SpeechBrain for speaker recognition
"""

import os
import queue
import threading
import json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import sounddevice as sd
import torch
import sys

# Add repo path to import utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'repo', 'realtime-transcript', 'backend'))

# Fix torchaudio backend issue with speechbrain
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='speechbrain')

# Patch torchaudio to fix speechbrain compatibility
try:
    import torchaudio
    if not hasattr(torchaudio, 'list_audio_backends'):
        # Monkey patch for newer torchaudio versions
        def _dummy_list_audio_backends():
            return ['soundfile']
        torchaudio.list_audio_backends = _dummy_list_audio_backends
except Exception:
    pass

# Patch SpeechBrain to fix huggingface_hub compatibility
try:
    import huggingface_hub
    
    # Check if hf_hub_download uses 'token' instead of 'use_auth_token'
    import inspect
    hf_sig = inspect.signature(huggingface_hub.hf_hub_download)
    if 'token' in hf_sig.parameters and 'use_auth_token' not in hf_sig.parameters:
        # Patch huggingface_hub.hf_hub_download to accept use_auth_token
        _original_hf_download = huggingface_hub.hf_hub_download
        
        def _patched_hf_download(*args, use_auth_token=None, token=None, **kwargs):
            # Convert use_auth_token to token
            if token is None and use_auth_token is not None and use_auth_token is not False:
                token = use_auth_token
            return _original_hf_download(*args, token=token, **kwargs)
        
        huggingface_hub.hf_hub_download = _patched_hf_download
        print("✓ Applied huggingface_hub compatibility patch")
except Exception as e:
    print(f"Warning: Could not patch huggingface_hub: {e}")
    pass

try:
    from faster_whisper import WhisperModel
    from silero_vad import VadOptions, get_speech_timestamps, collect_chunks
    from speechbrain.inference import EncoderClassifier
except ImportError as e:
    print("ERROR: Missing required packages. Please install:")
    print("  pip install faster-whisper speechbrain torch torchaudio soundfile")
    print(f"\nOriginal error: {e}")
    sys.exit(1)

# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
PARTIAL_CHUNK_DURATION = 1.0  # seconds - partial transcription every 1s
MIN_CHUNK_DURATION = 1.0  # seconds - minimum audio to process
MAX_CHUNK_DURATION = 10.0  # seconds - maximum before forced processing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
WHISPER_MODEL = "small"  # Options: tiny, base, small, medium, large-v3

# VAD settings
VAD_MIN_SILENCE_MS = 750  # 750ms silence triggers end of speech
VAD_SPEECH_PAD_MS = 300  # Pad speech with 300ms on each side

# Speaker detection settings
SPEAKER_THRESHOLD = 0.5  # Cosine similarity threshold for same speaker
UPDATE_ALPHA = 0.3  # EMA coefficient for speaker embedding updates

# Global queue for audio data
audio_queue = queue.Queue()
stop_recording = threading.Event()


class RealtimeDiarization:
    """Real-time speaker diarization and transcription system"""
    
    def __init__(self):
        print("Initializing models...")
        print(f"Using device: {DEVICE}")
        print(f"Compute type: {COMPUTE_TYPE}")
        
        # Load faster-whisper model
        print(f"Loading Whisper model: {WHISPER_MODEL}")
        self.whisper_model = WhisperModel(
            WHISPER_MODEL,
            device=DEVICE,
            compute_type=COMPUTE_TYPE
        )
        
        # Load SpeechBrain speaker recognition model
        print("Loading SpeechBrain speaker recognition...")
        
        # Check if model files exist locally
        local_model_dir = Path("pretrained_models/spkrec-ecapa-voxceleb")
        required_files = ["embedding_model.ckpt", "hyperparams.yaml"]
        has_local_model = all((local_model_dir / f).exists() for f in required_files)
        
        try:
            if has_local_model:
                # Load from local directory (preferred)
                print(f"  Loading from local directory: {local_model_dir}")
                self.speaker_model = EncoderClassifier.from_hparams(
                    source=str(local_model_dir),
                    run_opts={"device": DEVICE}
                )
                print(f"  ✓ Speaker model loaded successfully!")
            else:
                # Try downloading from HuggingFace
                print(f"  Downloading from HuggingFace (first time only)...")
                self.speaker_model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": DEVICE},
                    savedir=str(local_model_dir)
                )
                print(f"  ✓ Speaker model downloaded and loaded!")
        except Exception as e:
            error_msg = str(e)
            if "custom.py" in error_msg or "404" in error_msg:
                print(f"  ✗ HuggingFace download failed (404 error)")
                print(f"  Run: python download_speaker_model.py")
            else:
                print(f"  ✗ Error loading speaker model: {e}")
            print(f"  → Continuing without speaker detection...")
            self.speaker_model = None
        
        print("Models loaded successfully!\n")
        
        # Audio buffers
        self.recv_buffer = []  # ~1s buffer for partial transcription
        self.full_buffer = []  # Full speech segment buffer
        self.full_buffer_start_time = 0.0
        
        # State tracking
        self.processed_time = 0.0
        self.all_results = []
        self.start_time = None
        
        # Speaker tracking
        self.registered_speakers = []  # List of embeddings
        self.speaker_counts = []  # Count of appearances per speaker
        
        # VAD options
        self.vad_options = VadOptions(
            min_silence_duration_ms=VAD_MIN_SILENCE_MS,
            speech_pad_ms=VAD_SPEECH_PAD_MS
        )
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback function for sounddevice to capture audio"""
        if status:
            print(f"Status: {status}")
        
        # Add audio data to queue
        audio_queue.put(indata.copy())
    
    def start_recording(self):
        """Start recording audio from microphone"""
        
        # List available microphones
        print("\nAvailable audio devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  [{i}] {device['name']} (inputs: {device['max_input_channels']})")
        
        default_device = sd.query_devices(kind='input')
        print(f"\nUsing default input device: {default_device['name']}")
        print(f"Make sure your microphone volume is turned up!\n")
        
        print("Starting audio recording...")
        print(f"Sample rate: {SAMPLE_RATE} Hz")
        print(f"Partial transcription every ~{PARTIAL_CHUNK_DURATION}s")
        print(f"Full transcription on speech end (silence > {VAD_MIN_SILENCE_MS}ms)")
        print("-" * 80)
        
        # Set real start time
        self.start_time = datetime.now()
        
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            callback=self.audio_callback,
            dtype='float32',
            blocksize=4096
        ):
            print("Recording... (Press Ctrl+C to stop)\n")
            
            while not stop_recording.is_set():
                try:
                    # Get audio data from queue
                    audio_chunk = audio_queue.get(timeout=0.1)
                    self.recv_buffer.extend(audio_chunk.flatten())
                    self.full_buffer.extend(audio_chunk.flatten())
                    
                    current_samples = len(self.recv_buffer)
                    partial_samples = int(PARTIAL_CHUNK_DURATION * SAMPLE_RATE)
                    max_samples = int(MAX_CHUNK_DURATION * SAMPLE_RATE)
                    
                    # Process partial every ~1s OR if reached max duration
                    if current_samples >= partial_samples:
                        self.process_partial_chunk()
                    
                    # Force process if full_buffer too large
                    if len(self.full_buffer) >= max_samples:
                        print("[Max duration reached, processing full buffer]")
                        self.process_full_chunk()
                        
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    print("\nStopping recording...")
                    stop_recording.set()
                    break
    
    def get_speaker_id(self, audio_f32):
        """Get speaker ID using SpeechBrain embeddings"""
        # If speaker model not loaded, return default
        if self.speaker_model is None:
            return "spk_01"
        
        if audio_f32 is None or len(audio_f32) == 0:
            return "unknown"
        
        # Pad if too short (< 0.5s)
        min_audio_length = 8000  # 0.5s @ 16kHz
        if len(audio_f32) < min_audio_length:
            pad_length = min_audio_length - len(audio_f32)
            # Repeat audio instead of zero padding
            audio_f32 = np.concatenate([audio_f32, audio_f32[:pad_length]])
        
        try:
            # Extract embedding
            tensor = torch.tensor(audio_f32).unsqueeze(0)
            with torch.no_grad():
                emb = self.speaker_model.encode_batch(tensor).detach().cpu().numpy().mean(axis=1)[0]
            
            # Normalize
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            
            # If no speakers registered, create first one
            if not self.registered_speakers:
                self.registered_speakers.append(emb_norm)
                self.speaker_counts.append(1)
                return "spk_01"
            
            # Compare with existing speakers
            sims = [np.dot(emb_norm, spk_emb) for spk_emb in self.registered_speakers]
            max_sim = max(sims)
            idx = np.argmax(sims)
            
            # If similar enough, assign to existing speaker
            if max_sim > SPEAKER_THRESHOLD:
                # Update embedding with EMA
                self.registered_speakers[idx] = (
                    (1 - UPDATE_ALPHA) * self.registered_speakers[idx] + 
                    UPDATE_ALPHA * emb_norm
                )
                self.speaker_counts[idx] += 1
                return f"spk_{idx+1:02d}"
            
            # Otherwise, create new speaker
            self.registered_speakers.append(emb_norm)
            self.speaker_counts.append(1)
            return f"spk_{len(self.registered_speakers):02d}"
        
        except Exception as e:
            print(f"[Speaker detection error: {e}]")
            return "unknown"
    
    def process_partial_chunk(self):
        """Process partial chunk for quick feedback"""
        if len(self.recv_buffer) == 0:
            return
        
        try:
            audio_data = np.array(self.recv_buffer, dtype=np.float32)
            buffer_duration = len(audio_data) / SAMPLE_RATE
            
            # Check audio level for debugging
            rms = np.sqrt(np.mean(audio_data ** 2))
            max_val = np.abs(audio_data).max()
            
            # Print audio stats occasionally for debugging
            if int(self.processed_time) % 5 == 0:  # Every 5 seconds
                print(f"[Audio Level - RMS: {rms:.6f}, Max: {max_val:.6f}]")
            
            # Save full_buffer start time if this is the first chunk
            if len(self.full_buffer) == len(self.recv_buffer):
                self.full_buffer_start_time = self.processed_time
            
            # VAD check - if silence, trigger full transcription
            speech_chunks = get_speech_timestamps(audio_data, self.vad_options)
            
            if len(speech_chunks) == 0:
                # Only print occasionally to avoid spam
                if int(self.processed_time) % 10 == 0:
                    print("[No speech detected - waiting...]")
                self.process_full_chunk()
                self.recv_buffer = []
                self.processed_time += buffer_duration
                return
            
            # Check RMS
            rms = np.sqrt(np.mean(audio_data ** 2))
            if rms < 0.01:
                print(f"[Low RMS: {rms:.6f}, skipping]")
                self.recv_buffer = []
                self.processed_time += buffer_duration
                return
            
            # Get speaker ID from speech portions
            speech_audio = collect_chunks(audio_data, speech_chunks)
            speaker_id = self.get_speaker_id(speech_audio)
            
            # Transcribe with fast settings (partial)
            segments, info = self.whisper_model.transcribe(
                audio_data,
                language="vi",  # Specify Vietnamese for better accuracy
                vad_filter=False,  # Already did VAD
                beam_size=1,  # Fast
                best_of=1,
                condition_on_previous_text=False,
                temperature=0.0
            )
            
            # Process segments
            for seg in segments:
                text = seg.text.strip()
                if text:
                    seg_start = seg.start + self.processed_time
                    seg_end = seg.end + self.processed_time
                    real_time = self.start_time + timedelta(seconds=seg_start)
                    
                    print(f"[{real_time.strftime('%H:%M:%S.%f')[:-3]}] [PARTIAL] {speaker_id}: {text}")
            
            # Clear recv_buffer and update time
            self.recv_buffer = []
            self.processed_time += buffer_duration
            
        except Exception as e:
            print(f"[Error in partial transcription: {e}]")
            import traceback
            traceback.print_exc()
            self.recv_buffer = []
    
    def process_full_chunk(self):
        """Process full speech segment with high accuracy"""
        if len(self.full_buffer) == 0:
            return
        
        try:
            audio_data = np.array(self.full_buffer, dtype=np.float32)
            
            # VAD filter
            speech_chunks = get_speech_timestamps(audio_data, self.vad_options)
            if len(speech_chunks) == 0:
                print("[Full buffer is silent, skipping]")
                self.full_buffer = []
                return
            
            # Collect speech portions
            audio_data = collect_chunks(audio_data, speech_chunks)
            
            # Get speaker ID
            speaker_id = self.get_speaker_id(audio_data)
            
            # Transcribe with accurate settings (full)
            segments, info = self.whisper_model.transcribe(
                audio_data,
                language="vi",
                vad_filter=False,
                beam_size=5,  # Accurate
                best_of=5,
                condition_on_previous_text=True,  # Better context
                temperature=0.0
            )
            
            # Process segments and save
            full_texts = []
            for seg in segments:
                text = seg.text.strip()
                if text:
                    full_texts.append(text)
                    seg_start = seg.start + self.full_buffer_start_time
                    seg_end = seg.end + self.full_buffer_start_time
                    real_time = self.start_time + timedelta(seconds=seg_start)
                    
                    result = {
                        "timestamp": real_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                        "start": round(seg_start, 2),
                        "end": round(seg_end, 2),
                        "speaker": speaker_id,
                        "text": text,
                        "language": info.language,
                        "language_probability": round(info.language_probability, 2)
                    }
                    self.all_results.append(result)
                    
                    print(f"[{real_time.strftime('%H:%M:%S.%f')[:-3]}] [FULL] {speaker_id}: {text}")
            
            print(f"[Full transcription completed: {len(full_texts)} segments]\n")
            
            # Clear full_buffer
            self.full_buffer = []
            self.full_buffer_start_time = 0.0
            
        except Exception as e:
            print(f"[Error in full transcription: {e}]")
            import traceback
            traceback.print_exc()
            self.full_buffer = []


def main():
    """Main entry point"""
    
    print("="*80)
    print("Real-time Speaker Diarization (Improved Version)")
    print("="*80)
    print(f"Whisper Model: {WHISPER_MODEL}")
    print(f"Device: {DEVICE}")
    print(f"Compute Type: {COMPUTE_TYPE}")
    print(f"VAD Silence Threshold: {VAD_MIN_SILENCE_MS}ms")
    print("="*80)
    print()
    
    system = None
    try:
        # Initialize system
        system = RealtimeDiarization()
        
        # Start recording and processing
        system.start_recording()
        
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nShutting down...")
        
        # Save results to JSON file (always save, even if empty)
        if system:
            output_file = f"realtime_diarization_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "total_segments": len(system.all_results) if system.all_results else 0,
                        "total_duration": round(system.processed_time, 2) if system.processed_time else 0.0,
                        "segments": system.all_results if system.all_results else []
                    }, f, indent=2, ensure_ascii=False)
                print(f"\n✓ Results saved to: {output_file}")
                print(f"  Total segments: {len(system.all_results) if system.all_results else 0}")
                print(f"  Total duration: {system.processed_time:.2f}s" if system.processed_time else "  Total duration: 0.00s")
            except Exception as e:
                print(f"\n✗ Error saving results: {e}")
        else:
            print("\nSystem not initialized, no file saved.")


if __name__ == "__main__":
    main()
