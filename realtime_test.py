"""
Real-time Audio Capture & Speaker Diarization
Chương trình sẽ:
- Capture audio realtime từ system (loopback) hoặc microphone
- Detect speech với VAD
- Transcribe với Whisper/SenseVoice
- Speaker diarization với SpeechBrain
- Người dùng chỉ cần mở video YouTube, audio player, etc.
"""

import os
import sys
import json
import time
import queue
import threading
from datetime import datetime
from pathlib import Path
import numpy as np
import sounddevice as sd
import torch
from scipy.spatial.distance import cosine

# ==============================
# FIX HUGGINGFACE_HUB COMPATIBILITY
# ==============================
try:
    import huggingface_hub
    _original_hf_download = huggingface_hub.hf_hub_download
    
    def _patched_hf_download(*args, use_auth_token=None, token=None, **kwargs):
        """Convert use_auth_token to token for compatibility"""
        if token is None and use_auth_token is not None:
            token = use_auth_token
        return _original_hf_download(*args, token=token, **kwargs)
    
    huggingface_hub.hf_hub_download = _patched_hf_download
    print("✓ Đã patch huggingface_hub compatibility")
except Exception as e:
    print(f"⚠️  Không thể patch huggingface_hub: {e}")

# ==============================
# CONFIG
# ==============================
SAMPLE_RATE = 16000
DEVICE = "cpu"
CHANNELS = 1

# Audio capture settings
CHUNK_DURATION = 0.5  # Capture in 0.5s chunks
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

# Buffer settings
AUDIO_BUFFER_SEC = 30.0  # Keep last 30s of audio
MAX_BUFFER_SAMPLES = int(SAMPLE_RATE * AUDIO_BUFFER_SEC)

# VAD settings
VAD_THRESHOLD = 0.5
MIN_SILENCE_SEC = 1.0  # 1s silence = end of speech
MIN_SPEECH_SEC = 0.8   # Minimum 0.8s speech

# Speaker settings
SIMILARITY_THRESHOLD = 0.65
EMBEDDING_UPDATE_WEIGHT = 0.2
MAX_SPEAKERS = 20

# ASR Model selection
USE_WHISPER = True  # True: faster-whisper, False: SenseVoice

# Output
OUTPUT_DIR = "."
SESSION_START = datetime.now()

# Global state
audio_queue = queue.Queue()
audio_buffer = np.array([], dtype=np.float32)
is_running = False
all_segments = []

# ==============================
# LOAD VAD MODEL
# ==============================
print("🔄 Đang tải Silero VAD...")
try:
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    get_speech_timestamps, _, _, _, _ = vad_utils
    print("✓ Silero VAD đã sẵn sàng")
    VAD_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Không thể tải VAD: {e}")
    VAD_AVAILABLE = False
    vad_model = None

# ==============================
# LOAD ASR MODEL
# ==============================
print(f"🔄 Đang tải ASR model ({'Whisper' if USE_WHISPER else 'SenseVoice'})...")
if USE_WHISPER:
    try:
        from faster_whisper import WhisperModel
        asr_model = WhisperModel("base", device=DEVICE, compute_type="int8")
        print("✓ Faster-Whisper đã sẵn sàng")
        ASR_AVAILABLE = True
    except ImportError:
        print("⚠️  Cần cài đặt: pip install faster-whisper")
        ASR_AVAILABLE = False
        asr_model = None
    except Exception as e:
        print(f"⚠️  Không thể tải Whisper: {e}")
        ASR_AVAILABLE = False
        asr_model = None
else:
    try:
        from funasr import AutoModel
        asr_model = AutoModel(
            model="FunAudioLLM/SenseVoiceSmall",
            device=DEVICE,
            hub="hf",
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
        )
        print("✓ SenseVoice đã sẵn sàng")
        ASR_AVAILABLE = True
    except Exception as e:
        print(f"⚠️  Không thể tải SenseVoice: {e}")
        ASR_AVAILABLE = False
        asr_model = None

# ==============================
# LOAD SPEAKER MODEL
# ==============================
print("🔄 Đang tải SpeechBrain ECAPA-TDNN...")
try:
    from speechbrain.pretrained import EncoderClassifier
    
    local_model_dir = Path("pretrained_models/spkrec-ecapa-voxceleb")
    
    if local_model_dir.exists() and (local_model_dir / "hyperparams.yaml").exists():
        print(f"  → Tìm thấy model local: {local_model_dir}")
        speaker_model = EncoderClassifier.from_hparams(
            source=str(local_model_dir),
            savedir=str(local_model_dir),
            run_opts={"device": DEVICE}
        )
    else:
        print(f"  → Tải model từ HuggingFace...")
        speaker_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(local_model_dir),
            run_opts={"device": DEVICE}
        )
    
    print("✓ SpeechBrain đã sẵn sàng")
    SPEAKER_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Không thể tải SpeechBrain: {e}")
    print("→ Tiếp tục mà không có speaker diarization")
    speaker_model = None
    SPEAKER_AVAILABLE = False

# ==============================
# SPEAKER DIARIZATION CLASS
# ==============================
class SpeakerDiarization:
    """Speaker diarization with dual-tier matching"""
    
    def __init__(self, model):
        self.model = model
        self.speaker_memory = {}
        self.speaker_clusters = {}
        self.speaker_counts = {}
        self.next_id = 0
    
    def extract_embedding(self, audio_f32):
        """Extract speaker embedding from audio"""
        min_samples = int(0.5 * SAMPLE_RATE)
        if len(audio_f32) < min_samples:
            audio_f32 = np.pad(audio_f32, (0, min_samples - len(audio_f32)), mode='reflect')
        
        tensor = torch.tensor(audio_f32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = self.model.encode_batch(tensor).squeeze().cpu().numpy()
        
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb
    
    def match_speaker(self, embedding, duration=0.0):
        """Match embedding to existing speaker or create new one"""
        if not self.speaker_memory:
            self.next_id += 1
            speaker_id = f"spk_{self.next_id:02d}"
            self.speaker_memory[speaker_id] = embedding
            self.speaker_clusters[speaker_id] = [embedding]
            self.speaker_counts[speaker_id] = 1
            print(f"  ✓ Người nói đầu tiên: {speaker_id}")
            return speaker_id
        
        # Compute similarities
        ema_sims = {}
        for spk_id, ema_emb in self.speaker_memory.items():
            sim = 1 - cosine(embedding, ema_emb)
            ema_sims[spk_id] = sim
        
        centroid_sims = {}
        for spk_id, cluster in self.speaker_clusters.items():
            if cluster:
                centroid = np.mean(cluster, axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
                sim = 1 - cosine(embedding, centroid)
                centroid_sims[spk_id] = sim
            else:
                centroid_sims[spk_id] = 0.0
        
        best_ema_spk = max(ema_sims, key=ema_sims.get)
        best_ema_sim = ema_sims[best_ema_spk]
        
        best_centroid_spk = max(centroid_sims, key=centroid_sims.get)
        best_centroid_sim = centroid_sims[best_centroid_spk]
        
        matched_speaker = None
        max_sim = max(best_ema_sim, best_centroid_sim)
        
        if max_sim >= SIMILARITY_THRESHOLD:
            matched_speaker = best_ema_spk if best_ema_sim >= best_centroid_sim else best_centroid_spk
        else:
            if len(self.speaker_memory) < MAX_SPEAKERS:
                self.next_id += 1
                matched_speaker = f"spk_{self.next_id:02d}"
                self.speaker_memory[matched_speaker] = embedding
                self.speaker_clusters[matched_speaker] = [embedding]
                self.speaker_counts[matched_speaker] = 0
                print(f"  ✓ Phát hiện người nói mới: {matched_speaker} (sim: EMA={best_ema_sim:.3f}, centroid={best_centroid_sim:.3f})")
            else:
                matched_speaker = best_ema_spk if best_ema_sim >= best_centroid_sim else best_centroid_spk
        
        # Update embeddings
        if duration >= 1.0:
            old_ema = self.speaker_memory[matched_speaker]
            new_ema = (1 - EMBEDDING_UPDATE_WEIGHT) * old_ema + EMBEDDING_UPDATE_WEIGHT * embedding
            self.speaker_memory[matched_speaker] = new_ema
            
            self.speaker_clusters[matched_speaker].append(embedding)
            if len(self.speaker_clusters[matched_speaker]) > 30:
                self.speaker_clusters[matched_speaker] = self.speaker_clusters[matched_speaker][-30:]
        
        self.speaker_counts[matched_speaker] = self.speaker_counts.get(matched_speaker, 0) + 1
        return matched_speaker

# Initialize
speaker_diar = SpeakerDiarization(speaker_model) if SPEAKER_AVAILABLE else None

# ==============================
# ASR FUNCTIONS
# ==============================
def transcribe_audio(audio_f32):
    """Transcribe audio"""
    if USE_WHISPER and ASR_AVAILABLE:
        segments, info = asr_model.transcribe(
            audio_f32,
            beam_size=5,
            language="en",
            condition_on_previous_text=False
        )
        text = " ".join([seg.text.strip() for seg in segments])
        return text.strip(), info.language
    elif ASR_AVAILABLE:
        audio_int16 = (audio_f32 * 32768).astype(np.int16)
        result = asr_model.generate(
            input=audio_int16,
            cache={},
            language="auto",
            use_itn=True
        )
        if result and len(result) > 0:
            text = result[0].get("text", "")
            return text.strip(), "auto"
    return "", "unknown"

# ==============================
# AUDIO CALLBACK
# ==============================
def audio_callback(indata, frames, time_info, status):
    """Callback for audio capture"""
    if status:
        print(f"⚠️  Audio status: {status}")
    
    # Convert to mono if stereo
    if indata.shape[1] > 1:
        audio_data = indata.mean(axis=1)
    else:
        audio_data = indata[:, 0]
    
    # Add to queue
    audio_queue.put(audio_data.copy())

# ==============================
# SPEECH DETECTION & PROCESSING
# ==============================
def detect_and_process_speech():
    """Detect speech in buffer and process"""
    global audio_buffer, all_segments
    
    if len(audio_buffer) < int(SAMPLE_RATE * 1.0):  # Need at least 1s
        return
    
    # Detect speech segments with VAD
    audio_tensor = torch.FloatTensor(audio_buffer)
    
    try:
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            vad_model,
            sampling_rate=SAMPLE_RATE,
            threshold=VAD_THRESHOLD,
            min_silence_duration_ms=int(MIN_SILENCE_SEC * 1000),
            min_speech_duration_ms=int(MIN_SPEECH_SEC * 1000),
            speech_pad_ms=300
        )
    except Exception as e:
        print(f"⚠️  VAD error: {e}")
        return
    
    if not speech_timestamps:
        return
    
    # Process only complete speech segments (not at the end of buffer)
    buffer_end = len(audio_buffer)
    
    for ts in speech_timestamps:
        start_sample = ts['start']
        end_sample = ts['end']
        
        # Only process if segment is complete (not at buffer end)
        if end_sample < buffer_end - int(SAMPLE_RATE * MIN_SILENCE_SEC):
            duration = (end_sample - start_sample) / SAMPLE_RATE
            
            # Check if already processed (avoid duplicates)
            segment_time = (datetime.now() - SESSION_START).total_seconds() - (buffer_end - end_sample) / SAMPLE_RATE
            
            # Skip if too similar to last segment
            if all_segments and abs(all_segments[-1]['end'] - segment_time) < 0.5:
                continue
            
            print(f"\n🎤 Phát hiện speech: {duration:.1f}s")
            
            # Extract speech audio
            speech_audio = audio_buffer[start_sample:end_sample]
            
            # Transcribe
            process_start = time.time()
            text, language = transcribe_audio(speech_audio)
            process_time = time.time() - process_start
            rtf = process_time / duration
            
            # Speaker diarization
            speaker_id = "spk_01"
            if speaker_diar and SPEAKER_AVAILABLE:
                try:
                    embedding = speaker_diar.extract_embedding(speech_audio)
                    speaker_id = speaker_diar.match_speaker(embedding, duration)
                except Exception as e:
                    print(f"  ⚠️  Speaker error: {e}")
            
            # Calculate absolute time
            abs_start = segment_time - duration
            abs_end = segment_time
            
            segment = {
                "start": round(abs_start, 2),
                "end": round(abs_end, 2),
                "duration": round(duration, 2),
                "text": text,
                "speaker": speaker_id,
                "language": language,
                "rtf": round(rtf, 3),
                "timestamp": datetime.now().isoformat()
            }
            
            all_segments.append(segment)
            
            # Display
            print(f"  [{speaker_id}] {text[:100]}...")
            print(f"  ⏱️  RTF: {rtf:.3f}")
            
            # Remove processed audio from buffer
            audio_buffer = audio_buffer[end_sample:]

# ==============================
# WORKER THREAD
# ==============================
def worker():
    """Worker thread to process audio"""
    global audio_buffer, is_running
    
    print("\n🎧 Đang lắng nghe... (mở video YouTube hoặc audio bất kỳ)")
    print("   Nhấn Ctrl+C để dừng\n")
    
    last_process_time = time.time()
    
    while is_running:
        try:
            # Get audio from queue
            while not audio_queue.empty():
                chunk = audio_queue.get_nowait()
                audio_buffer = np.concatenate([audio_buffer, chunk])
                
                # Limit buffer size
                if len(audio_buffer) > MAX_BUFFER_SAMPLES:
                    audio_buffer = audio_buffer[-MAX_BUFFER_SAMPLES:]
            
            # Process periodically
            current_time = time.time()
            if current_time - last_process_time >= 1.0:  # Check every 1s
                if VAD_AVAILABLE:
                    detect_and_process_speech()
                last_process_time = current_time
            
            time.sleep(0.1)
            
        except Exception as e:
            print(f"⚠️  Worker error: {e}")
            import traceback
            traceback.print_exc()

# ==============================
# LIST AUDIO DEVICES
# ==============================
def list_audio_devices():
    """List available audio devices"""
    print("\n" + "="*70)
    print("🔊 Available Audio Devices:")
    print("="*70)
    
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        device_type = []
        if device['max_input_channels'] > 0:
            device_type.append("INPUT")
        if device['max_output_channels'] > 0:
            device_type.append("OUTPUT")
        
        print(f"{idx}: {device['name']}")
        print(f"   Type: {', '.join(device_type)}")
        print(f"   Channels: In={device['max_input_channels']}, Out={device['max_output_channels']}")
        print(f"   Sample Rate: {device['default_samplerate']}Hz")
        print()

# ==============================
# SAVE RESULTS
# ==============================
def save_results():
    """Save all segments to JSON"""
    if not all_segments:
        print("\n⚠️  Không có segments nào để lưu")
        return
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"realtime_capture_{timestamp}.json"
    output_path = os.path.join(OUTPUT_DIR, output_file)
    
    output_data = {
        "session_start": SESSION_START.isoformat(),
        "session_end": datetime.now().isoformat(),
        "config": {
            "asr_model": "faster-whisper-base" if USE_WHISPER else "SenseVoiceSmall",
            "speaker_model": "speechbrain/spkrec-ecapa-voxceleb" if SPEAKER_AVAILABLE else "none",
            "sample_rate": SAMPLE_RATE,
            "device": DEVICE
        },
        "total_segments": len(all_segments),
        "segments": all_segments
    }
    
    # Add speaker stats
    if speaker_diar:
        speaker_stats = {}
        for seg in all_segments:
            spk = seg['speaker']
            if spk not in speaker_stats:
                speaker_stats[spk] = {"count": 0, "duration": 0}
            speaker_stats[spk]["count"] += 1
            speaker_stats[spk]["duration"] += seg["duration"]
        
        output_data["speaker_stats"] = {
            spk: {
                "count": stats["count"],
                "duration": round(stats["duration"], 2)
            }
            for spk, stats in speaker_stats.items()
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Đã lưu kết quả: {output_path}")
    print(f"   📊 Tổng số segments: {len(all_segments)}")
    
    if "speaker_stats" in output_data:
        print(f"   👥 Số người nói: {len(output_data['speaker_stats'])}")
        for spk, stats in sorted(output_data['speaker_stats'].items()):
            print(f"      • {spk}: {stats['count']} segments, {stats['duration']}s")

# ==============================
# MAIN
# ==============================
def main():
    global is_running
    
    print("\n" + "="*70)
    print("🎯 Real-time Audio Capture & Speaker Diarization")
    print("="*70)
    
    if not ASR_AVAILABLE:
        print("❌ ASR model không khả dụng!")
        return
    
    if not VAD_AVAILABLE:
        print("⚠️  VAD không khả dụng!")
        return
    
    # List devices
    list_audio_devices()
    
    # Select device
    print("📌 Chọn audio device:")
    print("  - Nhập số device ID để chọn")
    print("  - Enter để dùng default device")
    print("  - Với Windows: Chọn 'Stereo Mix' hoặc 'WASAPI loopback' để capture system audio")
    
    device_input = input("\nNhập device ID (hoặc Enter): ").strip()
    
    if device_input:
        try:
            device_id = int(device_input)
        except ValueError:
            print("❌ Device ID không hợp lệ!")
            return
    else:
        device_id = None  # Use default
    
    print(f"\n🎙️  Sử dụng device: {device_id if device_id is not None else 'default'}")
    print("="*70)
    
    # Start worker thread
    is_running = True
    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()
    
    # Start audio stream
    try:
        with sd.InputStream(
            device=device_id,
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            callback=audio_callback
        ):
            print("\n✓ Đã bắt đầu capture audio")
            print("  → Mở video YouTube, audio player, hoặc bất kỳ nguồn âm thanh nào")
            print("  → Chương trình sẽ tự động detect speech và transcribe")
            print("  → Nhấn Ctrl+C để dừng\n")
            
            # Keep running until Ctrl+C
            while is_running:
                time.sleep(0.5)
                
    except KeyboardInterrupt:
        print("\n\n⏹️  Đang dừng...")
        is_running = False
        worker_thread.join(timeout=2)
        
        # Save results
        save_results()
        
        print("\n👋 Đã hoàn thành!")
        
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        is_running = False

if __name__ == "__main__":
    main()
