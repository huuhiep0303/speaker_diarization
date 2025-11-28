"""
Audio/Video Speaker Diarization with Dual-Buffer Strategy
Hỗ trợ:
- File audio/video local (.wav, .mp3, .m4a, .flac, etc.)
- YouTube video URL
- Dual-buffer strategy: recv_buffer (~1s) + full_buffer (chính xác)
- VAD để detect speech segments
- Speaker diarization với SpeechBrain ECAPA-TDNN
- ASR với Whisper (faster-whisper) hoặc SenseVoice (funasr)
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import soundfile as sf
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

# Buffer settings
RECV_BUFFER_SEC = 1.0  # Partial messages (~1s chunks)
FULL_BUFFER_SEC = 10.0  # Full messages (chính xác hơn)

# VAD settings
VAD_THRESHOLD = 0.5
MIN_SILENCE_SEC = 0.8  # Silence to detect end of speech
MIN_SPEECH_SEC = 0.5   # Minimum speech duration

# Speaker settings
SIMILARITY_THRESHOLD = 0.65  # Threshold để match speaker
EMBEDDING_UPDATE_WEIGHT = 0.2  # EMA weight cho update embedding
MAX_SPEAKERS = 20

# ASR Model selection
USE_WHISPER = True  # True: faster-whisper, False: SenseVoice

# Output
OUTPUT_DIR = "."

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
    
    # Try to load from local cache first
    local_model_dir = Path("pretrained_models/spkrec-ecapa-voxceleb")
    
    if local_model_dir.exists() and (local_model_dir / "hyperparams.yaml").exists():
        print(f"  → Tìm thấy model local: {local_model_dir}")
        speaker_model = EncoderClassifier.from_hparams(
            source=str(local_model_dir),
            savedir=str(local_model_dir),
            run_opts={"device": DEVICE}
        )
    else:
        # Download from HuggingFace
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
    import traceback
    traceback.print_exc()
    speaker_model = None
    SPEAKER_AVAILABLE = False

# ==============================
# SPEAKER DIARIZATION CLASS
# ==============================
class SpeakerDiarization:
    """Speaker diarization with dual-tier matching (EMA + centroids)"""
    
    def __init__(self, model):
        self.model = model
        self.speaker_memory = {}  # speaker_id -> EMA embedding
        self.speaker_clusters = {}  # speaker_id -> list of embeddings
        self.speaker_counts = {}  # speaker_id -> count
        self.next_id = 0
    
    def extract_embedding(self, audio_f32):
        """Extract speaker embedding from audio"""
        # Ensure minimum length (0.5s)
        min_samples = int(0.5 * SAMPLE_RATE)
        if len(audio_f32) < min_samples:
            audio_f32 = np.pad(audio_f32, (0, min_samples - len(audio_f32)), mode='reflect')
        
        # Convert to tensor and extract embedding
        tensor = torch.tensor(audio_f32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = self.model.encode_batch(tensor).squeeze().cpu().numpy()
        
        # Normalize
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb
    
    def match_speaker(self, embedding, duration=0.0):
        """Match embedding to existing speaker or create new one"""
        # No speakers yet
        if not self.speaker_memory:
            self.next_id += 1
            speaker_id = f"spk_{self.next_id:02d}"
            self.speaker_memory[speaker_id] = embedding
            self.speaker_clusters[speaker_id] = [embedding]
            self.speaker_counts[speaker_id] = 1
            print(f"  ✓ Người nói đầu tiên: {speaker_id}")
            return speaker_id
        
        # Compute similarities with EMA embeddings
        ema_sims = {}
        for spk_id, ema_emb in self.speaker_memory.items():
            sim = 1 - cosine(embedding, ema_emb)
            ema_sims[spk_id] = sim
        
        # Compute similarities with cluster centroids
        centroid_sims = {}
        for spk_id, cluster in self.speaker_clusters.items():
            if cluster:
                centroid = np.mean(cluster, axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
                sim = 1 - cosine(embedding, centroid)
                centroid_sims[spk_id] = sim
            else:
                centroid_sims[spk_id] = 0.0
        
        # Find best matches
        best_ema_spk = max(ema_sims, key=ema_sims.get)
        best_ema_sim = ema_sims[best_ema_spk]
        
        best_centroid_spk = max(centroid_sims, key=centroid_sims.get)
        best_centroid_sim = centroid_sims[best_centroid_spk]
        
        # Decision: use higher similarity
        matched_speaker = None
        max_sim = max(best_ema_sim, best_centroid_sim)
        
        if max_sim >= SIMILARITY_THRESHOLD:
            # Match to existing speaker
            matched_speaker = best_ema_spk if best_ema_sim >= best_centroid_sim else best_centroid_spk
        else:
            # Create new speaker
            if len(self.speaker_memory) < MAX_SPEAKERS:
                self.next_id += 1
                matched_speaker = f"spk_{self.next_id:02d}"
                self.speaker_memory[matched_speaker] = embedding
                self.speaker_clusters[matched_speaker] = [embedding]
                self.speaker_counts[matched_speaker] = 0
                print(f"  ✓ Phát hiện người nói mới: {matched_speaker} (sim: EMA={best_ema_sim:.3f}, centroid={best_centroid_sim:.3f})")
            else:
                # Max speakers reached, assign to best match
                matched_speaker = best_ema_spk if best_ema_sim >= best_centroid_sim else best_centroid_spk
                print(f"  ⚠️  Max speakers reached, gán vào {matched_speaker}")
        
        # Update embeddings (EMA + cluster)
        if duration >= 1.0:  # Only update for longer segments
            old_ema = self.speaker_memory[matched_speaker]
            new_ema = (1 - EMBEDDING_UPDATE_WEIGHT) * old_ema + EMBEDDING_UPDATE_WEIGHT * embedding
            self.speaker_memory[matched_speaker] = new_ema
            
            # Add to cluster (keep last 30)
            self.speaker_clusters[matched_speaker].append(embedding)
            if len(self.speaker_clusters[matched_speaker]) > 30:
                self.speaker_clusters[matched_speaker] = self.speaker_clusters[matched_speaker][-30:]
        
        self.speaker_counts[matched_speaker] = self.speaker_counts.get(matched_speaker, 0) + 1
        return matched_speaker
    
    def get_stats(self):
        """Get speaker statistics"""
        return {
            "total_speakers": len(self.speaker_memory),
            "speaker_counts": self.speaker_counts
        }

# Initialize speaker diarization
speaker_diar = SpeakerDiarization(speaker_model) if SPEAKER_AVAILABLE else None

# ==============================
# ASR FUNCTIONS
# ==============================
def transcribe_with_whisper(audio_f32):
    """Transcribe audio with faster-whisper"""
    segments, info = asr_model.transcribe(
        audio_f32,
        beam_size=5,
        language="en",
        condition_on_previous_text=False
    )
    
    # Combine all segments
    text = " ".join([seg.text.strip() for seg in segments])
    return text.strip(), info.language

def transcribe_with_sensevoice(audio_f32):
    """Transcribe audio with SenseVoice"""
    # SenseVoice expects 16-bit PCM
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
    return "", "auto"

# ==============================
# AUDIO PROCESSING WITH DUAL-BUFFER
# ==============================
def detect_speech_segments(audio_data):
    """Detect speech segments using VAD"""
    if not VAD_AVAILABLE:
        # Fallback: treat entire audio as one segment
        return [(0, len(audio_data))]
    
    # Convert to tensor
    audio_tensor = torch.FloatTensor(audio_data)
    
    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=VAD_THRESHOLD,
        min_silence_duration_ms=int(MIN_SILENCE_SEC * 1000),
        min_speech_duration_ms=int(MIN_SPEECH_SEC * 1000),
        speech_pad_ms=300
    )
    
    if not speech_timestamps:
        return []
    
    # Convert to (start, end) tuples
    segments = [(ts['start'], ts['end']) for ts in speech_timestamps]
    return segments

def process_audio_with_dual_buffer(audio_data, source_name="audio"):
    """
    Process audio with dual-buffer strategy:
    - recv_buffer: Process in ~1s chunks for partial/quick results
    - full_buffer: Process complete speech segments for accurate results
    """
    print("\n🔄 Đang phát hiện speech segments với VAD...")
    speech_segments = detect_speech_segments(audio_data)
    
    if not speech_segments:
        print("  ⚠️  Không phát hiện speech segments nào!")
        return []
    
    print(f"  ✓ Phát hiện {len(speech_segments)} speech segments")
    
    results = []
    total_duration = len(audio_data) / SAMPLE_RATE
    total_processing_time = 0
    
    for idx, (start_sample, end_sample) in enumerate(speech_segments):
        start_time = start_sample / SAMPLE_RATE
        end_time = end_sample / SAMPLE_RATE
        duration = end_time - start_time
        
        progress = (end_time / total_duration) * 100
        print(f"\n  [{progress:.1f}%] Segment {idx+1}/{len(speech_segments)}: {start_time:.1f}s - {end_time:.1f}s ({duration:.1f}s)")
        
        # Extract speech segment
        speech_audio = audio_data[start_sample:end_sample]
        
        # === DUAL-BUFFER STRATEGY ===
        
        # 1. RECV_BUFFER: Quick processing in ~1s chunks (for partial results)
        recv_buffer_size = int(RECV_BUFFER_SEC * SAMPLE_RATE)
        partial_texts = []
        
        if duration > RECV_BUFFER_SEC * 2:
            print(f"    📨 recv_buffer: Processing {int(duration / RECV_BUFFER_SEC)} chunks...")
            num_recv_chunks = int(np.ceil(len(speech_audio) / recv_buffer_size))
            
            for chunk_idx in range(num_recv_chunks):
                chunk_start = chunk_idx * recv_buffer_size
                chunk_end = min(chunk_start + recv_buffer_size, len(speech_audio))
                chunk_audio = speech_audio[chunk_start:chunk_end]
                
                if len(chunk_audio) < SAMPLE_RATE * 0.3:  # Skip too short
                    continue
                
                # Quick transcription (partial)
                chunk_start_time = time.time()
                if USE_WHISPER and ASR_AVAILABLE:
                    text, _ = transcribe_with_whisper(chunk_audio)
                elif ASR_AVAILABLE:
                    text, _ = transcribe_with_sensevoice(chunk_audio)
                else:
                    text = ""
                chunk_time = time.time() - chunk_start_time
                
                if text:
                    partial_texts.append(text)
                    print(f"      • Chunk {chunk_idx+1}: \"{text[:50]}...\" (RTF: {chunk_time/(len(chunk_audio)/SAMPLE_RATE):.3f})")
        
        # 2. FULL_BUFFER: Process complete segment (accurate)
        print(f"    📦 full_buffer: Processing complete segment...")
        full_start_time = time.time()
        
        # Transcribe full segment
        if USE_WHISPER and ASR_AVAILABLE:
            full_text, language = transcribe_with_whisper(speech_audio)
        elif ASR_AVAILABLE:
            full_text, language = transcribe_with_sensevoice(speech_audio)
        else:
            full_text, language = "", "unknown"
        
        full_time = time.time() - full_start_time
        total_processing_time += full_time
        rtf = full_time / duration
        
        # Speaker diarization
        speaker_id = "spk_01"
        if speaker_diar and SPEAKER_AVAILABLE:
            try:
                embedding = speaker_diar.extract_embedding(speech_audio)
                speaker_id = speaker_diar.match_speaker(embedding, duration)
            except Exception as e:
                print(f"    ⚠️  Lỗi speaker diarization: {e}")
        
        # Detect end of speech (repeated text or silence)
        end_of_speech = (idx == len(speech_segments) - 1)  # Last segment
        
        result = {
            "start": round(start_time, 2),
            "end": round(end_time, 2),
            "duration": round(duration, 2),
            "text": full_text,
            "speaker": speaker_id,
            "language": language,
            "rtf": round(rtf, 3),
            "partial_texts": partial_texts if partial_texts else None,
            "end_of_speech": end_of_speech
        }
        
        results.append(result)
        print(f"    ✓ [{speaker_id}] \"{full_text[:80]}...\" (RTF: {rtf:.3f})")
    
    # Calculate overall RTF
    overall_rtf = total_processing_time / total_duration
    print(f"\n✓ Hoàn thành xử lý {len(results)} segments")
    print(f"  📊 Overall RTF: {overall_rtf:.3f} (tổng: {total_processing_time:.1f}s / {total_duration:.1f}s)")
    
    return results, overall_rtf

# ==============================
# YOUTUBE DOWNLOADER
# ==============================
def download_youtube_audio(url, output_path="temp_youtube_audio.wav"):
    """Download audio from YouTube URL"""
    try:
        import yt_dlp
        
        print(f"📥 Đang tải audio từ YouTube...")
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'outtmpl': output_path.replace('.wav', ''),
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'Unknown')
            print(f"✓ Đã tải: {title}")
        
        return output_path, title
    except ImportError:
        print("❌ Cần cài đặt yt-dlp: pip install yt-dlp")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Lỗi tải YouTube: {e}")
        sys.exit(1)

# ==============================
# AUDIO FILE LOADER
# ==============================
def load_audio_file(file_path):
    """Load audio file and convert to 16kHz mono float32"""
    print(f"📂 Đang đọc file: {file_path}")
    try:
        audio_data, sr = sf.read(file_path, dtype='float32')
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Resample to 16kHz if needed
        if sr != SAMPLE_RATE:
            print(f"🔄 Đang chuyển đổi từ {sr}Hz sang {SAMPLE_RATE}Hz...")
            import scipy.signal
            audio_data = scipy.signal.resample(
                audio_data,
                int(len(audio_data) * SAMPLE_RATE / sr)
            )
        
        duration = len(audio_data) / SAMPLE_RATE
        print(f"✓ Đã đọc: {duration:.1f}s")
        return audio_data
        
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}")
        sys.exit(1)

# ==============================
# MAIN
# ==============================
def main():
    print("\n" + "="*70)
    print("🎯 Speaker Diarization with Dual-Buffer Strategy")
    print("="*70)
    
    # Check dependencies
    if not ASR_AVAILABLE:
        print("❌ ASR model không khả dụng!")
        return
    
    if not VAD_AVAILABLE:
        print("⚠️  VAD không khả dụng, kết quả có thể kém chính xác")
    
    # Select input source
    print("\n📌 Chọn nguồn input:")
    print("  1. File audio/video local")
    print("  2. YouTube URL")
    
    choice = input("\nNhập lựa chọn (1 hoặc 2): ").strip()
    
    audio_file = None
    source_name = ""
    cleanup_temp = False
    
    if choice == "2":
        url = input("Nhập YouTube URL: ").strip()
        audio_file, source_name = download_youtube_audio(url)
        cleanup_temp = True
    else:
        audio_file = input("Nhập đường dẫn file audio/video: ").strip()
        source_name = Path(audio_file).stem
    
    # Generate output filename
    safe_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '' for c in source_name)
    safe_name = safe_name.replace(' ', ' ')[:50]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"test_{safe_name}_{timestamp}.json"
    output_path = os.path.join(OUTPUT_DIR, output_file)
    
    print(f"\n📝 Kết quả sẽ lưu vào: {output_path}")
    print("="*70)
    
    try:
        # Load audio
        audio_data = load_audio_file(audio_file)
        
        # Process with dual-buffer
        segments, overall_rtf = process_audio_with_dual_buffer(audio_data, source_name)
        
        # Prepare output
        output_data = {
            "source": source_name,
            "start_time": datetime.now().isoformat(),
            "config": {
                "asr_model": "faster-whisper-base" if USE_WHISPER else "SenseVoiceSmall",
                "speaker_model": "speechbrain/spkrec-ecapa-voxceleb" if SPEAKER_AVAILABLE else "none",
                "sample_rate": SAMPLE_RATE,
                "device": DEVICE,
                "recv_buffer_sec": RECV_BUFFER_SEC,
                "full_buffer_sec": FULL_BUFFER_SEC
            },
            "overall_rtf": round(overall_rtf, 3),
            "total_segments": len(segments),
            "total_duration": round(len(audio_data) / SAMPLE_RATE, 2),
            "segments": segments
        }
        
        # Add speaker stats
        if speaker_diar:
            stats = speaker_diar.get_stats()
            output_data["speaker_stats"] = {
                spk: {
                    "count": count,
                    "duration": round(sum(seg["duration"] for seg in segments if seg["speaker"] == spk), 2)
                }
                for spk, count in stats["speaker_counts"].items()
            }
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        # Print summary
        print("\n" + "="*70)
        print(f"✅ Đã lưu kết quả: {output_path}")
        print(f"   📊 Tổng số segments: {len(segments)}")
        print(f"   ⏱️  Tổng thời lượng: {output_data['total_duration']}s")
        print(f"   ⚡ Overall RTF: {overall_rtf:.3f}")
        
        if speaker_diar and "speaker_stats" in output_data:
            print(f"   👥 Số người nói: {len(output_data['speaker_stats'])}")
            print(f"\n   📋 Chi tiết người nói:")
            for spk, stats in sorted(output_data['speaker_stats'].items()):
                print(f"      • {spk}: {stats['count']} segments, {stats['duration']}s")
        
        print("="*70 + "\n")
        
        # Print sample segments
        if segments:
            print("📝 Một số segments mẫu:")
            for seg in segments[:5]:
                print(f"   [{seg['start']}s - {seg['end']}s] {seg['speaker']}: {seg['text'][:80]}...")
            if len(segments) > 5:
                print(f"   ... và {len(segments) - 5} segments nữa")
    
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup temporary YouTube file
        if cleanup_temp and audio_file and os.path.exists(audio_file):
            try:
                os.remove(audio_file)
                print(f"\n🧹 Đã xóa file tạm: {audio_file}")
            except:
                pass
    
    print("\n👋 Đã hoàn thành!")

if __name__ == "__main__":
    main()
