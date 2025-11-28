"""
SenseVoice + SpeechBrain Speaker Diarization for Audio/Video Files
Hỗ trợ:
- File audio local (.wav, .mp3, .m4a, .flac, etc.)
- YouTube video URL
Kết quả lưu vào file JSON với speaker diarization.
"""

import os
import sys
import tempfile
import json
from datetime import datetime
import numpy as np
import soundfile as sf
import torch
from scipy.spatial.distance import cdist
from pathlib import Path

# Fix huggingface_hub compatibility BEFORE importing speechbrain
try:
    import huggingface_hub
    _original_hf_download = huggingface_hub.hf_hub_download
    
    def _patched_hf_download(*args, use_auth_token=None, token=None, **kwargs):
        """Convert use_auth_token to token for compatibility"""
        if token is None and use_auth_token is not None:
            token = use_auth_token
        return _original_hf_download(*args, token=token, **kwargs)
    
    huggingface_hub.hf_hub_download = _patched_hf_download
    print("✓ Đã áp dụng patch tương thích huggingface_hub")
except Exception as e:
    print(f"⚠️  Cảnh báo: Không thể patch huggingface_hub: {e}")

# Import models
from speechbrain.pretrained import EncoderClassifier
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# ==============================
# CONFIG
# ==============================
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SEC = 3.0
OVERLAP_SEC = 0.3
LANGUAGE = "auto"
MIN_AUDIO_LENGTH = 8000  # 0.5s @ 16kHz
DEVICE = "cpu"

# Output
OUTPUT_DIR = "."
OUTPUT_FILE = f"senvoi_spebrai_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# Speaker settings
SIMILARITY_THRESHOLD = 0.6
EMBEDDING_UPDATE_WEIGHT = 0.3
MAX_SPEAKERS = 10
MIN_DURATION_FOR_UPDATE = 2.0

# Global results
all_results = {
    "start_time": datetime.now().isoformat(),
    "model": {
        "asr": "FunAudioLLM/SenseVoiceSmall",
        "speaker": "speechbrain/spkrec-ecapa-voxceleb"
    },
    "device": DEVICE,
    "sample_rate": SAMPLE_RATE,
    "segments": []
}

# ==============================
# LOAD MODELS
# ==============================
print("🔄 Đang tải model trên thiết bị:", DEVICE)

# Load SenseVoice
print("📥 Đang tải SenseVoiceSmall...")
asr_model = AutoModel(
    model="FunAudioLLM/SenseVoiceSmall",
    device=DEVICE,
    hub="hf",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
)
print("✓ SenseVoice đã sẵn sàng")

# Load SpeechBrain
print("📥 Đang tải SpeechBrain ECAPA-TDNN...")
try:
    speaker_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": DEVICE},
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )
    print("✓ SpeechBrain đã sẵn sàng")
    SPEAKER_ENABLED = True
except Exception as e:
    print(f"⚠️  Không thể tải SpeechBrain: {e}")
    print("→ Tiếp tục mà không có nhận diện người nói")
    speaker_model = None
    SPEAKER_ENABLED = False

# ==============================
# SPEAKER DIARIZATION
# ==============================
class SpeakerManager:
    """Quản lý nhận diện người nói"""
    
    def __init__(self, model):
        self.model = model
        self.speakers = []  # List of embeddings
        self.counts = []    # Count per speaker
        self.next_id = 0
    
    def get_embedding(self, audio_f32):
        """Trích xuất embedding từ audio"""
        if len(audio_f32) < MIN_AUDIO_LENGTH:
            # Pad bằng cách lặp lại audio
            pad_len = MIN_AUDIO_LENGTH - len(audio_f32)
            audio_f32 = np.concatenate([audio_f32, audio_f32[:pad_len]])
        
        tensor = torch.tensor(audio_f32).unsqueeze(0)
        with torch.no_grad():
            emb = self.model.encode_batch(tensor).detach().cpu().numpy()[0]
        return emb / (np.linalg.norm(emb) + 1e-8)
    
    def identify(self, audio_f32, duration=0.0):
        """Nhận diện hoặc đăng ký người nói mới"""
        try:
            emb = self.get_embedding(audio_f32)
            
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
                # Update embedding với EMA (chỉ nếu đủ dài)
                if duration >= MIN_DURATION_FOR_UPDATE:
                    self.speakers[idx] = (
                        (1 - EMBEDDING_UPDATE_WEIGHT) * self.speakers[idx] +
                        EMBEDDING_UPDATE_WEIGHT * emb
                    )
                self.counts[idx] += 1
                return f"spk_{idx+1:02d}"
            
            # Tạo speaker mới (nếu chưa quá MAX_SPEAKERS)
            if len(self.speakers) < MAX_SPEAKERS:
                self.speakers.append(emb)
                self.counts.append(1)
                self.next_id += 1
                return f"spk_{self.next_id:02d}"
            
            # Nếu quá MAX_SPEAKERS, gán vào người tương đồng nhất
            self.counts[idx] += 1
            return f"spk_{idx+1:02d}"
            
        except Exception as e:
            print(f"⚠️  Lỗi nhận diện người nói: {e}")
            return "spk_???"

# Initialize speaker manager
speaker_mgr = SpeakerManager(speaker_model) if SPEAKER_ENABLED else None

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
# AUDIO FILE PROCESSOR
# ==============================
def load_audio_file(file_path):
    """Load audio file and convert to 16kHz mono"""
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

def process_audio_chunks(audio_data, source_info=""):
    """Process audio data in chunks with speaker diarization"""
    print(f"\n🔄 Đang xử lý audio...")
    
    chunk_samples = int(CHUNK_SEC * SAMPLE_RATE)
    overlap_samples = int(OVERLAP_SEC * SAMPLE_RATE)
    segment_start_time = 0.0
    total_samples = len(audio_data)
    
    segments = []
    i = 0
    chunk_count = 0
    
    while i < total_samples:
        # Get chunk
        end_idx = min(i + chunk_samples, total_samples)
        proc_chunk = audio_data[i:end_idx]
        
        # Skip if too short
        if len(proc_chunk) < SAMPLE_RATE * 0.5:  # Skip chunks < 0.5s
            break
        
        chunk_duration = len(proc_chunk) / SAMPLE_RATE
        segment_end_time = segment_start_time + chunk_duration
        chunk_count += 1
        
        # Progress
        progress = (segment_end_time / (total_samples / SAMPLE_RATE)) * 100
        print(f"  [{progress:.1f}%] Xử lý chunk {chunk_count} ({segment_start_time:.1f}s - {segment_end_time:.1f}s)...", end='\r')
        
        # Save temp WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp_wav = tf.name
        sf.write(tmp_wav, proc_chunk, SAMPLE_RATE, subtype="PCM_16")

        try:
            # ASR
            res = asr_model.generate(
                input=tmp_wav,
                cache={},
                language=LANGUAGE,
                use_itn=True,
                batch_size_s=CHUNK_SEC,
                merge_vad=True,
                merge_length_s=CHUNK_SEC,
            )
            
            # Parse result
            text_raw = ""
            language = "auto"
            emotion = None
            event = None
            
            if isinstance(res, list) and len(res) > 0:
                result_item = res[0]
                if isinstance(result_item, dict):
                    text_raw = result_item.get("text", "")
                    if "key" in result_item:
                        key_info = result_item["key"]
                        if isinstance(key_info, str) and "<|" in key_info:
                            tags = key_info.split("<|")
                            for tag in tags:
                                if "|>" in tag:
                                    tag_value = tag.replace("|>", "").strip()
                                    if tag_value in ["zh", "en", "ja", "ko", "yue", "vi"]:
                                        language = tag_value
                                    elif tag_value in ["HAPPY", "SAD", "ANGRY", "NEUTRAL", "EXCITED"]:
                                        emotion = tag_value
                                    elif tag_value in ["Speech", "Music", "Applause", "Laughter", "Cough"]:
                                        event = tag_value
                else:
                    text_raw = str(result_item)
            
            text = rich_transcription_postprocess(text_raw)
            
            # Speaker diarization
            speaker_id = "spk_01"
            if text.strip() and speaker_mgr:
                speaker_id = speaker_mgr.identify(proc_chunk, chunk_duration)
            
            # Save result
            if text.strip():
                segment_data = {
                    "start": round(segment_start_time, 2),
                    "end": round(segment_end_time, 2),
                    "text": text,
                    "speaker": speaker_id,
                    "language": language
                }
                if emotion:
                    segment_data["emotion"] = emotion
                if event:
                    segment_data["event"] = event
                
                segments.append(segment_data)
            
        except Exception as e:
            print(f"\n⚠️  Lỗi xử lý chunk {chunk_count}: {e}")
        finally:
            try:
                os.remove(tmp_wav)
            except:
                pass
        
        # Move to next chunk with overlap
        i += (chunk_samples - overlap_samples)
        segment_start_time += (chunk_duration - OVERLAP_SEC)
    
    print(f"\n✓ Hoàn thành xử lý {chunk_count} chunks")
    return segments

# ==============================
# MAIN
# ==============================
def main():
    print("\n" + "="*70)
    print("🎯 SenseVoice + SpeechBrain Speaker Diarization")
    print("="*70)
    
    # Get input from user
    print("\n📌 Chọn nguồn input:")
    print("  1. File audio/video local")
    print("  2. YouTube URL")
    
    choice = input("\nNhập lựa chọn (1 hoặc 2): ").strip()
    
    audio_file = None
    source_info = ""
    cleanup_temp = False
    
    if choice == "1":
        file_path = input("Nhập đường dẫn file audio/video: ").strip().strip('"')
        if not os.path.exists(file_path):
            print(f"❌ File không tồn tại: {file_path}")
            sys.exit(1)
        
        audio_file = file_path
        source_info = Path(file_path).name
        all_results["source"] = {"type": "local_file", "path": file_path}
        
    elif choice == "2":
        youtube_url = input("Nhập YouTube URL: ").strip()
        audio_file, title = download_youtube_audio(youtube_url)
        source_info = title
        cleanup_temp = True
        all_results["source"] = {"type": "youtube", "url": youtube_url, "title": title}
        
    else:
        print("❌ Lựa chọn không hợp lệ!")
        sys.exit(1)
    
    # Update output filename
    global OUTPUT_FILE
    safe_name = "".join(c for c in source_info if c.isalnum() or c in (' ', '-', '_'))[:50]
    OUTPUT_FILE = f"test_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    print(f"\n📝 Kết quả sẽ lưu vào: {OUTPUT_FILE}")
    print("="*70 + "\n")
    
    try:
        # Load audio
        audio_data = load_audio_file(audio_file)
        
        # Process audio
        segments = process_audio_chunks(audio_data, source_info)
        
        # Update results
        all_results["segments"] = segments
        all_results["end_time"] = datetime.now().isoformat()
        all_results["total_segments"] = len(segments)
        
        if segments:
            full_transcript = " ".join([seg["text"] for seg in segments])
            all_results["full_transcript"] = full_transcript
            
            # Speaker stats
            if speaker_mgr:
                speaker_stats = {}
                for seg in segments:
                    spk = seg["speaker"]
                    if spk not in speaker_stats:
                        speaker_stats[spk] = {"count": 0, "duration": 0.0}
                    speaker_stats[spk]["count"] += 1
                    speaker_stats[spk]["duration"] += seg["end"] - seg["start"]
                all_results["speaker_stats"] = speaker_stats
        
        # Save results
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"✅ Đã lưu kết quả: {output_path}")
        print(f"   📊 Tổng số đoạn: {all_results['total_segments']}")
        
        if segments:
            print(f"   ⏱️  Tổng thời lượng: {segments[-1]['end']:.1f}s")
            if speaker_mgr and 'speaker_stats' in all_results:
                unique_speakers = len(all_results['speaker_stats'])
                print(f"   👥 Số người nói: {unique_speakers}")
                print(f"\n   📋 Chi tiết người nói:")
                for spk, stats in sorted(all_results['speaker_stats'].items()):
                    print(f"      • {spk}: {stats['count']} đoạn, {stats['duration']:.1f}s")
        
        print(f"{'='*70}\n")
        
        # Print some sample segments
        if segments:
            print("📝 Một số đoạn transcript mẫu:")
            for seg in segments[:5]:
                print(f"   [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['speaker']}: {seg['text'][:80]}...")
            if len(segments) > 5:
                print(f"   ... và {len(segments) - 5} đoạn nữa")
        
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup temporary YouTube file
        if cleanup_temp and audio_file and os.path.exists(audio_file):
            try:
                os.remove(audio_file)
                print(f"🧹 Đã xóa file tạm: {audio_file}")
            except:
                pass
    
    print("\n👋 Đã hoàn thành!")

if __name__ == "__main__":
    main()
