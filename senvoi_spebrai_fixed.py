"""
Realtime SenseVoice + SpeechBrain Speaker Diarization
Ghi âm từ microphone, nhận diện người nói và chuyển đổi giọng nói thành văn bản realtime.
Kết quả được lưu vào file JSON.
"""

import os
import sys
import tempfile
import queue
import threading
import time
import json
from datetime import datetime
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from scipy.spatial.distance import cdist

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
# AUDIO PROCESSING
# ==============================
q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    """Callback nhận audio từ microphone"""
    if status:
        print("⚠️  Sounddevice status:", status)
    q.put(indata.copy())

def worker():
    """Worker thread xử lý audio"""
    buffer = np.zeros((0, CHANNELS), dtype=np.float32)
    chunk_samples = int(CHUNK_SEC * SAMPLE_RATE)
    overlap_samples = int(OVERLAP_SEC * SAMPLE_RATE)
    segment_start_time = 0.0

    while True:
        frames = q.get()
        if frames is None:
            break
        buffer = np.concatenate([buffer, frames], axis=0)

        while buffer.shape[0] >= chunk_samples:
            proc_chunk = buffer[:chunk_samples]
            chunk_duration = len(proc_chunk) / SAMPLE_RATE
            segment_end_time = segment_start_time + chunk_duration
            
            buffer = buffer[chunk_samples - overlap_samples:]

            # Lưu temp WAV
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
                    speaker_id = speaker_mgr.identify(proc_chunk[:, 0], chunk_duration)
                
                # Save result
                if text.strip():
                    ts = time.strftime("%H:%M:%S")
                    segment_data = {
                        "start": round(segment_start_time, 2),
                        "end": round(segment_end_time, 2),
                        "text": text,
                        "speaker": speaker_id,
                        "language": language,
                        "timestamp": ts
                    }
                    if emotion:
                        segment_data["emotion"] = emotion
                    if event:
                        segment_data["event"] = event
                    
                    all_results["segments"].append(segment_data)
                    
                    # Console output
                    info_str = f"[{ts}] [{segment_start_time:.1f}s-{segment_end_time:.1f}s]"
                    if language != "auto":
                        info_str += f" <{language}>"
                    if emotion:
                        info_str += f" ({emotion})"
                    print(f"{info_str} {speaker_id}: {text}")
                
            except Exception as e:
                print(f"⚠️  Lỗi xử lý: {e}")
            finally:
                try:
                    os.remove(tmp_wav)
                except:
                    pass
            
            segment_start_time += (chunk_duration - OVERLAP_SEC)

# ==============================
# MAIN
# ==============================
print("\n" + "="*70)
print("🎤 Realtime SenseVoice + SpeechBrain Speaker Diarization")
print("="*70)
print(f"📝 Kết quả sẽ lưu vào: {OUTPUT_FILE}")
print(f"🎙️  Bắt đầu ghi âm... (Nhấn Ctrl+C để dừng)")
print("="*70 + "\n")

worker_thread = threading.Thread(target=worker, daemon=True)
worker_thread.start()

try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=audio_callback,
        blocksize=int(0.1*SAMPLE_RATE)
    ):
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    print("\n⛔ Đang dừng...")
finally:
    q.put(None)
    worker_thread.join(timeout=2)
    
    # Save results
    all_results["end_time"] = datetime.now().isoformat()
    all_results["total_segments"] = len(all_results["segments"])
    
    if all_results["segments"]:
        full_transcript = " ".join([seg["text"] for seg in all_results["segments"]])
        all_results["full_transcript"] = full_transcript
        
        # Speaker stats
        if speaker_mgr:
            speaker_stats = {}
            for seg in all_results["segments"]:
                spk = seg["speaker"]
                if spk not in speaker_stats:
                    speaker_stats[spk] = {"count": 0, "duration": 0.0}
                speaker_stats[spk]["count"] += 1
                speaker_stats[spk]["duration"] += seg["end"] - seg["start"]
            all_results["speaker_stats"] = speaker_stats
    
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Đã lưu kết quả: {output_path}")
        print(f"   📊 Tổng số đoạn: {all_results['total_segments']}")
        if all_results["segments"]:
            print(f"   ⏱️  Tổng thời lượng: {all_results['segments'][-1]['end']:.1f}s")
            if speaker_mgr:
                print(f"   👥 Số người nói: {len(set(seg['speaker'] for seg in all_results['segments']))}")
    except Exception as e:
        print(f"❌ Lỗi lưu file: {e}")
    
    print("\n👋 Đã thoát.")
