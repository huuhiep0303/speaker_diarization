"""
Realtime microphone -> SenseVoiceSmall (FunAudioLLM) demo.

Nguyên lý:
- Ghi audio từ microphone thành các đoạn (chunk) dài CHUNK_SEC (ví dụ 3s).
- Lưu tạm thành WAV và gọi funasr.AutoModel.generate() xử lý đoạn đó.
- In kết quả realtime ra console và lưu vào file JSON.

Lưu ý:
- Thay device="cuda:0" nếu có GPU; ngược lại dùng device="cpu".
- Bạn có thể gộp chunk nhỏ hơn / lớn hơn tuỳ tradeoff latency <-> quality.
"""

import os
import tempfile
import queue
import threading
import time
import json
from datetime import datetime
import numpy as np
import sounddevice as sd
import soundfile as sf
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# --- Config ---
MODEL_NAME = "FunAudioLLM/SenseVoiceSmall"   # model trên HF
DEVICE = "cpu"   # hoặc "cuda:0" nếu có GPU
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SEC = 3.0     # chiều dài mỗi chunk (giây) gửi đi để transcribe
OVERLAP_SEC = 0.3   # overlap giữa các chunk (tăng độ chính xác ở ranh giới)
LANGUAGE = "auto"   # hoặc "en", "zh", "yue", "ja", "ko"

# Output file
OUTPUT_DIR = "."
OUTPUT_FILE = f"sensevoice_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# Global storage for results
all_results = {
    "start_time": datetime.now().isoformat(),
    "model": MODEL_NAME,
    "device": DEVICE,
    "sample_rate": SAMPLE_RATE,
    "segments": []
}

# --- Prepare model (tải lần đầu sẽ mất thời gian) ---
print("Loading SenseVoice model (AutoModel)...")
model = AutoModel(
    model=MODEL_NAME,
    device=DEVICE,
    hub="hf",            # lấy từ HuggingFace hub
    vad_model="fsmn-vad",# bật VAD để tự động cắt đoạn (có thể tắt nếu muốn tự quản lý chunk)
    vad_kwargs={"max_single_segment_time": 30000},
)
print("Model loaded.")

# --- Audio queue nhận frame từ callback ---
q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    """Sounddevice callback — push raw audio frames (float32) vào queue."""
    if status:
        print("Sounddevice status:", status)
    # convert to mono float32 numpy
    audio_chunk = indata.copy()
    q.put(audio_chunk)

# --- Worker thread: gom chunk từ queue, lưu file tạm, gọi model.generate() ---
def worker():
    buffer = np.zeros((0, CHANNELS), dtype=np.float32)
    chunk_samples = int(CHUNK_SEC * SAMPLE_RATE)
    overlap_samples = int(OVERLAP_SEC * SAMPLE_RATE)
    
    # Track timing for segments
    segment_start_time = 0.0  # in seconds
    recording_start = time.time()

    while True:
        # blocking get - lấy 1 frame (kích thước mặc định do sounddevice)
        frames = q.get()
        if frames is None:
            break
        buffer = np.concatenate([buffer, frames], axis=0)

        # nếu buffer đủ dài, xử lý một chunk (có overlap)
        while buffer.shape[0] >= chunk_samples:
            proc_chunk = buffer[:chunk_samples]  # lấy CHUNK_SEC đầu
            
            # Calculate timestamps
            chunk_duration = len(proc_chunk) / SAMPLE_RATE
            segment_end_time = segment_start_time + chunk_duration
            
            # giữ lại phần overlap ở buffer (shift)
            buffer = buffer[chunk_samples - overlap_samples :]

            # lưu file tạm
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tmp_wav = tf.name
            # soundfile expects shape (n_samples, n_channels)
            sf.write(tmp_wav, proc_chunk, SAMPLE_RATE, subtype="PCM_16")

            try:
                # gọi model inference: sử dụng generate() như docs
                res = model.generate(
                    input=tmp_wav,
                    cache={},
                    language=LANGUAGE,
                    use_itn=True,
                    batch_size_s=CHUNK_SEC,   # dynamic batching theo giây
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
                    
                    # Extract text
                    if isinstance(result_item, dict):
                        text_raw = result_item.get("text", "")
                        # Extract language tag if available
                        if "key" in result_item:
                            key_info = result_item["key"]
                            if isinstance(key_info, str):
                                # Parse tags like <|zh|><|NEUTRAL|><|Speech|>
                                if "<|" in key_info and "|>" in key_info:
                                    tags = key_info.split("<|")
                                    for tag in tags:
                                        if "|>" in tag:
                                            tag_value = tag.replace("|>", "").strip()
                                            # Language tags: zh, en, ja, ko, yue, etc.
                                            if tag_value in ["zh", "en", "ja", "ko", "yue", "vi"]:
                                                language = tag_value
                                            # Emotion tags: HAPPY, SAD, ANGRY, NEUTRAL, etc.
                                            elif tag_value in ["HAPPY", "SAD", "ANGRY", "NEUTRAL", "EXCITED"]:
                                                emotion = tag_value
                                            # Event tags: Speech, Music, Applause, Laughter, etc.
                                            elif tag_value in ["Speech", "Music", "Applause", "Laughter", "Cough"]:
                                                event = tag_value
                    else:
                        try:
                            text_raw = result_item[0]["text"] if isinstance(result_item, list) else str(result_item)
                        except:
                            text_raw = str(res)
                
                text = rich_transcription_postprocess(text_raw)
                
                # Only save non-empty results
                if text.strip():
                    ts = time.strftime("%H:%M:%S")
                    
                    # Create segment data
                    segment_data = {
                        "start": round(segment_start_time, 2),
                        "end": round(segment_end_time, 2),
                        "text": text,
                        "language": language,
                        "timestamp": ts
                    }
                    
                    # Add optional fields
                    if emotion:
                        segment_data["emotion"] = emotion
                    if event:
                        segment_data["event"] = event
                    
                    # Save to global results
                    all_results["segments"].append(segment_data)
                    
                    # Print to console with more info
                    info_str = f"[{ts}] [{segment_start_time:.1f}s - {segment_end_time:.1f}s]"
                    if language != "auto":
                        info_str += f" <{language}>"
                    if emotion:
                        info_str += f" ({emotion})"
                    if event:
                        info_str += f" [{event}]"
                    print(f"{info_str} >> {text}")
                
            except Exception as e:
                print("Inference error:", e)
            finally:
                # xóa file tạm
                try:
                    os.remove(tmp_wav)
                except:
                    pass
            
            # Update start time for next segment (accounting for overlap)
            segment_start_time += (chunk_duration - OVERLAP_SEC)

# --- Start audio stream and worker thread ---
worker_thread = threading.Thread(target=worker, daemon=True)
worker_thread.start()

try:
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32",
                        callback=audio_callback, blocksize=int(0.1*SAMPLE_RATE)):
        print("Recording... Press Ctrl+C to stop.")
        print(f"Results will be saved to: {OUTPUT_FILE}")
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    # stop worker
    q.put(None)
    worker_thread.join(timeout=2)
    
    # Save results to JSON file
    all_results["end_time"] = datetime.now().isoformat()
    all_results["total_segments"] = len(all_results["segments"])
    
    # Calculate total transcript
    full_transcript = " ".join([seg["text"] for seg in all_results["segments"]])
    all_results["full_transcript"] = full_transcript
    
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Results saved to: {output_path}")
        print(f"  - Total segments: {all_results['total_segments']}")
        print(f"  - Total duration: {all_results['segments'][-1]['end']:.1f}s" if all_results['segments'] else "  - No segments recorded")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    print("Exited.")
