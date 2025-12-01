# SenseVoice (FunAudioLLM) Realtime

SenseVoiceSmall được dùng để:

- Nhận dạng đa ngôn ngữ (auto language)
- Trả về thẻ (tags) emotion (HAPPY, SAD, NEUTRAL…) và event (Speech, Music, Applause…)
- Tích hợp sẵn VAD (`fsmn-vad`) giúp đơn giản pipeline khi chỉ cần ASR.

## Khởi tạo model

```python
from funasr import AutoModel
model = AutoModel(
    model="FunAudioLLM/SenseVoiceSmall",
    device="cpu",  # hoặc "cuda:0"
    hub="hf",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
)
```

## Chiến lược chunk + overlap

Trong `sen_voice.py`:

- `CHUNK_SEC = 3.0`: mỗi 3 giây gửi một lần để inference.
- `OVERLAP_SEC = 0.3`: giữ lại 0.3 giây cuối chunk trước nối vào đầu chunk sau (giảm mất mát biên câu).

Buffer xử lý:

```python
buffer = np.concatenate([buffer, frames])
while buffer.shape[0] >= chunk_samples:
    proc_chunk = buffer[:chunk_samples]
    buffer = buffer[chunk_samples - overlap_samples:]
```

## Trích xuất kết quả

Mỗi kết quả trả về danh sách `res`. Phần tử đầu tiên thường là dict:

```python
result_item = res[0]
text_raw = result_item.get("text", "")
key_info = result_item.get("key")  # chứa chuỗi tag <|en|><|NEUTRAL|><|Speech|>
```

Tách tag:

- Ngôn ngữ: `zh`, `en`, `ja`, `ko`, `yue`, `vi`...
- Emotion: `HAPPY`, `SAD`, `ANGRY`, `NEUTRAL`, `EXCITED`...
- Event: `Speech`, `Music`, `Applause`, `Laughter`, `Cough`...

## Post-process

Dùng:

```python
from funasr.utils.postprocess_utils import rich_transcription_postprocess
text = rich_transcription_postprocess(text_raw)
```

## Khi tích hợp Speaker Diarization

File `senvoi_spebrai_fixed.py` thêm lớp `SpeakerManager`:

- Trích embedding từ SpeechBrain với chuẩn hoá.
- So sánh cosine similarity > `SIMILARITY_THRESHOLD` → gán speaker cũ, ngược lại tạo mới.
- Cập nhật embedding = EMA nếu độ dài chunk >= ngưỡng tối thiểu.

## Ưu / Nhược

| Ưu điểm                 | Nhược điểm                                  |
| ----------------------- | ------------------------------------------- |
| Có emotion/event tags   | Beam ít tuỳ chỉnh hơn Whisper               |
| Tự động đa ngôn ngữ tốt | Chất lượng phụ thuộc vào mô hình pretrained |
| Tích hợp VAD nội bộ     | Model nhỏ vẫn chậm trên CPU nếu chunk dài   |

## Gợi ý tối ưu

- Giảm `CHUNK_SEC` xuống 2.0s nếu cần độ trễ thấp hơn.
- Tắt `merge_vad=True` nếu muốn tự quản lý ranh giới bằng Silero VAD bên ngoài.
- Dùng GPU khi cần xử lý đồng thời nhiều phiên.

## Mẫu JSON kết quả (rút gọn)

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 2.9,
      "text": "Xin chào mọi người",
      "language": "vi",
      "emotion": "NEUTRAL",
      "event": "Speech"
    }
  ]
}
```

## Lỗi thường gặp

| Lỗi                  | Nguyên nhân                       | Khắc phục                                    |
| -------------------- | --------------------------------- | -------------------------------------------- |
| Không có thẻ emotion | Model không tự sinh cho đoạn ngắn | Dùng chunk dài hơn (>2s) hoặc ghép các chunk |
| Chậm trên CPU        | Chunk lớn 3–5s                    | Giảm `CHUNK_SEC`, dùng GPU                   |
| File tạm không xoá   | Quyền ghi hoặc crash              | Thêm try/except khi `os.remove(tmp_wav)`     |

## Khi nên chọn SenseVoice?

- Cần cảm xúc và loại sự kiện.
- Muốn auto language detection nhiều ngôn ngữ trong cùng phiên.
- Muốn đơn giản hoá (VAD tích hợp).

Nếu cần độ kiểm soát beam và chất lượng cao cho một ngôn ngữ → cân nhắc Whisper.
