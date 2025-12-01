# Whisper trong xử lý thời gian thực

`faster-whisper` được dùng để tăng tốc inference Whisper bằng CTranslate2.

## Cấu hình chính trong mã

```python
WHISPER_MODEL = "small"  # hoặc tiny/base/medium/large-v3
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" (GPU) hoặc "int8" (CPU)
```

- `beam_size` thấp (1) cho partial transcription → tốc độ cao.
- `beam_size` cao (5+) cho full transcription → độ chính xác tốt hơn.
- `condition_on_previous_text=True` bật ngữ cảnh trong full transcription.

## Hai chế độ sử dụng

| Chế độ  | Thời điểm chạy                                    | Tham số                                                        | Mục đích                         |
| ------- | ------------------------------------------------- | -------------------------------------------------------------- | -------------------------------- |
| Partial | Mỗi ~1s âm thanh                                  | `beam_size=1`, `best_of=1`, `condition_on_previous_text=False` | Phản hồi nhanh nội dung đang nói |
| Full    | Khi kết thúc câu (VAD silence) hoặc đạt max chunk | `beam_size=5`, `best_of=5`, `condition_on_previous_text=True`  | Kết quả cuối chính xác hơn       |

## VAD phối hợp Whisper

- Dùng Silero VAD để xác định ranh giới câu nói.
- Giảm việc transcribe các đoạn im lặng → tiết kiệm thời gian.
- Sau khi phát hiện im lặng > ngưỡng (ví dụ 750ms) → xử lý full chunk.

## Lấy speaker ID

Kết hợp với SpeechBrain: sau khi gom tín hiệu speech thực tế (đã VAD), trích embedding và match hoặc tạo speaker mới.

## Ưu / Nhược

| Ưu điểm                    | Nhược điểm                           |
| -------------------------- | ------------------------------------ |
| Beam linh hoạt             | Yêu cầu GPU để tốc độ tốt            |
| Chất lượng tốt đa ngôn ngữ | Model lớn ngốn RAM VRAM              |
| Có thể chạy int8 trên CPU  | Partial đôi khi không mượt tiếng dài |

## Gợi ý tối ưu thêm

- Dùng `medium` hoặc `large-v3` khi cần chất lượng cao (chấp nhận trễ lớn hơn).
- Gộp partial liên tiếp thành một chuỗi hiển thị trên UI (frontend) thay vì in từng dòng.
- Bật `temperature=0.0` giảm dao động kết quả.

## Mẫu code rút gọn partial

```python
segments, info = whisper_model.transcribe(
    audio_data,
    language="vi",
    vad_filter=False,
    beam_size=1,
    best_of=1,
    condition_on_previous_text=False,
    temperature=0.0
)
```

## Mẫu code full

```python
segments, info = whisper_model.transcribe(
    audio_data_full,
    language="vi",
    vad_filter=False,
    beam_size=5,
    best_of=5,
    condition_on_previous_text=True,
    temperature=0.0
)
```

## Các lỗi thường gặp

| Lỗi                  | Nguyên nhân               | Khắc phục                                        |
| -------------------- | ------------------------- | ------------------------------------------------ |
| "CUDA out of memory" | Model quá lớn             | Dùng model nhỏ hơn (base/small) hoặc CPU int8    |
| Sai ngôn ngữ         | Không chỉ định `language` | Thêm `language="vi"` hoặc dùng auto detect trước |
| Chậm partial         | Chunk quá dài             | Giảm `PARTIAL_CHUNK_DURATION` xuống 0.8–1.0s     |

## Khi nào chọn Whisper?

- Muốn kiểm soát beam, temperature, context.
- Cần chất lượng ổn định ngay cả với noise vừa phải.
- Có GPU hoặc CPU mạnh (int8 vẫn OK cho model nhỏ).

Nếu cần emotion/event tags → chuyển sang SenseVoice.
