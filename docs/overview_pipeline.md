# Tổng quan pipeline realtime

Pipeline tổng quát (ví dụ `realtime_diarization_improved.py` / `realtime_test.py`):

1. Khởi tạo & vá (patch) môi trường:
   - Fix `huggingface_hub` tham số `use_auth_token` → `token`.
   - Patch `torchaudio` nếu thiếu API cũ để SpeechBrain hoạt động.
2. Tải mô hình:
   - VAD: Silero (torchaud hub) hoặc VAD tích hợp SenseVoice.
   - ASR: Whisper (faster-whisper) hoặc SenseVoiceSmall.
   - Speaker: SpeechBrain ECAPA-TDNN (tải local trước bằng `download_speaker_model.py`).
3. Khởi tạo cấu trúc dữ liệu:
   - Queue (`queue.Queue`) để gom audio từ callback.
   - Buffer ngắn (partial) & buffer dài (full) hoặc circular buffer (giữ tối đa X giây).
4. Thu âm realtime:
   - Dùng `sounddevice.InputStream` callback đẩy mảng `float32` vào queue.
   - Chuyển stereo → mono nếu cần.
5. Gom chunk & tiền xử lý:
   - Nối các frame tới đủ kích thước chunk theo cấu hình (ví dụ 1s / 3s / 0.5s).
   - Overlap một phần (ví dụ 0.3s) giảm mất mát ở biên câu.
6. Phát hiện speech:
   - VAD chạy trên buffer → tách vùng có tiếng nói.
   - Điều kiện kết thúc speech: đủ im lặng > `min_silence_ms`.
7. Partial transcription (tùy chọn):
   - Khi buffer đạt `PARTIAL_CHUNK_DURATION` (1s), chạy ASR nhanh (beam thấp, không dùng context).
   - In ra console gợi ý nội dung đang nói.
8. Full transcription:
   - Khi phát hiện kết thúc câu (silence) hoặc buffer quá dài → chạy ASR đầy đủ (beam cao, context bật).
   - Lưu segment vào danh sách kết quả.
9. Speaker diarization:
   - Trích embedding từ audio speech chunk.
   - So sánh cosine similarity với embeddings đã lưu.
   - Cập nhật embedding bằng EMA nếu đủ dài.
10. Ghi kết quả:
    - Mỗi segment: start/end, speaker, text, language, optional emotion/event.
    - Sau khi dừng: ghi file JSON `realtime_diarization_output_YYYYMMDD_HHMMSS.json`.

## Sơ đồ luồng đơn giản

```
[Microphone / Loopback] --> [Callback] --> [Queue] --> [Worker]
                                          |           |
                                          v           v
                                   [Buffer tích lũy]  [VAD]
                                          |           |
                                 (partial transcription)  (speech segment)
                                                          |
                                                   [Full ASR]
                                                          |
                                               [Speaker Diarization]
                                                          |
                                                    [Lưu JSON]
```

## Các chiến lược được áp dụng

| Chiến lược         | Mục đích             | Mô tả                                           |
| ------------------ | -------------------- | ----------------------------------------------- |
| Dual-buffer        | Latency vs Accuracy  | Partial nhanh + full chính xác khi kết thúc câu |
| Overlap chunk      | Giảm mất từ đầu/cuối | Giữ lại 0.3s cuối chunk trước làm đầu chunk sau |
| EMA embedding      | Ổn định speaker ID   | Cập nhật embedding dần theo thời gian           |
| VAD trước ASR      | Tiết kiệm tài nguyên | Không transcribe đoạn im lặng                   |
| Force max duration | Tránh delay quá lâu  | Nếu vượt `MAX_CHUNK_DURATION` thì buộc xử lý    |

## Khi nào dùng Whisper vs SenseVoice?

| Trường hợp                          | Nên chọn Whisper | Nên chọn SenseVoice                |
| ----------------------------------- | ---------------- | ---------------------------------- |
| Hỗ trợ cảm xúc/sự kiện              | Không            | Có (emotion/event tags)            |
| Tài nguyên GPU hạn chế              | Có thể với int8  | CPU vẫn chạy được (nhưng chậm hơn) |
| Tối ưu tiếng Việt                   | Cần tinh chỉnh   | Tự động multi-lingual tốt          |
| Beam linh hoạt / kiểm soát chi tiết | Rất tốt          | Ít tham số hơn                     |

## File JSON mẫu (rút gọn)

```json
{
  "total_segments": 3,
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "speaker": "spk_01",
      "text": "Xin chào mọi người"
    },
    {
      "start": 2.6,
      "end": 5.1,
      "speaker": "spk_02",
      "text": "Hôm nay chúng ta demo"
    }
  ]
}
```

## Điểm mở rộng

- Thêm WebSocket streaming cho giao diện web.
- Gắn phân loại chủ đề (topic classification) theo chunk.
- Kết nối vector DB để tìm kiếm đoạn hội thoại sau này.
- Đồng bộ timestamp với video gốc (nếu là loopback YouTube).

Đọc chi tiết về kiến trúc: `realtime_kien_truc.md`.
