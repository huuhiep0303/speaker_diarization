# Tài liệu thư mục `realtime`

Mục tiêu của folder `realtime/` là xây dựng pipeline xử lý âm thanh thời gian thực (real-time) cho các bài toán:

- Ghi âm trực tiếp từ microphone hoặc loopback hệ thống (capture audio đang phát)
- Tự động phát hiện đoạn có tiếng nói (VAD)
- Nhận dạng tiếng nói (ASR) bằng Whisper hoặc SenseVoice
- Gán nhãn người nói (Speaker Diarization) dùng SpeechBrain (ECAPA-TDNN)
- Kết hợp chiến lược buffer và chunk để cân bằng latency và độ chính xác
- Lưu kết quả ra JSON (phục vụ các bước hậu xử lý hoặc phân tích)

## Các script chính

| File                               | Chức năng                                                                          |
| ---------------------------------- | ---------------------------------------------------------------------------------- |
| `realtime_diarization_improved.py` | Pipeline Whisper + VAD Silero + SpeechBrain, hai chế độ partial/full transcription |
| `sen_voice.py`                     | Demo SenseVoice realtime, chưa gắn speaker diarization                             |
| `senvoi_spebrai_fixed.py`          | Kết hợp SenseVoice + SpeechBrain để vừa nhận dạng vừa gán người nói                |
| `realtime_test.py`                 | Thu âm hệ thống/mic, VAD + (Whisper hoặc SenseVoice) + diarization tổng hợp        |
| `download_speaker_model.py`        | Script tải trước model speaker để tránh lỗi mạng/huggingface                       |

## Các mô hình được dùng

- Whisper (qua `faster-whisper`) cho tốc độ tốt + beam search linh hoạt
- SenseVoice (FunAudioLLM) cho đa ngôn ngữ + nhận cảm xúc/sự kiện (emotion/event tags)
- SpeechBrain ECAPA-TDNN cho embedding người nói và cập nhật/so khớp động
- Silero VAD để phân định ranh giới speech / silence chính xác trong thời gian thực

## Triết lý thiết kế

1. Ưu tiên tính ổn định: patch các incompatibility (huggingface_hub, torchaudio) ngay đầu file.
2. Giảm độ trễ: dùng chunk nhỏ (0.5–3s) + overlap, partial transcription nhanh (beam_size thấp) và full transcription khi phát hiện kết thúc đoạn.
3. Tối ưu tài nguyên: Tự động chọn `cuda` nếu có GPU, fallback `cpu` và giảm `compute_type`.
4. Mềm dẻo: Cho phép chuyển đổi giữa Whisper / SenseVoice đơn giản bằng flag.
5. Tự mở rộng: Speaker diarization tự sinh ID mới khi độ tương đồng < ngưỡng, cập nhật embedding bằng EMA.

## Đọc tiếp

- `overview_pipeline.md`: Tổng quan luồng xử lý.
- `model_whisper.md`: Chi tiết cách sử dụng Whisper trong realtime.
- `model_sensevoice.md`: Khai thác SenseVoice và các tag emotion/event.
- `model_sensevoice_speechbrain.md`: Tích hợp SenseVoice + Speaker diarization.
- `realtime_kien_truc.md`: Kiến trúc module, thread, queue, buffer.
- `realtime_xu_ly_thoi_gian_thuc.md`: Chiến lược chia chunk, partial vs full.
- `danh_gia_va_so_sanh.md`: So sánh hiệu năng giữa các mô hình và cấu hình.
- `huong_dan_chay_thu.md`: Cách chạy nhanh cho người mới.
- `khac_phuc_su_co.md`: Lỗi thường gặp và cách xử lý.

---

Nếu bạn là người mới: đọc lần lượt `overview_pipeline.md` → `huong_dan_chay_thu.md` → tài liệu mô hình bạn quan tâm.
