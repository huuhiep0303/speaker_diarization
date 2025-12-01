# Tích hợp SenseVoice + SpeechBrain (Speaker Diarization)

File chính: `senvoi_spebrai_fixed.py`.
Mục tiêu: vừa nhận dạng tiếng nói (ASR) + gán người nói theo thời gian thực.

## Luồng hoạt động

1. Thu chunk (3s) + overlap (0.3s).
2. Gọi `asr_model.generate()` lấy text + tags.
3. Trích xuất speaker embedding bằng SpeechBrain ECAPA-TDNN:
   - Chuẩn hoá vector (L2 norm).
   - So sánh cosine với các embedding đã có.
4. Quyết định người nói:
   - Nếu similarity ≥ `SIMILARITY_THRESHOLD` → gán speaker cũ.
   - Ngược lại tạo ID mới (tối đa `MAX_SPEAKERS`).
5. Cập nhật embedding (EMA) nếu đoạn đủ dài (`MIN_DURATION_FOR_UPDATE`).
6. Lưu segment vào `all_results["segments"]`.

## Lớp `SpeakerManager`

```python
class SpeakerManager:
    def get_embedding(audio_f32): # tạo embedding chuẩn hoá
    def identify(audio_f32, duration): # trả về mã 'spk_XX'
```

- `counts`: đếm số lần speaker xuất hiện.
- Chiến lược padding: nếu < 0.5s → lặp lại mẫu ban đầu tránh vector rỗng.

## Tham số quan trọng

| Tham số                   | Vai trò                         | Gợi ý chỉnh               |
| ------------------------- | ------------------------------- | ------------------------- |
| `SIMILARITY_THRESHOLD`    | Ngưỡng cosine phân biệt speaker | 0.55–0.7 tuỳ môi trường   |
| `EMBEDDING_UPDATE_WEIGHT` | Tốc độ cập nhật EMA             | 0.2–0.4 (quá cao dễ trôi) |
| `MIN_DURATION_FOR_UPDATE` | Độ dài tối thiểu để update      | ≥2.0s giảm nhiễu          |
| `MAX_SPEAKERS`            | Giới hạn số người               | Tăng nếu hội thoại lớn    |

## Chiến lược ổn định

- EMA giúp embedding đại diện tiến hoá theo thời gian (giảm drift nhanh).
- Tách logic nhận diện khỏi phần ASR để dễ thay thế bằng model khác.
- Giới hạn số cluster để tránh tràn bộ nhớ với phiên dài.

## JSON mở rộng

Khi có diarization, file đầu ra có thêm:

```json
"speaker_stats": {
  "spk_01": {"count": 12, "duration": 35.4},
  "spk_02": {"count": 8, "duration": 21.7}
}
```

## Mẹo giảm lỗi nhận diện sai

| Vấn đề                         | Nguyên nhân                       | Cách khắc phục                                       |
| ------------------------------ | --------------------------------- | ---------------------------------------------------- |
| Speaker ID nhảy liên tục       | Threshold quá cao hoặc chunk ngắn | Giảm threshold; tăng chunk hoặc gộp nhiều chunk      |
| Nhiễu tạo speaker mới          | Noise / âm nền dài                | Thêm VAD lớp 2 hoặc filter RMS                       |
| Hai người giống nhau gộp chung | Giọng quá tương đồng              | Bật chế độ phân mảnh: giảm `EMBEDDING_UPDATE_WEIGHT` |

## Mở rộng trong tương lai

- Áp dụng clustering theo thời gian (agglomerative) sau phiên để tinh chỉnh ID.
- Kết hợp vad + energy + spectral gating làm sạch trước khi embedding.
- Thêm giao diện hiển thị timeline speaker.

## Khi so với Whisper + SpeechBrain

| Tiêu chí             | SenseVoice + SpeechBrain | Whisper + SpeechBrain                       |
| -------------------- | ------------------------ | ------------------------------------------- |
| Emotion/Event tags   | Có                       | Không                                       |
| Auto language        | Tốt                      | Cần chọn hoặc detect riêng                  |
| Kiểm soát beam       | Ít                       | Nhiều                                       |
| Độ linh hoạt tham số | Trung bình               | Cao                                         |
| Dễ thêm diarization  | Đã tích hợp demo         | Tùy file `realtime_diarization_improved.py` |

## Đo lường nhanh

- Log số speaker, trung bình similarity khi gán.
- RTF (real-time factor) = thời gian xử lý / thời lượng chunk.

Nếu cần pipeline lai (SenseVoice partial + Whisper full) có thể tạo class điều phối chunk dựa trên độ dài và ngữ cảnh.
