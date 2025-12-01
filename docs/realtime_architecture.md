# Kiến trúc bên trong thư mục `realtime`

## Thành phần chính

| Thành phần                                    | Mô tả                                                 |
| --------------------------------------------- | ----------------------------------------------------- |
| Callback âm thanh (`sounddevice.InputStream`) | Đẩy frame PCM vào queue                               |
| Queue (`queue.Queue`)                         | Tách thread ghi âm khỏi thread xử lý                  |
| Worker thread                                 | Lấy dữ liệu từ queue, gom chunk, chạy VAD/ASR/Speaker |
| Buffer tạm                                    | Mảng numpy tích luỹ mẫu âm thanh                      |
| VAD                                           | Xác định vùng speech, kết thúc câu                    |
| ASR                                           | Chuyển tiếng nói → văn bản (Whisper / SenseVoice)     |
| Speaker Diarization                           | Gán ID người nói, cập nhật embedding                  |
| JSON Writer                                   | Ghi lại kết quả cuối cùng                             |

## Mô hình threading

- 1 thread chính: Khởi tạo + quản lý vòng đời.
- 1 thread audio: `sounddevice` callback (không block).
- 1 thread worker: xử lý chunk (có thể mở rộng thành nhiều worker nếu muốn song song hoá ASR và diarization).

## Data Flow

```
[InputStream] --(frames)--> [Queue] --(pop)--> [Worker]
                                     |            |
                                     |            +--> [Buffer cập nhật]
                                     |            +--> [Kiểm đủ độ dài chunk]
                                     |            +--> [VAD hoặc merge_vad (SenseVoice)]
                                     |            +--> [ASR] --> [Speaker] --> [Segments]
                                     |                                         |
                                     +-----------------------------------------+
```

## Quản lý buffer

| Kiểu buffer                | File áp dụng                              | Chiến lược                                                       |
| -------------------------- | ----------------------------------------- | ---------------------------------------------------------------- |
| Dual-buffer (partial/full) | `realtime_diarization_improved.py`        | `recv_buffer` cho phản hồi nhanh, `full_buffer` cho kết quả cuối |
| Linear chunk + overlap     | `sen_voice.py`, `senvoi_spebrai_fixed.py` | Nối chunk; bỏ phần đã xử lý trừ overlap                          |
| Circular buffer + VAD      | `realtime_test.py`                        | Giữ tối đa X giây, xử lý khi detect speech hoàn chỉnh            |

## VAD tích hợp vs VAD ngoài

| Loại                 | Ưu điểm                    | Nhược điểm               |
| -------------------- | -------------------------- | ------------------------ |
| Silero VAD ngoài     | Kiểm soát tham số chi tiết | Thêm bước convert tensor |
| VAD trong SenseVoice | Ít thao tác                | Ít tuỳ chỉnh sâu         |

## Diarization chiến lược

- So khớp hai tầng: EMA embedding + centroid cluster (trong bản nâng cao `realtime_test.py`).
- Giới hạn kích thước cluster để tránh tiêu tốn RAM.
- Cập nhật khi độ dài đoạn đủ lớn (tránh nhiễu do speech rất ngắn).

## Quản lý lỗi

| Rủi ro                   | Cách giảm thiểu                            |
| ------------------------ | ------------------------------------------ |
| HuggingFace 404          | Dùng `download_speaker_model.py` trước     |
| Thiếu backend torchaudio | Monkey patch `list_audio_backends`         |
| Tắc nghẽn queue          | Giảm kích thước chunk / tăng tốc xử lý ASR |
| Memory leak temp WAV     | Xoá sau mỗi inference với try/except       |

## Mở rộng kiến trúc

- Tách ASR & Speaker thành hai hàng đợi (`asr_queue`, `spk_queue`).
- Dùng `asyncio` cho giao tiếp WebSocket tới frontend.
- Sử dụng GPU đa luồng: chia phiên thành nhiều worker.

## Ghi chú tối ưu

| Kỹ thuật                                         | Hiệu quả                    |
| ------------------------------------------------ | --------------------------- |
| Giảm beam partial                                | Giảm độ trễ cảm nhận        |
| Batch theo thời gian (`batch_size_s`) SenseVoice | Tối ưu throughput           |
| Int8 Whisper trên CPU                            | Giảm tiêu thụ RAM           |
| Đẩy VAD trước ASR                                | Giảm thời gian xử lý vô ích |

---

Kiến trúc này thiên về đơn giản + dễ mở rộng. Khi phiên dài hoặc tải cao → xem xét điều phối nhiều worker và cơ chế backpressure queue.
