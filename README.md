# Real-time Speaker Diarization System

Hệ thống nhận diện người nói và chuyển đổi giọng nói thành văn bản thời gian thực với 3 models khác nhau.

## 📁 Cấu trúc thư mục

```
realtime/
├── README.md                          # Tài liệu này
├── requirements.txt                   # Dependencies chung
├── requirements_sen_voice.txt         # Dependencies cho SenseVoice
│
├── realtime_diarization_improved.py   # Model 1: Whisper + SpeechBrain
├── sen_voice.py                      # Model 2: SenseVoice
├── senvoi_spebrai_fixed.py           # Model 3: SenseVoice + SpeechBrain
│
├── dataset/                          # Dataset JVS để đánh giá
│   └── jvs_ver1/
├── pretrained_models/                # Models đã tải về
├── evaluation/                       # Scripts đánh giá và so sánh
│   ├── eval_asr.py                  # Đánh giá ASR (Speech Recognition)
│   ├── eval_diarization.py          # Đánh giá Diarization
│   ├── compared.py                  # So sánh kết quả ASR
│   ├── compare_diarization.py       # So sánh kết quả Diarization
│   ├── eval_results/                # Kết quả đánh giá
│   └── *.bat                        # Batch files để chạy dễ dàng
│
├── tmp_model/                        # Models tạm thời
├── venv/                            # Virtual environment
└── *.json                           # Output files từ các lần chạy
```

## 🚀 Cài đặt và Khởi tạo

### 1. Tạo Virtual Environment

```bash
cd realtime
python -m venv venv

# Kích hoạt virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. Cài đặt Dependencies

```bash
# Cài đặt packages cơ bản
pip install -r requirements.txt

# Cài đặt packages cho SenseVoice
pip install -r requirements_sen_voice.txt

# Packages bổ sung cho evaluation
pip install pandas matplotlib seaborn scikit-learn
```

## 🎯 Các Models Available

### 1. **Whisper + SpeechBrain** (`realtime_diarization_improved.py`)

- **ASR**: Whisper Small
- **Speaker Diarization**: SpeechBrain ECAPA-TDNN
- **Ưu điểm**: Độ chính xác ASR cao, diarization tốt
- **Nhược điểm**: Tốc độ chậm nhất (RTF ~2.7)

### 2. **SenseVoice** (`sen_voice.py`)

- **ASR**: SenseVoice Small (FunAudioLLM)
- **Speaker Diarization**: Không có
- **Ưu điểm**: Tốc độ nhanh, độ chính xác ASR cao
- **Nhược điểm**: Không phân biệt được người nói

### 3. **SenseVoice + SpeechBrain** (`senvoi_spebrai_fixed.py`)

- **ASR**: SenseVoice Small
- **Speaker Diarization**: SpeechBrain ECAPA-TDNN
- **Ưu điểm**: **Tốt nhất** - Tốc độ nhanh nhất (RTF ~0.35), độ chính xác cao nhất
- **Nhược điểm**: Cài đặt phức tạp hơn

## 🎤 Chạy Real-time Recognition

### Chạy từng model riêng lẻ:

```bash
# Model 1: Whisper + SpeechBrain
python realtime_diarization_improved.py

# Model 2: SenseVoice only
python sen_voice.py

# Model 3: SenseVoice + SpeechBrain (recommended)
python senvoi_spebrai_fixed.py
```

### Output:

- Console: Hiển thị real-time transcript
- JSON file: Lưu chi tiết với timestamp (format: `[model]_output_YYYYMMDD_HHMMSS.json`)

## 📊 Đánh giá và So sánh Models

### 1. Đánh giá ASR (Speech Recognition)

```bash
cd evaluation

# Chạy đánh giá ASR cho cả 3 models
eval_asr.bat

# Hoặc Python trực tiếp
python eval_asr.py --max_files 100  # Test nhanh với 100 files
python eval_asr.py                  # Full dataset (~14,000+ files)
```

### 2. Đánh giá Diarization

```bash
cd evaluation

# Chạy đánh giá diarization
eval_diarization.bat

# Hoặc Python trực tiếp
python eval_diarization.py --max_files 100  # Test nhanh
python eval_diarization.py                  # Full dataset
```

### 3. So sánh kết quả

```bash
# So sánh kết quả ASR
python compared.py

# So sánh kết quả Diarization
python compare_diarization.py
```

## 📈 Kết quả Đánh giá

### ASR Performance (trên JVS dataset):

| Model                        | WER (%)   | CER (%)  | RTF       | Real-time |
| ---------------------------- | --------- | -------- | --------- | --------- |
| **SenseVoice + SpeechBrain** | **11.70** | **8.32** | **0.355** | ✅        |
| SenseVoice                   | 13.89     | 10.08    | 0.805     | ✅        |
| Whisper + SpeechBrain        | 16.12     | 12.58    | 2.749     | ❌        |

### Diarization Performance:

| Model                        | DER (%)  | F1 (%)    | RTF       | Real-time |
| ---------------------------- | -------- | --------- | --------- | --------- |
| **SenseVoice + SpeechBrain** | **5.33** | **91.84** | **0.049** | ✅        |
| Whisper + SpeechBrain        | 10.36    | 93.54     | 0.168     | ✅        |
| SenseVoice                   | 87.27    | 76.58     | 0.084     | ✅        |

### 🏆 **Kết luận**:

**SenseVoice + SpeechBrain** là model tốt nhất với:

- WER thấp nhất (11.70%)
- DER thấp nhất (5.33%)
- RTF nhanh nhất (0.355 cho ASR, 0.049 cho diarization)
- Khả năng real-time tốt nhất

## 🔧 Troubleshooting

### Lỗi thường gặp:

1. **ImportError với SpeechBrain/FunASR**:

```bash
pip install --upgrade speechbrain funasr
pip install soundfile librosa torch torchaudio
```

2. **CUDA not available**:

- Models sẽ tự động chuyển về CPU
- Tốc độ chậm hơn nhưng vẫn hoạt động

3. **Microphone không hoạt động**:

```bash
pip install sounddevice
# Kiểm tra device available
python -c "import sounddevice as sd; print(sd.query_devices())"
```

4. **Memory errors**:

- Giảm `CHUNK_SEC` trong config
- Sử dụng CPU thay vì CUDA

### Cấu hình tùy chỉnh:

Sửa các thông số trong file Python:

```python
SAMPLE_RATE = 16000    # Sample rate
CHUNK_SEC = 3.0        # Độ dài mỗi chunk (giây)
OVERLAP_SEC = 0.3      # Overlap giữa các chunk
DEVICE = "cpu"         # hoặc "cuda"
```

## 📝 Output Format

### JSON Output Structure:

```json
{
  "start_time": "2025-11-28T10:30:00.000000",
  "model": "SenseVoice + SpeechBrain",
  "device": "cpu",
  "sample_rate": 16000,
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 3.2,
      "duration": 3.2,
      "text": "Xin chào, tôi là người nói số một",
      "speaker": "speaker_1",
      "confidence": 0.95
    },
    {
      "start_time": 3.5,
      "end_time": 6.8,
      "duration": 3.3,
      "text": "Và tôi là người nói số hai",
      "speaker": "speaker_2",
      "confidence": 0.92
    }
  ]
}
```

## 🛠 Development

### Thêm model mới:

1. Tạo file Python mới theo template
2. Implement interface tương tự các model có sẵn
3. Thêm vào `MODELS` dict trong evaluation scripts
4. Chạy evaluation để so sánh

### Customize evaluation:

- Sửa `eval_asr.py` và `eval_diarization.py`
- Thêm metrics mới vào comparison scripts
- Tùy chỉnh visualizations trong plot functions

## 📚 References

- **Whisper**: OpenAI Whisper ASR model
- **SenseVoice**: FunAudioLLM SenseVoice Small
- **SpeechBrain**: ECAPA-TDNN speaker recognition
- **JVS Dataset**: Japanese Versatile Speech corpus
- **Evaluation Metrics**: WER, CER, DER, RTF, F1-score

---