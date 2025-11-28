# 📊 Speaker Diarization Evaluation

Hệ thống đánh giá chất lượng cho các model speaker diarization và ASR trong folder `realtime/`.

## 📁 Cấu trúc thư mục

```
evaluation/
├── create_dataset.py       # Tạo dataset cho evaluation
├── eval_diarization.py     # Đánh giá speaker verification
├── eval_asr.py            # Đánh giá ASR/transcription quality
├── eval_diarization.bat   # Script Windows cho diarization
├── eval_asr.bat          # Script Windows cho ASR
├── README.md             # File này
├── eval_cache/           # Cache embeddings
├── eval_results/         # Kết quả evaluation
└── test_audio/           # Audio files cho testing
```

## 🎯 Models được đánh giá

### Speaker Diarization Models

1. **realtime_diarization_improved.py** - Whisper + SpeechBrain
2. **sen_voice.py** - SenseVoice only
3. **senvoi_spebrai_fixed.py** - SenseVoice + SpeechBrain

### ASR Models

1. **Whisper** (faster-whisper)
2. **SenseVoice** (FunASR)

## 📦 Cài đặt Dependencies

```bash
# Core packages
pip install numpy scipy scikit-learn matplotlib tqdm

# ASR evaluation
pip install jiwer regex librosa soundfile

# Speaker diarization
pip install torch torchaudio speechbrain

# Models (optional)
pip install faster-whisper funasr
```

## 🚀 Sử dụng

### 1. Tạo Dataset

#### Option A: Tạo sample dataset

```bash
cd realtime/evaluation
python create_dataset.py --mode sample
```

#### Option B: Từ folder audio

```bash
python create_dataset.py --mode audio --audio_dir <path_to_audio> --output dataset.csv
```

#### Option C: Từ JSON outputs có sẵn

```bash
python create_dataset.py --mode json --output dataset_from_json.csv
```

### 2. Đánh giá Speaker Diarization

```bash
# Basic usage
python eval_diarization.py --data_dir <path_to_speaker_folders> --model speechbrain

# Full options
python eval_diarization.py \
  --data_dir ./test_audio \
  --model speechbrain \
  --max_genuine 50 \
  --max_impostor 100 \
  --use_cache \
  --output_name my_eval
```

**Tham số:**

- `--data_dir`: Thư mục chứa audio files (phân theo speaker)
- `--model`: Model type (hiện tại hỗ trợ: speechbrain)
- `--max_genuine`: Số cặp genuine pairs tối đa mỗi speaker (default: 50)
- `--max_impostor`: Số cặp impostor pairs mỗi speaker (default: 100)
- `--use_cache`: Sử dụng cache cho embeddings
- `--output_name`: Tên file output

**Kết quả:**

- JSON file với metrics: EER, FAR, FRR, Precision, Recall, F1, AUC
- PNG plots: ROC curve và Precision-Recall curve
- Cache embeddings để chạy lại nhanh hơn

**Metrics giải thích:**

- **EER (Equal Error Rate)**: Điểm mà FAR = FRR, càng thấp càng tốt
- **FAR (False Acceptance Rate)**: Tỷ lệ chấp nhận nhầm người khác
- **FRR (False Rejection Rate)**: Tỷ lệ từ chối nhầm cùng người
- **ROC AUC**: Area Under ROC Curve, càng gần 1.0 càng tốt
- **PR AUC**: Area Under PR Curve, đánh giá hiệu suất tổng thể

### 3. Đánh giá ASR Quality

```bash
# Whisper
python eval_asr.py \
  --dataset dataset.csv \
  --model whisper \
  --whisper_size small \
  --device cpu \
  --compute_type int8

# SenseVoice
python eval_asr.py \
  --dataset dataset.csv \
  --model sensevoice \
  --device cpu
```

**Tham số:**

- `--dataset`: File CSV chứa dataset
- `--model`: whisper hoặc sensevoice
- `--whisper_size`: tiny, base, small, medium, large, large-v3
- `--device`: cpu hoặc cuda
- `--compute_type`: int8, float16, float32 (cho Whisper)
- `--output`: Đường dẫn file output (tự động nếu không chỉ định)

**Kết quả:**

- JSON file với summary và detailed results
- CSV file để dễ xem trong Excel
- Metrics: WER, CER, RTF cho từng sample

**Metrics giải thích:**

- **WER (Word Error Rate)**: Tỷ lệ lỗi từ, càng thấp càng tốt (0.0 = perfect)
- **CER (Character Error Rate)**: Tỷ lệ lỗi ký tự
- **RTF (Real-Time Factor)**: < 1.0 = nhanh hơn realtime, > 1.0 = chậm hơn

## 📊 Format Dataset

### CSV cho ASR Evaluation

```csv
file_name,file_path,transcript
sample1,/path/to/audio1.wav,This is the transcript
sample2,/path/to/audio2.wav,Another transcript here
```

### Folder structure cho Diarization

```
test_audio/
├── speaker1/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── audio3.wav
├── speaker2/
│   ├── audio1.wav
│   └── audio2.wav
└── speaker3/
    └── audio1.wav
```

## 🎯 Ví dụ thực tế

### Example 1: Đánh giá nhanh với sample data

```bash
# Tạo sample dataset
python create_dataset.py --mode sample

# Đánh giá (sau khi có audio trong test_audio/)
python eval_diarization.py --data_dir test_audio --model speechbrain
python eval_asr.py --dataset sample_dataset.csv --model whisper --device cpu
```

### Example 2: Đánh giá trên dataset lớn với GPU

```bash
# Diarization với cache
python eval_diarization.py \
  --data_dir /data/jvs_ver1 \
  --model speechbrain \
  --max_genuine 100 \
  --max_impostor 200 \
  --use_cache \
  --output_name jvs_full_eval

# ASR với Whisper large
python eval_asr.py \
  --dataset dataset_400_testcases.csv \
  --model whisper \
  --whisper_size large-v3 \
  --device cuda \
  --compute_type float16
```

### Example 3: So sánh nhiều models

```bash
# Whisper small
python eval_asr.py --dataset dataset.csv --model whisper --whisper_size small --output eval_whisper_small.json

# Whisper medium
python eval_asr.py --dataset dataset.csv --model whisper --whisper_size medium --output eval_whisper_medium.json

# SenseVoice
python eval_asr.py --dataset dataset.csv --model sensevoice --output eval_sensevoice.json

# So sánh kết quả trong folder eval_results/
```

## 📈 Đọc kết quả

### Diarization Results (JSON)

```json
{
  "eer": 0.0523,
  "threshold_at_eer": 0.6234,
  "far_at_eer": 0.0523,
  "frr_at_eer": 0.0523,
  "best_f1": 0.9512,
  "roc_auc": 0.9876,
  "pr_auc": 0.9823
}
```

**Giải thích:**

- EER = 5.23% → Model phân biệt tốt speakers (càng thấp càng tốt)
- Best F1 = 0.95 → Hiệu suất tốt
- ROC AUC = 0.99 → Model rất tốt (gần 1.0 là perfect)

### ASR Results (JSON)

```json
{
  "summary": {
    "avg_wer": 0.1234,
    "avg_cer": 0.0567,
    "avg_rtf": 0.45
  }
}
```

**Giải thích:**

- WER = 12.34% → Lỗi từ khá thấp (tốt)
- CER = 5.67% → Lỗi ký tự rất thấp
- RTF = 0.45 → Xử lý nhanh hơn realtime 2.2x (tốt cho realtime)

## 🔧 Troubleshooting

### Lỗi thiếu packages

```bash
pip install speechbrain torch torchaudio soundfile
pip install jiwer regex librosa
pip install faster-whisper funasr
```

### Lỗi CUDA/GPU

```bash
# Dùng CPU thay vì
python eval_asr.py --dataset dataset.csv --model whisper --device cpu --compute_type int8
```

### Lỗi cache

```bash
# Xóa cache và chạy lại
rm -rf eval_cache/*
python eval_diarization.py --data_dir test_audio --model speechbrain
```

### Lỗi không tìm thấy audio files

- Kiểm tra đường dẫn trong CSV file
- Đảm bảo audio files tồn tại
- Sử dụng absolute paths

## 📝 Notes

- **Cache embeddings**: Giúp chạy lại nhanh hơn nhiều, đặc biệt với dataset lớn
- **RTF < 1.0**: Cần thiết cho ứng dụng realtime
- **EER < 5%**: Model speaker verification tốt
- **WER < 15%**: Transcription chất lượng cao

## 🔗 Related Files

- `../realtime_diarization_improved.py` - Model chính
- `../sen_voice.py` - SenseVoice model
- `../senvoi_spebrai_fixed.py` - Hybrid model
- `../../repo/realtime-transcript/backend/eval/` - Evaluation code tham khảo

## 📧 Support

Nếu gặp vấn đề, kiểm tra:

1. Dependencies đã cài đúng chưa
2. Audio files có đúng format không (WAV, 16kHz preferred)
3. Dataset CSV có đúng format không
4. GPU memory đủ không (nếu dùng CUDA)

---

**Tạo bởi:** VJ Speaker Diarization Evaluation System
**Version:** 1.0
**Last updated:** November 28, 2025
