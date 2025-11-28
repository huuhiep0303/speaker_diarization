# Quick Start Guide

## 🚀 Bắt đầu nhanh trong 3 bước

### Bước 1: Cài đặt dependencies

```bash
# Windows
setup.bat

# Hoặc manual
pip install numpy scipy scikit-learn matplotlib tqdm jiwer regex librosa soundfile torch speechbrain
```

### Bước 2: Tạo sample dataset

```bash
python create_dataset.py --mode sample
```

### Bước 3: Chạy evaluation

#### Đánh giá Speaker Diarization

```bash
# Windows
eval_diarization.bat test_audio speechbrain

# Linux/Mac
python eval_diarization.py --data_dir test_audio --model speechbrain
```

#### Đánh giá ASR

```bash
# Windows
eval_asr.bat sample_dataset.csv whisper cpu

# Linux/Mac
python eval_asr.py --dataset sample_dataset.csv --model whisper --device cpu
```

## 📊 Xem kết quả

Kết quả được lưu trong folder `eval_results/`:

- JSON files: Chi tiết metrics
- CSV files: Dữ liệu tabular
- PNG files: Visualization plots

## 🎯 Ví dụ với dữ liệu thực

```bash
# 1. Download JVS dataset (hoặc dataset khác)
# 2. Giải nén vào test_audio/
# 3. Chạy evaluation

python eval_diarization.py \
  --data_dir test_audio \
  --model speechbrain \
  --max_genuine 100 \
  --max_impostor 200 \
  --use_cache

# Kết quả sẽ có EER, F1, ROC curves, etc.
```

## 📖 Đọc thêm

Xem `README.md` để biết chi tiết đầy đủ về:

- Metrics giải thích
- Advanced usage
- Troubleshooting
- Format dataset

---

**Happy Evaluating! 🎉**
