# 📊 Hệ thống Evaluation - Tổng quan

## ✅ Đã hoàn thành

Đã tạo hệ thống evaluation hoàn chỉnh cho các models trong folder `realtime/`:

### 1. **create_dataset.py** - Tạo dataset từ JVS Corpus

- ✅ Hỗ trợ JVS dataset structure
- ✅ Random sampling từ 4 categories (parallel100, nonpara30, whisper10, falset10)
- ✅ Export CSV với format chuẩn
- ✅ Thống kê dataset

### 2. **eval_asr.py** - Đánh giá ASR Quality

- ✅ Hỗ trợ Whisper (faster-whisper)
- ✅ Hỗ trợ SenseVoice (FunASR)
- ✅ Japanese tokenization với Sudachi (WER tính theo từ)
- ✅ Metrics: WER, CER, RTF
- ✅ Checkpoint mechanism (auto-resume)
- ✅ Output: CSV chi tiết + JSON summary

### 3. **eval_diarization.py** - Đánh giá Speaker Verification

- ✅ Hỗ trợ SpeechBrain ECAPA-TDNN
- ✅ Genuine/Impostor trials
- ✅ Metrics: EER, FAR, FRR, F1, AUC
- ✅ Embedding cache
- ✅ ROC & PR curves visualization

### 4. **Documentation**

- ✅ README.md - Hướng dẫn tổng quan
- ✅ RUN_EVALUATION.md - Hướng dẫn chạy chi tiết
- ✅ QUICKSTART_EVALUATION.md - Quick start 3 bước
- ✅ Batch scripts cho Windows

### 5. **Folder Structure**

```
evaluation/
├── create_dataset.py
├── eval_asr.py
├── eval_diarization.py
├── eval_asr.bat
├── eval_diarization.bat
├── setup.bat
├── README.md
├── RUN_EVALUATION.md
├── QUICKSTART_EVALUATION.md
├── eval_cache/          # Cache embeddings
├── eval_results/        # Kết quả evaluation
└── test_audio/         # Audio files (optional)
```

---

## 🎯 Models được đánh giá

### ASR Models

1. **Whisper** (faster-whisper)

   - Sizes: tiny, base, small, medium, large, large-v3, turbo
   - Device: CPU, CUDA
   - Compute type: int8, float16, float32

2. **SenseVoice** (FunASR)
   - Model: FunAudioLLM/SenseVoiceSmall
   - Device: CPU, CUDA
   - Optimized cho tiếng Nhật

### Diarization Models

1. **SpeechBrain ECAPA-TDNN**
   - Model: speechbrain/spkrec-ecapa-voxceleb
   - Speaker embedding extraction
   - Cosine similarity for verification

---

## 📋 Cách sử dụng

### Quick Start (3 bước)

```bash
# 1. Tạo dataset
python create_dataset.py --jvs_root ../dataset/jvs_ver1

# 2. Chạy ASR evaluation
python eval_asr.py --dataset dataset_400_testcases.csv --model whisper --device cpu

# 3. Xem kết quả
cat eval_results/eval_whisper-small_summary.json
```

### Windows Users

```cmd
REM ASR Evaluation
eval_asr.bat dataset_400_testcases.csv whisper cpu

REM Diarization Evaluation
eval_diarization.bat ..\dataset\jvs_ver1 speechbrain
```

---

## 📊 Output Files

### ASR Evaluation

```
eval_results/
├── eval_whisper-small_checkpoint.csv    # Chi tiết từng sample
└── eval_whisper-small_summary.json      # Thống kê tổng hợp
```

**Checkpoint CSV format:**

```csv
file_path,ground_truth,prediction,wer,cer,rtf,audio_duration,processing_time
```

**Summary JSON format:**

```json
{
  "model": "whisper-small",
  "num_samples": 400,
  "avg_wer": 0.1234,
  "avg_cer": 0.0567,
  "avg_rtf": 0.45,
  "median_rtf": 0.42
}
```

### Diarization Evaluation

```
eval_results/
├── eval_results_speechbrain.json        # Metrics
└── eval_results_speechbrain_curves.png  # ROC & PR plots
```

**Metrics JSON format:**

```json
{
  "eer": 0.0523,
  "threshold_at_eer": 0.6234,
  "far_at_eer": 0.0523,
  "frr_at_eer": 0.0523,
  "best_f1": 0.9512,
  "roc_auc": 0.9876
}
```

---

## 🔧 Tính năng chính

### 1. Checkpoint Auto-Resume

- ✅ Tự động lưu progress sau mỗi sample
- ✅ Có thể dừng và resume bất cứ lúc nào
- ✅ Không mất công việc đã làm

### 2. Japanese Text Processing

- ✅ Sudachi tokenizer cho WER calculation
- ✅ Text normalization cho tiếng Nhật
- ✅ Remove punctuation, tags, spaces

### 3. Embedding Cache

- ✅ Cache speaker embeddings
- ✅ Chạy lại nhanh hơn nhiều lần
- ✅ Tự động detect cache

### 4. Comprehensive Metrics

**ASR:**

- WER (Word Error Rate) - với Japanese tokenization
- CER (Character Error Rate)
- RTF (Real-Time Factor)

**Diarization:**

- EER (Equal Error Rate)
- FAR/FRR (False Accept/Reject Rate)
- F1, Precision, Recall
- ROC AUC, PR AUC

---

## 📈 Expected Results

### JVS Dataset (400 samples)

**Whisper Small (CPU):**

- WER: ~8-12%
- CER: ~4-6%
- RTF: ~0.3-0.5 (2x faster than realtime)
- Time: ~30-40 minutes

**SenseVoice (CPU):**

- WER: ~10-15%
- CER: ~5-8%
- RTF: ~0.2-0.3 (3x faster than realtime)
- Time: ~20-30 minutes

**SpeechBrain Diarization:**

- EER: ~2-5%
- ROC AUC: ~0.97-0.99
- Time: ~20-30 minutes (with cache)

---

## 🚀 Next Steps

### 1. Chạy Evaluation

```bash
cd realtime/evaluation

# Tạo dataset
python create_dataset.py --jvs_root ../dataset/jvs_ver1

# Đánh giá Whisper
python eval_asr.py --dataset dataset_400_testcases.csv --model whisper --device cpu

# Đánh giá SenseVoice
python eval_asr.py --dataset dataset_400_testcases.csv --model sensevoice --device cpu

# Đánh giá Diarization
python eval_diarization.py --data_dir ../dataset/jvs_ver1 --model speechbrain --use_cache
```

### 2. So sánh kết quả

```bash
# Xem tất cả summary files
ls eval_results/*_summary.json

# So sánh metrics
cat eval_results/eval_whisper-small_summary.json
cat eval_results/eval_sensevoice_summary.json
```

### 3. Chọn model tốt nhất

- **Whisper Small**: Balance tốt (WER ~10%, RTF ~0.4)
- **Whisper Large-v3**: Chất lượng cao nhất (WER ~6%, RTF ~0.8)
- **SenseVoice**: Nhanh nhất (WER ~12%, RTF ~0.3)

### 4. Deploy vào production

- Integrate model được chọn vào `realtime_diarization_improved.py`
- Test với real-world audio
- Monitor performance

---

## 📚 Documentation

**Đọc thêm:**

- `README.md` - Tổng quan hệ thống
- `RUN_EVALUATION.md` - Hướng dẫn chi tiết từng bước
- `QUICKSTART_EVALUATION.md` - Quick start 3 bước
- `test_audio/README.md` - Hướng dẫn chuẩn bị audio

**Troubleshooting:**

- Check `RUN_EVALUATION.md` section "Troubleshooting"
- Xem error messages trong terminal
- Kiểm tra checkpoint files trong `eval_results/`

---

## ✨ Features

- ✅ **Easy to use** - Chỉ cần 3 lệnh
- ✅ **Auto-resume** - Checkpoint mechanism
- ✅ **Japanese support** - Sudachi tokenizer
- ✅ **Comprehensive metrics** - WER, CER, RTF, EER, AUC
- ✅ **Visualization** - ROC & PR curves
- ✅ **Cache embeddings** - Fast re-evaluation
- ✅ **Windows support** - Batch scripts
- ✅ **Well documented** - Multiple README files

---

## 🎉 Summary

Hệ thống evaluation hoàn chỉnh đã sẵn sàng để:

1. ✅ Đánh giá ASR quality (Whisper, SenseVoice)
2. ✅ Đánh giá Speaker Diarization (SpeechBrain)
3. ✅ So sánh nhiều models
4. ✅ Chọn model tốt nhất cho production

**Bắt đầu ngay:** Xem `QUICKSTART_EVALUATION.md`

**Good luck with your evaluation! 🚀**

---

**Created:** November 28, 2025  
**Author:** VJ Speaker Diarization Team  
**Version:** 1.0
