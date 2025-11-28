# ⚡ QUICK START - Chạy Evaluation trong 3 bước

## Bước 1: Tạo Dataset (1 phút)

```bash
cd realtime/evaluation

# Tạo dataset 400 test cases từ JVS
python create_dataset.py --jvs_root ../dataset/jvs_ver1
```

**Output:** `dataset_400_testcases.csv`

---

## Bước 2: Chạy ASR Evaluation

### Windows:

```cmd
eval_asr.bat dataset_400_testcases.csv whisper cpu
```

### Linux/Mac:

```bash
python eval_asr.py --dataset dataset_400_testcases.csv --model whisper --device cpu
```

**Thời gian:** ~30-40 phút cho 400 samples

**Có thể dừng và resume bất cứ lúc nào!**

---

## Bước 3: Xem kết quả

```bash
# Mở file CSV trong Excel
cd eval_results
start eval_whisper-small_checkpoint.csv

# Hoặc xem summary JSON
cat eval_whisper-small_summary.json
```

**Kết quả:**

- WER: ~8-12% (Word Error Rate)
- CER: ~4-6% (Character Error Rate)
- RTF: ~0.3-0.5 (Real-Time Factor - nhanh hơn 2x)

---

## Bonus: Đánh giá SenseVoice

```bash
# Windows
eval_asr.bat dataset_400_testcases.csv sensevoice cpu

# Linux/Mac
python eval_asr.py --dataset dataset_400_testcases.csv --model sensevoice --device cpu
```

---

## Bonus: Đánh giá Speaker Diarization

```bash
# Windows
eval_diarization.bat ..\dataset\jvs_ver1 speechbrain

# Linux/Mac
python eval_diarization.py --data_dir ../dataset/jvs_ver1 --model speechbrain --use_cache
```

**Thời gian:** ~20-30 phút với cache

---

## 🎯 So sánh Models

| Model          | WER  | RTF | Thời gian |
| -------------- | ---- | --- | --------- |
| Whisper Small  | ~10% | 0.4 | 35 phút   |
| Whisper Medium | ~8%  | 0.7 | 55 phút   |
| SenseVoice     | ~12% | 0.3 | 25 phút   |

**Khuyến nghị:**

- **Whisper Small** - Balance tốt giữa chất lượng và tốc độ
- **SenseVoice** - Nhanh nhất, tốt cho realtime

---

## 📁 Files quan trọng

```
evaluation/
├── create_dataset.py           # Tạo dataset
├── eval_asr.py                # Đánh giá ASR
├── eval_diarization.py        # Đánh giá diarization
├── eval_results/              # Kết quả
│   ├── eval_*_checkpoint.csv  # Chi tiết từng sample
│   └── eval_*_summary.json    # Thống kê tổng hợp
└── RUN_EVALUATION.md          # Hướng dẫn chi tiết
```

---

## 🆘 Gặp lỗi?

### Thiếu packages:

```bash
pip install sudachipy jiwer regex librosa
pip install faster-whisper funasr
```

### Audio file không tìm thấy:

```bash
# Kiểm tra dataset path
python create_dataset.py --jvs_root ../dataset/jvs_ver1

# Đảm bảo đang ở folder evaluation/
cd realtime/evaluation
```

### CUDA out of memory:

```bash
# Dùng CPU
eval_asr.bat dataset_400_testcases.csv whisper cpu
```

---

**Xem hướng dẫn đầy đủ:** `RUN_EVALUATION.md`

**Happy Evaluating! 🚀**
