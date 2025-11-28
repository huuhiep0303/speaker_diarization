# 🚀 Hướng dẫn chạy Evaluation

## 📋 Chuẩn bị

### 1. Cài đặt Dependencies

```bash
cd realtime/evaluation

# Core packages
pip install numpy scipy scikit-learn matplotlib tqdm

# Japanese text processing
pip install sudachipy

# ASR evaluation
pip install jiwer regex librosa soundfile

# Speaker diarization
pip install torch torchaudio speechbrain

# ASR models
pip install faster-whisper  # Whisper
pip install funasr         # SenseVoice
```

### 2. Kiểm tra Dataset

Dataset JVS đã có sẵn trong `realtime/dataset/jvs_ver1/`:

```
dataset/jvs_ver1/
├── jvs001/
│   ├── parallel100/
│   │   ├── transcripts_utf8.txt
│   │   └── wav24kHz16bit/
│   │       └── *.wav
│   ├── nonpara30/
│   ├── whisper10/
│   └── falset10/
├── jvs002/
└── ...
```

## 📊 Bước 1: Tạo Dataset CSV

Tạo file CSV chứa danh sách test cases từ JVS corpus:

```bash
# Tạo dataset với 1 sample/category/speaker (400 samples)
python create_dataset.py --jvs_root ../dataset/jvs_ver1 --output dataset_400_testcases.csv

# Hoặc tạo nhiều samples hơn
python create_dataset.py --jvs_root ../dataset/jvs_ver1 --samples_per_category 2 --output dataset_800_testcases.csv
```

**Output:** File CSV với format:

```csv
speaker,category,file_name,wav_path,transcript
jvs001,parallel100,VOICEACTRESS100_069,dataset/jvs_ver1/jvs001/parallel100/wav24kHz16bit/VOICEACTRESS100_069.wav,ブルーリッジ山脈の源流から...
```

## 🎤 Bước 2: Đánh giá ASR Quality

### Option A: Đánh giá Whisper

```bash
# Whisper Small on CPU (khuyến nghị để test nhanh)
python eval_asr.py --dataset dataset_400_testcases.csv --model whisper --whisper_size small --device cpu --compute_type int8

# Whisper Small on GPU
python eval_asr.py --dataset dataset_400_testcases.csv --model whisper --whisper_size small --device cuda --compute_type float16

# Whisper Large-v3 on GPU (chất lượng cao nhất)
python eval_asr.py --dataset dataset_400_testcases.csv --model whisper --whisper_size large-v3 --device cuda --compute_type float16
```

### Option B: Đánh giá SenseVoice

```bash
# SenseVoice on CPU
python eval_asr.py --dataset dataset_400_testcases.csv --model sensevoice --device cpu

# SenseVoice on GPU
python eval_asr.py --dataset dataset_400_testcases.csv --model sensevoice --device cuda
```

### Option C: Windows Batch Scripts

```cmd
REM Whisper
eval_asr.bat dataset_400_testcases.csv whisper cpu

REM SenseVoice
eval_asr.bat dataset_400_testcases.csv sensevoice cpu
```

### ⏸️ Resume từ Checkpoint

Nếu quá trình bị gián đoạn, chỉ cần chạy lại lệnh tương tự:

```bash
# Script tự động resume từ checkpoint
python eval_asr.py --dataset dataset_400_testcases.csv --model whisper --resume
```

### 📈 Kết quả ASR

Kết quả được lưu trong `eval_results/`:

**Checkpoint CSV** (`eval_whisper-small_checkpoint.csv`):

```csv
file_path,ground_truth,prediction,wer,cer,rtf,audio_duration,processing_time
dataset/.../audio1.wav,ブルーリッジ山脈...,ブルーリッジ山脈...,0.0523,0.0234,0.45,5.2,2.34
```

**Summary JSON** (`eval_whisper-small_summary.json`):

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

**Metrics giải thích:**

- **WER** (Word Error Rate): Tỷ lệ lỗi từ - càng thấp càng tốt
  - < 10%: Excellent
  - 10-20%: Good
  - 20-30%: Fair
  - \> 30%: Poor
- **CER** (Character Error Rate): Tỷ lệ lỗi ký tự - cho tiếng Nhật

  - < 5%: Excellent
  - 5-10%: Good
  - 10-15%: Fair
  - \> 15%: Poor

- **RTF** (Real-Time Factor): Tốc độ xử lý
  - < 1.0: Nhanh hơn realtime ✓ (ví dụ: 0.5 = 2x nhanh hơn)
  - = 1.0: Đúng realtime
  - \> 1.0: Chậm hơn realtime ✗

## 👥 Bước 3: Đánh giá Speaker Diarization

### Chạy Evaluation

```bash
# Đánh giá với SpeechBrain ECAPA-TDNN
python eval_diarization.py --data_dir ../dataset/jvs_ver1 --model speechbrain --max_genuine 50 --max_impostor 100 --use_cache

# Windows
eval_diarization.bat ..\dataset\jvs_ver1 speechbrain
```

**Tham số:**

- `--data_dir`: Thư mục JVS (sẽ quét tất cả speakers)
- `--model`: speechbrain (hiện tại chỉ hỗ trợ SpeechBrain)
- `--max_genuine`: Số cặp genuine pairs/speaker (cùng người)
- `--max_impostor`: Số cặp impostor pairs/speaker (khác người)
- `--use_cache`: Lưu embeddings cache để chạy lại nhanh hơn

### 📈 Kết quả Diarization

Kết quả được lưu trong `eval_results/`:

**JSON** (`eval_results_speechbrain.json`):

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

**Plots:**

- `eval_results_speechbrain_curves.png`: ROC và PR curves

**Metrics giải thích:**

- **EER** (Equal Error Rate): Tỷ lệ lỗi khi FAR = FRR
  - < 1%: Excellent
  - 1-5%: Good
  - 5-10%: Fair
  - \> 10%: Poor
- **FAR** (False Accept Rate): Chấp nhận nhầm người khác
- **FRR** (False Reject Rate): Từ chối nhầm cùng người
- **ROC AUC**: Diện tích dưới ROC curve (gần 1.0 = tốt)
- **Best F1**: F1-score tốt nhất (balance giữa precision và recall)

## 🔄 So sánh nhiều Models

### ASR Comparison

```bash
# Whisper Small
python eval_asr.py --dataset dataset_400_testcases.csv --model whisper --whisper_size small --device cpu

# Whisper Medium
python eval_asr.py --dataset dataset_400_testcases.csv --model whisper --whisper_size medium --device cpu

# SenseVoice
python eval_asr.py --dataset dataset_400_testcases.csv --model sensevoice --device cpu

# So sánh kết quả
ls eval_results/eval_*_summary.json
```

### Compare Results

Mở các file JSON trong `eval_results/` và so sánh:

| Model          | WER   | CER   | RTF  |
| -------------- | ----- | ----- | ---- |
| whisper-small  | 0.123 | 0.056 | 0.45 |
| whisper-medium | 0.098 | 0.043 | 0.78 |
| sensevoice     | 0.145 | 0.068 | 0.32 |

## 🛠️ Troubleshooting

### Lỗi thiếu packages

```bash
pip install sudachipy jiwer regex librosa soundfile
pip install faster-whisper funasr
pip install torch torchaudio speechbrain
```

### Lỗi CUDA out of memory

```bash
# Dùng CPU hoặc model nhỏ hơn
python eval_asr.py --dataset dataset_400_testcases.csv --model whisper --whisper_size tiny --device cpu
```

### Lỗi không tìm thấy audio files

```bash
# Kiểm tra đường dẫn trong CSV
head dataset_400_testcases.csv

# Đảm bảo đường dẫn tương đối đúng
cd realtime/evaluation
python eval_asr.py --dataset dataset_400_testcases.csv --model whisper
```

### Lỗi Japanese tokenizer

```bash
# Cài đặt Sudachi dictionary
pip install sudachipy
python -m sudachipy link -t full
```

### Xóa cache và chạy lại

```bash
# Xóa cache embeddings
rm -rf eval_cache/*

# Xóa checkpoint để chạy lại từ đầu
rm eval_results/eval_*_checkpoint.csv
```

## 📊 Expected Performance

### JVS Dataset (400 samples)

**Whisper Small (CPU, int8):**

- WER: ~8-12%
- CER: ~4-6%
- RTF: ~0.3-0.5
- Time: ~30-40 minutes

**Whisper Large-v3 (GPU, float16):**

- WER: ~5-8%
- CER: ~2-4%
- RTF: ~0.6-0.8
- Time: ~45-60 minutes

**SenseVoice (CPU):**

- WER: ~10-15%
- CER: ~5-8%
- RTF: ~0.2-0.3
- Time: ~20-30 minutes

**SpeechBrain Diarization:**

- EER: ~2-5%
- Time: ~20-30 minutes (with cache)

## 💡 Tips

1. **Bắt đầu với dataset nhỏ** để test:

   ```bash
   # Chỉ 100 samples
   python create_dataset.py --jvs_root ../dataset/jvs_ver1 --samples_per_category 0.25 --output dataset_100_test.csv
   ```

2. **Dùng CPU cho test nhanh**, GPU cho production
3. **Cache embeddings** giúp evaluation nhanh hơn nhiều
4. **Checkpoint tự động** - có thể dừng và resume bất cứ lúc nào
5. **So sánh nhiều models** để chọn model tốt nhất cho use case

## 📝 Next Steps

Sau khi có kết quả evaluation:

1. **Chọn model tốt nhất** dựa trên WER/CER và RTF
2. **Fine-tune parameters** nếu cần (beam size, temperature, etc.)
3. **Test với real-world audio** từ ứng dụng của bạn
4. **Deploy model** vào production

---

**Happy Evaluating! 🎉**

Nếu cần hỗ trợ, check file `README.md` hoặc `QUICKSTART.md` trong folder này.
