# NeMo Diarization Finetuning và Evaluation Pipeline

## 📋 Tổng quan

Pipeline đầy đủ để finetune và đánh giá NeMo Diarization models trên Modal cloud, bao gồm:

1. **Upload datasets** lên Modal cloud
2. **Finetune 3 variants**:
   - VAD only
   - Speaker Embedding only
   - Full Pipeline (VAD + Speaker + Clustering)
3. **Evaluation tổng hợp** so sánh với pretrained models

## 📁 Files mới

### 1. `upload_nemo_datasets.py`

Upload datasets lên Modal cloud với cấu trúc train/test

**Datasets:**

- **Training**: voxconverse_dev, jvs_ver1, callhome_eng (70%)
- **Testing**: voxconverse_test, callhome_jpn, callhome_eng (30%)

**Usage:**

```bash
modal run upload_nemo_datasets.py
```

---

### 2. `finetune_nemo_vad.py`

Finetune **chỉ VAD model** (MarbleNet) cho diarization

**Features:**

- Sử dụng pretrained MarbleNet VAD
- Train trên speech/non-speech segments từ RTTM
- Sliding window với augmentation
- Early stopping + checkpoint best model

**Usage:**

```bash
# Default: 50 epochs, batch_size=32
modal run finetune_nemo_vad.py

# Custom parameters
modal run finetune_nemo_vad.py --epochs 100 --batch-size 64 --learning-rate 1e-4

# Resume từ checkpoint
modal run finetune_nemo_vad.py --resume best_vad_model.pt
```

**Output:**

- `/results/checkpoints/vad/best_vad_model.nemo`
- `/results/training_summary_vad.json`

---

### 3. `finetune_nemo_speaker_new.py`

Finetune **chỉ Speaker Embedding model** (TitaNet) cho diarization

**Features:**

- Sử dụng pretrained TitaNet Large
- Train trên speaker segments từ RTTM
- Speaker classification với augmentation
- Multi-speaker learning (auto-detect unique speakers)

**Usage:**

```bash
# Default: 30 epochs, batch_size=64
modal run finetune_nemo_speaker_new.py

# Custom parameters
modal run finetune_nemo_speaker_new.py --epochs 50 --batch-size 128

# Resume từ checkpoint
modal run finetune_nemo_speaker_new.py --resume best_speaker_model.nemo
```

**Output:**

- `/results/checkpoints/speaker/best_speaker_model.nemo`
- `/results/training_summary_speaker.json`

---

### 4. `finetune_nemo_full_pipeline.py`

Finetune **toàn bộ NeMo diarization pipeline** (VAD + Speaker + Clustering)

**Features:**

- Joint training của VAD + Speaker models
- End-to-end optimization với DER loss
- Cosine annealing learning rate
- Validation sau mỗi epoch với DER metric

**Usage:**

```bash
# Default: 40 epochs, batch_size=16
modal run finetune_nemo_full_pipeline.py

# Custom parameters
modal run finetune_nemo_full_pipeline.py --epochs 60 --batch-size 32 --learning-rate 5e-5

# Resume từ checkpoint
modal run finetune_nemo_full_pipeline.py --resume best_pipeline_model.pt
```

**Output:**

- `/results/checkpoints/full_pipeline/best_vad_model.nemo`
- `/results/checkpoints/full_pipeline/best_speaker_model.nemo`
- `/results/checkpoints/full_pipeline/best_pipeline_model_epoch*.pt`
- `/results/training_summary_full_pipeline.json`

---

### 5. `eval_nemo_diarization.py`

Đánh giá tổng hợp **tất cả models** trên **tất cả test datasets**

**Models được đánh giá:**

1. ✅ Pretrained NeMo (baseline)
2. ✅ Pyannote 3.1 (baseline)
3. ✅ Fine-tuned VAD only
4. ✅ Fine-tuned Speaker only
5. ✅ Fine-tuned Full Pipeline

**Metrics đầy đủ:**

- **DER** (Diarization Error Rate)
- **JER** (Jaccard Error Rate)
- **FA** (False Alarm rate)
- **Miss** (Missed Speech rate)
- **EER** (Equal Error Rate)
- **FAR** (False Acceptance Rate)
- **FRR** (False Rejection Rate)
- **AUC** (Area Under ROC Curve)
- **F1**, Precision, Recall

**Usage:**

```bash
# Evaluate all models on all datasets
modal run eval_nemo_diarization.py --models all --dataset all

# Evaluate specific models
modal run eval_nemo_diarization.py --models pretrained_nemo,pyannote,vad

# Evaluate on specific dataset
modal run eval_nemo_diarization.py --dataset voxconverse_test

# Custom combination
modal run eval_nemo_diarization.py --models vad,speaker,full_pipeline --dataset callhome_eng_test
```

**Output:**

- `/results/evaluation_YYYYMMDD_HHMMSS/detailed_results.json`
- `/results/evaluation_YYYYMMDD_HHMMSS/comparison_table.csv`
- `/results/evaluation_YYYYMMDD_HHMMSS/der_comparison.png`
- `/results/evaluation_YYYYMMDD_HHMMSS/all_metrics_comparison.png`

---

## 🚀 Workflow đầy đủ

### Bước 1: Upload Datasets

```bash
modal run upload_nemo_datasets.py
```

**Kết quả:**

- Datasets uploaded lên Modal volume `nemo-dataset`
- Structure: `/dataset/train/` và `/dataset/test/`
- Callhome_eng được chia 70/30 tự động

---

### Bước 2: Finetune Models (song song hoặc tuần tự)

#### Option A: Chạy tuần tự (recommended)

```bash
# 1. Finetune VAD
modal run finetune_nemo_vad.py --epochs 50

# 2. Finetune Speaker
modal run finetune_nemo_speaker_new.py --epochs 30

# 3. Finetune Full Pipeline
modal run finetune_nemo_full_pipeline.py --epochs 40
```

#### Option B: Chạy song song (nếu có nhiều GPU credits)

Mở 3 terminals khác nhau và chạy đồng thời.

**Thời gian ước tính:**

- VAD: ~1-2 giờ
- Speaker: ~2-3 giờ
- Full Pipeline: ~3-4 giờ

---

### Bước 3: Evaluation và So sánh

```bash
# Evaluate all models
modal run eval_nemo_diarization.py --models all --dataset all
```

**Kết quả:**

- Bảng so sánh chi tiết (CSV + JSON)
- Biểu đồ so sánh (PNG)
- Per-file results cho từng model

---

## 📊 Cấu trúc Modal Volumes

### `/dataset` (nemo-dataset volume)

```
/dataset/
├── train/
│   ├── voxconverse_dev/
│   │   ├── audio/
│   │   └── rttm/
│   ├── jvs_ver1/
│   │   ├── audio/
│   │   └── rttm/
│   └── callhome_eng/          # 70% của tổng số files
│       ├── audio/
│       └── labels/
├── test/
│   ├── voxconverse_test/
│   ├── callhome_jpn/
│   └── callhome_eng/          # 30% của tổng số files
└── train_*.json               # Manifest files
```

### `/results` (nemo-results volume)

```
/results/
├── checkpoints/
│   ├── vad/
│   │   └── best_vad_model.nemo
│   ├── speaker/
│   │   └── best_speaker_model.nemo
│   └── full_pipeline/
│       ├── best_vad_model.nemo
│       ├── best_speaker_model.nemo
│       └── best_pipeline_model_epoch*.pt
├── training_summary_vad.json
├── training_summary_speaker.json
├── training_summary_full_pipeline.json
└── evaluation_YYYYMMDD_HHMMSS/
    ├── detailed_results.json
    ├── comparison_table.csv
    └── *.png (plots)
```

---

## 🎯 Expected Results

### Baseline (Pretrained)

- **DER**: ~25-35% (callhome), ~15-25% (voxconverse)

### After Finetuning

- **VAD only**: DER cải thiện 5-10%
- **Speaker only**: DER cải thiện 10-15%
- **Full Pipeline**: DER cải thiện 15-25%

---

## 🔧 Troubleshooting

### 1. Dataset không tìm thấy

```bash
# Kiểm tra Modal volume
modal volume ls nemo-dataset

# Re-upload nếu cần
modal run upload_nemo_datasets.py
```

### 2. Checkpoint không load được

```bash
# Kiểm tra checkpoints trong volume
modal volume ls nemo-results

# List files trong checkpoints/
modal run eval_nemo_diarization.py --models pretrained_nemo,pyannote
```

### 3. Out of Memory

- Giảm batch_size
- Sử dụng GPU lớn hơn (A100 thay vì A10G)

### 4. Training quá chậm

- Tăng batch_size nếu GPU memory còn
- Giảm số epochs cho quick test
- Sử dụng subset nhỏ hơn của dataset

---

## 📝 Notes

### JVS Dataset (Japanese)

- 100 speakers tiếng Nhật
- Converted to audio/ + rttm/ format
- Mỗi speaker có ~100 utterances từ parallel100 folder

### Callhome Split

- Sử dụng sorted file list để đảm bảo reproducibility
- Seed=42 cho consistency
- Split 70/30 based on file index

### Metrics Explanation

- **DER**: Tổng error (Miss + FA + Confusion)
- **JER**: Jaccard similarity error
- **FA**: False alarm (non-speech classified as speech)
- **Miss**: Missed speech (speech classified as non-speech)
- **EER**: Equal Error Rate cho speaker verification

---

## 🙏 Credits

- **NeMo**: NVIDIA NeMo Toolkit
- **Pyannote**: pyannote.audio 3.1
- **Datasets**: Voxconverse, JVS, Callhome

---

## 📧 Contact

Nếu có vấn đề hoặc câu hỏi, hãy báo lại để tôi hỗ trợ!
