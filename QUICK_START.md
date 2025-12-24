# ⚡ Quick Start Guide

## 🚀 Chạy ngay trong 3 bước

### Bước 1: Upload Datasets (10-15 phút)

```bash
modal run realtime/upload_nemo_datasets.py
```

**Kiểm tra upload thành công:**

```bash
modal volume ls nemo-dataset
```

---

### Bước 2: Chọn 1 trong 3 Training Modes

#### 🎯 Option A: VAD Only (nhanh nhất - 1-2h)

```bash
modal run realtime/finetune_nemo_vad.py --epochs 50
```

#### 🎯 Option B: Speaker Only (trung bình - 2-3h)

```bash
modal run realtime/finetune_nemo_speaker_new.py --epochs 30
```

#### 🎯 Option C: Full Pipeline (tốt nhất - 3-4h)

```bash
modal run realtime/finetune_nemo_full_pipeline.py --epochs 40
```

#### 🎯 Option D: Tất cả (recommended - chạy tuần tự)

```bash
# Chạy lần lượt
modal run realtime/finetune_nemo_vad.py --epochs 50
modal run realtime/finetune_nemo_speaker_new.py --epochs 30
modal run realtime/finetune_nemo_full_pipeline.py --epochs 40
```

---

### Bước 3: Evaluation và So sánh (30-60 phút)

```bash
modal run realtime/eval_nemo_diarization.py --models all --dataset all
```

**Xem kết quả:**

```bash
# Download results từ Modal
modal volume get nemo-results evaluation_*/comparison_table.csv .
modal volume get nemo-results evaluation_*/detailed_results.json .
modal volume get nemo-results evaluation_*/*.png .
```

---

## 📊 Kiểm tra Kết quả

### Trong Terminal:

- Training progress: TQDM progress bars
- Metrics: Printed sau mỗi epoch
- Best checkpoint: Auto-saved

### Trong Modal Dashboard:

1. Vào https://modal.com
2. Click vào "Volumes"
3. Browse `nemo-results` volume
4. Download files về local

---

## ⚙️ Custom Parameters

### Training nhanh (test):

```bash
modal run realtime/finetune_nemo_vad.py --epochs 5 --batch-size 16
```

### Training chậm (best quality):

```bash
modal run realtime/finetune_nemo_full_pipeline.py --epochs 100 --batch-size 32 --learning-rate 1e-5
```

### Evaluate chỉ 1 model:

```bash
modal run realtime/eval_nemo_diarization.py --models vad --dataset voxconverse_test
```

---

## 🔍 Troubleshooting

### Lỗi "Dataset not found":

```bash
# Re-upload datasets
modal run realtime/upload_nemo_datasets.py
```

### Lỗi "Out of Memory":

```bash
# Giảm batch size
modal run realtime/finetune_nemo_vad.py --epochs 50 --batch-size 16
```

### Lỗi "Checkpoint not found":

```bash
# Kiểm tra checkpoints
modal volume ls nemo-results

# Hoặc train lại model đó
modal run realtime/finetune_nemo_vad.py --epochs 50
```

---

## 📝 Files được tạo

### Sau Upload:

- `/dataset/train/*` (training data)
- `/dataset/test/*` (test data)
- `/dataset/upload_summary.json`

### Sau Training:

- `/results/checkpoints/vad/best_vad_model.nemo`
- `/results/checkpoints/speaker/best_speaker_model.nemo`
- `/results/checkpoints/full_pipeline/*.nemo`
- `/results/training_summary_*.json`

### Sau Evaluation:

- `/results/evaluation_*/detailed_results.json`
- `/results/evaluation_*/comparison_table.csv`
- `/results/evaluation_*/der_comparison.png`
- `/results/evaluation_*/all_metrics_comparison.png`

---

## 🎯 Expected Timeline

| Step             | Duration        | GPU Time   |
| ---------------- | --------------- | ---------- |
| Upload Datasets  | 10-15 min       | No GPU     |
| Finetune VAD     | 1-2 hours       | ~$1-2      |
| Finetune Speaker | 2-3 hours       | ~$2-4      |
| Finetune Full    | 3-4 hours       | ~$4-6      |
| Evaluation       | 30-60 min       | ~$0.5-1    |
| **TOTAL**        | **~7-10 hours** | **~$8-13** |

_Note: GPU costs are estimates based on Modal A10G pricing_

---

## ✅ Success Checklist

- [ ] Upload datasets thành công
- [ ] Training chạy không bị crash
- [ ] Checkpoints được save
- [ ] Evaluation chạy được
- [ ] Results files được tạo
- [ ] Comparison table có data
- [ ] Plots được generate

---

## 🚀 Ready to Start?

```bash
# Bắt đầu ngay!
cd D:\WORKSPACE\VJ\speaker-diarization
modal run realtime/upload_nemo_datasets.py
```

Sau khi upload xong, chọn 1 trong 3 training modes và chạy!

Good luck! 🎉
