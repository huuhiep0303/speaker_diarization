# Đánh Giá Model NeMo Fine-tuned cho Diarization

## Tổng Quan

File `eval_finetuned_diarization.py` đánh giá model speaker đã fine-tune trên 2 datasets:

- **Callhome**: Low overlap dataset (speakers nói xen kẽ)
- **Voxconverse-dev**: High overlap dataset (speakers nói chồng chéo)

## Metrics Đánh Giá

### 1. Diarization Metrics

- **DER (Diarization Error Rate)**: Tổng lỗi = Miss + False Alarm + Speaker Confusion
- **JER (Jaccard Error Rate)**: 1 - IoU của speaker segments
- **Purity**: Độ "thuần khiết" của clusters (segments có đúng speaker không)
- **Coverage**: Độ "phủ" của clusters (có đủ segments cho mỗi speaker không)

### 2. Speaker Verification Metrics

- **EER (Equal Error Rate)**: Điểm FAR = FRR (càng thấp càng tốt)
- **FAR (False Acceptance Rate)**: Tỉ lệ nhận diện sai speaker khác thành cùng speaker
- **FRR (False Rejection Rate)**: Tỉ lệ từ chối sai cùng speaker thành speaker khác
- **AUC (Area Under Curve)**: Diện tích dưới ROC curve (càng cao càng tốt, tối đa 1.0)
- **Precision**: Trong số speaker được nhận diện, có bao nhiêu % đúng
- **Recall**: Trong số speaker thực tế, nhận diện được bao nhiêu %
- **F1-Score**: Trung bình điều hòa của Precision và Recall

### 3. Per-file Metrics

- Duration, số speakers thực tế vs dự đoán
- DER, JER, Purity, Coverage cho từng file

## Cách Sử Dụng

### Bước 1: Upload Datasets (nếu chưa có)

```bash
# Upload Callhome dataset
modal run upload_callhome.py

# Upload Voxconverse dataset
modal run upload_voxconverse.py

# Kiểm tra datasets đã upload
modal volume ls nemo-dataset
```

### Bước 2: Kiểm Tra Checkpoints

```bash
# Xem danh sách checkpoints có sẵn
modal volume ls nemo-results/checkpoints/

# Kết quả ví dụ:
# best_model.pt           <-- Model tốt nhất (DEFAULT)
# epoch_10_model.pt
# epoch_20_model.pt
# epoch_30_model.pt
```

⚠️ **LUU Ý**: Model được lưu với extension `.pt` (không phải `.nemo`)

### Bước 3: Chạy Evaluation

#### Đánh giá trên Callhome (với detach mode)

```bash
modal run --detach eval_finetuned_diarization.py --dataset callhome
```

#### Đánh giá trên Voxconverse

```bash
modal run --detach eval_finetuned_diarization.py --dataset voxconverse
```

#### Đánh giá trên cả 2 datasets (KHUYẾN NGHỊ)

```bash
modal run --detach eval_finetuned_diarization.py --dataset both
```

#### Chỉ định checkpoint cụ thể

```bash
modal run --detach eval_finetuned_diarization.py --dataset both --checkpoint epoch_30_model.pt
```

**🔑 Detach Mode Benefits:**

- ✅ Không bị ngắt kết nối khi chạy lâu
- ✅ Có thể tắt terminal/máy tính
- ✅ Xem progress trên Modal dashboard
- ✅ Results vẫn được lưu vào Modal volume

### Bước 4: Download Results

```bash
# Download tất cả kết quả evaluation
modal volume get nemo-results eval_*.json ./evaluation_results/

# Hoặc download file cụ thể
modal volume get nemo-results eval_callhome_20251217_120000.json ./evaluation_results/
```

## Kết Quả Output

### 1. Console Output

```
================================================================================
📊 EVALUATION SUMMARY
================================================================================

Dataset         DER        JER        EER        F1         Files
--------------------------------------------------------------------------------
callhome        15.30%     18.45%     3.25%      92.50%     50
voxconverse     22.15%     25.80%     5.10%      88.20%     100

================================================================================
✅ Evaluation Completed Successfully!
================================================================================
```

### 2. JSON Results File

```json
{
  "dataset": "callhome",
  "dataset_description": "Callhome - Low overlap (speakers xen kẽ)",
  "checkpoint": "best_model.nemo",
  "timestamp": "2025-12-17T12:00:00",
  "files_processed": 50,
  "total_duration": 1500.5,
  "metrics": {
    "DER_mean": 0.153,
    "DER_std": 0.045,
    "JER_mean": 0.1845,
    "JER_std": 0.052,
    "purity": 0.895,
    "coverage": 0.912,
    "EER": 0.0325,
    "FAR_at_EER": 0.033,
    "FRR_at_EER": 0.032,
    "threshold_EER": 0.7845,
    "AUC": 0.982,
    "precision": 0.935,
    "recall": 0.915,
    "F1": 0.925,
    "TP": 1850,
    "TN": 1920,
    "FP": 132,
    "FN": 168,
    "total_pairs": 4070,
    "same_speaker_pairs": 2018,
    "diff_speaker_pairs": 2052
  },
  "per_file_results": [
    {
      "file": "audio_0.wav",
      "duration": 30.5,
      "num_speakers_ref": 2,
      "num_speakers_hyp": 2,
      "DER": 0.125,
      "JER": 0.145,
      "purity": 0.92,
      "coverage": 0.88
    }
  ]
}
```

## Hiểu Kết Quả

### DER (Diarization Error Rate)

- **< 10%**: Rất tốt (production ready)
- **10-20%**: Tốt (có thể dùng trong nhiều trường hợp)
- **20-30%**: Trung bình (cần cải thiện)
- **> 30%**: Kém (cần train lại hoặc điều chỉnh)

### EER (Equal Error Rate)

- **< 5%**: Rất tốt
- **5-10%**: Tốt
- **10-15%**: Trung bình
- **> 15%**: Kém

### F1-Score

- **> 90%**: Rất tốt
- **80-90%**: Tốt
- **70-80%**: Trung bình
- **< 70%**: Kém

## So Sánh Callhome vs Voxconverse

### Callhome (Low Overlap)

- Speakers nói **xen kẽ nhau**, ít overlap
- Dễ hơn cho diarization
- DER thường **thấp hơn** (10-20%)
- Phù hợp kiểm tra: model có phân biệt được speakers riêng biệt không?

### Voxconverse (High Overlap)

- Speakers nói **chồng lên nhau**, nhiều overlap
- Khó hơn cho diarization
- DER thường **cao hơn** (20-30%)
- Phù hợp kiểm tra: model có xử lý được overlap không?

## Troubleshooting

### ❌ Dataset not found

```
💡 Giải pháp:
modal run upload_callhome.py
modal run upload_voxconverse.py
```

### ❌ Checkpoint not found

```
💡 Kiểm tra:
modal MemoryError during local upload
```

💡 Nguyên nhân: Dataset local quá lớn (>2GB)

Giải pháp:

1. Upload dataset lên Modal volume (KHUYẾN NGHỊ):
   modal run upload_callhome.py
   modal run upload_voxconverse.py
   modal run --detach eval_finetuned_diarization.py --dataset both

2. Giảm số files local:
   modal run --detach eval_finetuned_diarization.py --dataset callhome --use-local-dataset --max-files 20

3. Tăng RAM máy local (cần >4GB free RAM cho 50 files)

```

### ❌ Out of memory on Modalo-results/checkpoints/

Nếu không có checkpoints:
modal run modal_setup.py --dataset ../dataset/jvs_ver1/jvs_ver1 --epochs 30
```

### ❌ Out of memory

```
💡 Giải pháp:
- Giảm batch size trong code
- Tăng memory trong @app.function (hiện tại: 32GB)
- Sử dụng GPU lớn hơn (A100 thay vì A10G)
```

### ❌ No embeddings extracted

```
💡 Nguyên nhân: Audio quá ngắn (< 1.5s)
Giải pháp: Bỏ qua file này hoặc giảm window_size
```

## Tùy Chỉnh

### Thay đổi Window Size cho Embedding Extraction

Trong file `eval_finetuned_diarization.py`, dòng 248-249:

````python
window_size = 1.5  # seconds - tăng lên nếu cần context dài hơn
hop_size = 0.75     # seconds - giảm xuống nếu cần phân giải cao hơn
### Option 1: Full Dataset (KHUYẾN NGHỊ)
```bash
# 1. Fine-tune model (nếu chưa làm)
modal run modal_setup.py --dataset ../dataset/jvs_ver1/jvs_ver1 --epochs 30

# 2. Upload evaluation datasets (1 lần duy nhất)
modal run upload_callhome.py
modal run upload_voxconverse.py

# 3. Evaluate model với DETACH mode
modal run --detach eval_finetuned_diarization.py --dataset both --checkpoint best_model.pt

# 4. Kiểm tra progress (optional)
# Truy cập: https://modal.com/apps

# 5. Download results khi hoàn tất
modal volume get nemo-results eval_*.json ./evaluation_results/

# 6. Analyze results
# Mở file JSON và so sánh metrics
````

### Option 2: Quick Test (Local Upload - Limited)

```bash
# 1. Ensure model đã được fine-tune

# 2. Quick test với 50 files đầu
modal run --detach eval_finetuned_diarization.py \
  --dataset callhome \
  --use-local-dataset \
  --max-files 50 \
  --checkpoint best_model.pt

# 3. Download results
modal volume get nemo-results eval_*.json ./evaluation_results/
```

**⚠️ KHUYẾN CÁO**:

- Dùng Option 1 cho kết quả chính xác (full dataset)
- Dùng Option 2 chỉ để test nhanh
- Luôn dùng `--detach` để tránh ngắt kết nối_metric = JaccardErrorRate(collar=0.25)

# collar: 0.0 = strict, 0.25 = standard, 0.5 = lenient

````

## Workflow Hoàn Chỉnh

```bash
# 1. Fine-tune model (nếu chưa làm)
modal run modal_setup.py --dataset ../dataset/jvs_ver1/jvs_ver1 --epochs 30

# 2. Upload evaluation datasets
modal run upload_callhome.py
modal run upload_voxconverse.py

# 3. Evaluate model
modal run eval_finetuned_diarization.py --dataset both

# 4. Download results
modal volume get nemo-results eval_*.json ./evaluation_results/

# 5. Analyze results
# Mở file JSON và so sánh metrics
````

## Kết Luận

File này giúp bạn:

1. ✅ Đánh giá model fine-tuned trên datasets chuẩn
2. ✅ Đo lường đầy đủ metrics: DER, JER, EER, FAR, FRR, AUC, F1,...
3. ✅ So sánh performance trên low-overlap vs high-overlap
4. ✅ Export kết quả chi tiết dạng JSON
5. ✅ Chạy hoàn toàn trên Modal cloud (không cần GPU local)

**Chúc bạn đánh giá thành công! 🎉**
