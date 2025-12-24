# Quick Start Guide - Đánh giá NeMo Diarization trên Modal

## 📋 Tổng quan

Hệ thống gồm 3 bước chính:
1. **Upload datasets** lên Modal cloud
2. **Run evaluation** trên Modal GPU
3. **Download & analyze** kết quả

## 🚀 Các File Đã Tạo

```
realtime/
├── upload_callhome.py           # Upload Callhome dataset
├── upload_voxconverse.py        # Upload Voxconverse dataset
├── eval_diarization_modal.py    # Chạy evaluation trên Modal
├── download_eval_results.py     # Download và visualize kết quả
├── EVAL_DIARIZATION_README.md   # Hướng dẫn chi tiết
└── QUICK_START_EVAL.md          # File này
```

## ⚡ Quick Commands

### 1. Upload Datasets (Chạy 1 lần)

```bash
# Di chuyển vào thư mục realtime
cd D:\WORKSPACE\VJ\speaker-diarization\realtime

# Upload Callhome
modal run upload_callhome.py

# Upload Voxconverse
modal run upload_voxconverse.py

# Kiểm tra đã upload thành công
modal volume ls nemo-dataset
```

### 2. Run Evaluation

```bash
# Đánh giá trên Callhome
modal run eval_diarization_modal.py --dataset callhome --checkpoint best_model.nemo

# Đánh giá trên Voxconverse
modal run eval_diarization_modal.py --dataset voxconverse --checkpoint best_model.nemo

# Đánh giá cả 2 (khuyến nghị)
modal run eval_diarization_modal.py --dataset both --checkpoint best_model.nemo
```

### 3. Download & Analyze Results

```bash
# Download tất cả kết quả và tạo plots
python download_eval_results.py --compare

# Chỉ download Callhome
python download_eval_results.py --dataset callhome

# Chỉ download Voxconverse
python download_eval_results.py --dataset voxconverse
```

## 📊 Kết quả Mong đợi

### Callhome (Low Overlap)
```
DER:       8-12%   (càng thấp càng tốt)
JER:       12-18%  (càng thấp càng tốt)
EER:       2-5%    (càng thấp càng tốt)
F1:        88-94%  (càng cao càng tốt)
Purity:    90-95%  (càng cao càng tốt)
Coverage:  88-93%  (càng cao càng tốt)
```

### Voxconverse (High Overlap - Khó hơn)
```
DER:       12-20%  (càng thấp càng tốt)
JER:       18-28%  (càng thấp càng tốt)
EER:       4-8%    (càng thấp càng tốt)
F1:        80-88%  (càng cao càng tốt)
Purity:    85-92%  (càng cao càng tốt)
Coverage:  82-90%  (càng cao càng tốt)
```

> **Note**: Voxconverse có speaker overlap nên metrics sẽ kém hơn Callhome. Đây là điều bình thường.

## 📁 Output Files

Sau khi chạy xong, bạn sẽ có:

```
eval_results_downloaded/
├── eval_callhome_20251217_103045.json        # Kết quả Callhome
├── eval_voxconverse_20251217_110823.json     # Kết quả Voxconverse
├── metrics_comparison_20251217_111234.png    # So sánh metrics
├── per_file_distribution_callhome_*.png      # Phân bố DER/JER per file
└── per_file_distribution_voxconverse_*.png
```

## 🎯 Các Metrics Quan trọng

### DER (Diarization Error Rate) - Quan trọng nhất
- **Công thức**: `(False Alarm + Miss + Confusion) / Total`
- **Ý nghĩa**: Tổng lỗi của hệ thống diarization
- **Target**: < 10% là tốt, < 5% là rất tốt

### JER (Jaccard Error Rate)
- **Công thức**: `1 - (Intersection / Union)` của segments
- **Ý nghĩa**: Độ chính xác phân đoạn thời gian
- **Target**: < 15% là tốt

### EER (Equal Error Rate)
- **Ý nghĩa**: Điểm FAR = FRR (speaker verification)
- **Target**: < 5% là tốt, < 3% là rất tốt

### F1 Score
- **Ý nghĩa**: Harmonic mean của Precision và Recall
- **Target**: > 90% là tốt

## 🔍 Troubleshooting

### Dataset không tìm thấy
```bash
# Kiểm tra volume
modal volume ls nemo-dataset

# Nếu không có, upload lại
modal run upload_callhome.py
modal run upload_voxconverse.py
```

### Checkpoint không tồn tại
```bash
# Xem các checkpoint có sẵn
modal volume ls nemo-results

# Script sẽ tự động dùng pre-trained model nếu không tìm thấy checkpoint
```

### Lỗi GPU memory
- Giảm batch size trong code
- Hoặc dùng GPU lớn hơn: Sửa `gpu="A10G"` thành `gpu="A100"`

### Python dependencies thiếu
```bash
# Cài đặt dependencies cho download script
pip install matplotlib seaborn numpy
```

## 📝 Workflow Đầy đủ

```bash
# 1. Upload datasets (chỉ cần làm 1 lần)
modal run upload_callhome.py
modal run upload_voxconverse.py

# 2. Verify upload
modal volume ls nemo-dataset/callhome
modal volume ls nemo-dataset/voxconverse_dev

# 3. Run evaluation
modal run eval_diarization_modal.py --dataset both --checkpoint best_model.nemo

# 4. Download results và tạo visualizations
python download_eval_results.py --compare

# 5. Xem kết quả
cd eval_results_downloaded
# Mở file .json để xem chi tiết
# Mở file .png để xem plots
```

## 💡 Tips

1. **Chạy evaluation cả 2 dataset cùng lúc** để tiết kiệm thời gian:
   ```bash
   modal run eval_diarization_modal.py --dataset both
   ```

2. **So sánh kết quả** để hiểu model perform như thế nào trên low vs high overlap:
   ```bash
   python download_eval_results.py --compare
   ```

3. **Xem per-file results** để identify các file khó:
   ```python
   import json
   with open('eval_results_downloaded/eval_callhome_*.json') as f:
       data = json.load(f)
   
   # Sort by DER to find worst files
   sorted_files = sorted(data['per_file_results'], 
                        key=lambda x: x['DER'], 
                        reverse=True)
   print("Top 5 worst files:")
   for f in sorted_files[:5]:
       print(f"  {f['file']}: DER={f['DER']:.2%}")
   ```

4. **Cache embeddings** để tránh re-compute khi test lại:
   - Evaluation script đã tự động cache
   - Chỉ cần chạy lại với cùng checkpoint

## 📚 Đọc thêm

- [EVAL_DIARIZATION_README.md](./EVAL_DIARIZATION_README.md) - Hướng dẫn chi tiết
- [NeMo Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/)
- [Pyannote Metrics Guide](https://pyannote.github.io/pyannote-metrics/)

## ✅ Checklist

- [ ] Upload Callhome dataset
- [ ] Upload Voxconverse dataset
- [ ] Verify datasets trong Modal volume
- [ ] Run evaluation trên Callhome
- [ ] Run evaluation trên Voxconverse
- [ ] Download results
- [ ] Analyze và visualize
- [ ] So sánh kết quả với baseline

---

**Happy Evaluating! 🎉**
