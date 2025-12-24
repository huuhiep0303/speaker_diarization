# 🚀 HƯỚNG DẪN CẢI THIỆN DIARIZATION - TỪ DER 150% → < 20%

## 📋 TÓM TẮT VẤN ĐỀ

**Kết quả hiện tại (RẤT TỆ):**

- Callhome: DER = 153.78% ❌
- Voxconverse: DER = 132.69% ❌

**Nguyên nhân chính:**

1. ❌ Training objective sai (Classification thay vì Metric Learning)
2. ❌ Domain mismatch (Japanese → English/Multi-lingual)
3. ❌ Catastrophic forgetting (unfreeze toàn bộ encoder)
4. ❌ Clustering parameters không tối ưu
5. ❌ Thiếu VAD và post-processing

**Chi tiết phân tích:** Xem [diarization_analysis.md](diarization_analysis.md)

---

## 🎯 GIẢI PHÁP 3 BƯỚC (QUICK → ADVANCED)

### 🔥 BƯỚC 1: QUICK FIX (30 phút - Giảm DER xuống ~35-40%)

**Sử dụng pretrained model thay vì fine-tuned model + cải thiện clustering**

```bash
cd d:\WORKSPACE\VJ\speaker-diarization\realtime

# Chạy evaluation với pretrained model
modal run eval_finetuned_diarization_improved.py \
    --dataset both \
    --use-pretrained \
    --enable-vad \
    --auto-detect-speakers
```

**Cải tiến:**

- ✅ Pretrained TitaNet (tốt hơn fine-tuned model hiện tại)
- ✅ Window size: 1.5s → 0.5s (độ phân giải cao hơn)
- ✅ Hop size: 0.75s → 0.25s (ít miss speaker boundaries)
- ✅ VAD preprocessing (loại bỏ silence/noise)
- ✅ Spectral clustering với auto-detection số speakers
- ✅ Post-processing: median filter smoothing, merge short segments

**Expected results:**

```
Callhome:    DER ≈ 35-40% (giảm 75%)
Voxconverse: DER ≈ 30-35% (giảm 73%)
```

---

### 🚀 BƯỚC 2: TRAIN MỚI VỚI TRIPLET LOSS (1 tuần - DER < 25%)

**Train model mới với Metric Learning thay vì Classification**

#### 2.1. Chuẩn bị dataset

```bash
# Đảm bảo có JVS dataset
# Nếu chưa có, download từ: https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus

# Cấu trúc:
# dataset/jvs_ver1/jvs_ver1/
#   jvs001/
#   jvs002/
#   ...
#   jvs100/
```

#### 2.2. Train với Triplet Loss

```bash
# Local training
python finetune_nemo_speaker_triplet.py \
    --dataset ../dataset/jvs_ver1/jvs_ver1 \
    --epochs 30 \
    --batch_size 32 \
    --lr 1e-3 \
    --margin 0.2 \
    --mining hard \
    --freeze_encoder

# Output: finetuned_models_triplet/checkpoints/best_model_triplet.pt
```

**Cải tiến so với training cũ:**
| Feature | Old (Classification) | New (Triplet Loss) |
|---------|---------------------|-------------------|
| **Loss function** | Cross-Entropy | Triplet Loss |
| **Training goal** | Classify 10 speakers | Learn speaker similarity |
| **Encoder** | Unfreeze all (catastrophic forgetting) | Freeze (preserve pretrained) |
| **Generalization** | ❌ Chỉ 10 speakers | ✅ Bất kỳ speaker nào |
| **Augmentation** | ❌ None | ✅ Noise, pitch, reverb |
| **Mining** | ❌ None | ✅ Hard negative mining |

#### 2.3. Upload checkpoint lên Modal

```bash
# Upload trained model
modal volume put nemo-results \
    finetuned_models_triplet/checkpoints/best_model_triplet.pt \
    /results/checkpoints/best_model_triplet.pt
```

#### 2.4. Evaluate model mới

```bash
# Evaluate with new triplet-trained model
modal run eval_finetuned_diarization_improved.py \
    --dataset both \
    --checkpoint best_model_triplet.pt \
    --enable-vad \
    --auto-detect-speakers
```

**Expected results:**

```
Callhome:    DER ≈ 20-25% (SOTA baseline)
Voxconverse: DER ≈ 18-22%
```

---

### 🏆 BƯỚC 3: DOMAIN ADAPTATION (2-3 tuần - DER < 15%)

**Fine-tune trên target domain (Callhome/Voxconverse)**

#### 3.1. Prepare domain adaptation dataset

```python
# Create pseudo-labels from pretrained model
python create_pseudo_labels.py \
    --dataset callhome \
    --model pretrained \
    --confidence_threshold 0.8

# Output: pseudo_labels/callhome_pseudo.json
```

#### 3.2. Domain adaptation training

```python
python finetune_domain_adaptation.py \
    --source_dataset jvs \
    --target_dataset callhome \
    --pseudo_labels pseudo_labels/callhome_pseudo.json \
    --epochs 20 \
    --adaptation_method mmd  # Maximum Mean Discrepancy
```

#### 3.3. Ensemble multiple models

```python
# Combine predictions from multiple models
python eval_ensemble.py \
    --models pretrained,triplet,domain_adapted \
    --dataset both \
    --fusion_method weighted_average
```

**Expected results:**

```
Callhome:    DER ≈ 12-15% (SOTA competitive)
Voxconverse: DER ≈ 10-12%
```

---

## 📊 SO SÁNH KẾT QUẢ

| Method                    | Callhome DER   | Voxconverse DER | Time   | Effort     |
| ------------------------- | -------------- | --------------- | ------ | ---------- |
| **Hiện tại (Fine-tuned)** | **153.78%** ❌ | **132.69%** ❌  | -      | -          |
| **Bước 1: Quick Fix**     | **~35%** ⚠️    | **~30%** ⚠️     | 30 min | ⭐         |
| **Bước 2: Triplet Loss**  | **~20%** ✅    | **~18%** ✅     | 1 tuần | ⭐⭐⭐     |
| **Bước 3: Domain Adapt**  | **~13%** 🏆    | **~11%** 🏆     | 3 tuần | ⭐⭐⭐⭐⭐ |
| Pyannote 3.1 (SOTA)       | 21.7%          | 11.2%           | -      | -          |

---

## 🔧 CHI TIẾT KỸ THUẬT

### Files đã tạo

1. **diarization_analysis.md** - Phân tích chi tiết vấn đề
2. **eval_finetuned_diarization_improved.py** - Evaluation script cải tiến
3. **finetune_nemo_speaker_triplet.py** - Training script với Triplet Loss
4. **IMPROVEMENT_GUIDE.md** - Hướng dẫn này

### Thay đổi chính trong evaluation

```python
# OLD (eval_finetuned_diarization.py)
window_size = 1.5  # seconds
hop_size = 0.75    # seconds
clustering = AgglomerativeClustering(
    n_clusters=num_ref_speakers,  # Ground truth
    metric='cosine',
    linkage='average'
)

# NEW (eval_finetuned_diarization_improved.py)
window_size = 0.5   # seconds - Better resolution
hop_size = 0.25     # seconds - Less boundary miss
vad_preprocessing = True  # Remove silence
clustering = SpectralClustering(
    n_clusters=estimated_speakers,  # Auto-detect
    affinity='precomputed'
)
smoothing = median_filter(labels, kernel_size=5)
```

### Thay đổi chính trong training

```python
# OLD (finetune_nemo_speaker.py)
# Classification objective
loss = CrossEntropyLoss(logits, labels)

# Unfreeze all
for param in model.parameters():
    param.requires_grad = True  # ❌ Catastrophic forgetting

# NEW (finetune_nemo_speaker_triplet.py)
# Metric learning objective
loss = TripletLoss(anchor, positive, negative, margin=0.2)

# Freeze encoder, train adapter only
for param in encoder.parameters():
    param.requires_grad = False  # ✅ Preserve pretrained knowledge

# Only train adapter
adapter = nn.Sequential(
    nn.Linear(192, 256),
    nn.ReLU(),
    nn.Linear(256, 192)
)
```

---

## 🎓 METRICS MỚI ĐÃ THÊM

### Diarization Metrics (đã có)

- ✅ **DER** (Diarization Error Rate): Miss + False Alarm + Confusion
- ✅ **JER** (Jaccard Error Rate): 1 - IoU của segments
- ✅ **Purity**: Đồng nhất của clusters
- ✅ **Coverage**: Độ che phủ của ground truth

### Speaker Verification Metrics (MỚI)

- ✅ **EER** (Equal Error Rate): FAR = FRR point
- ✅ **FAR** (False Acceptance Rate): Nhận diện sai
- ✅ **FRR** (False Rejection Rate): Từ chối đúng
- ✅ **AUC** (Area Under ROC Curve): Tổng thể performance
- ✅ **Precision, Recall, F1**: Frame-level metrics
- ✅ **TP, TN, FP, FN**: Confusion matrix elements

### Visualizations (MỚI)

- ✅ **ROC Curve**: FPR vs TPR
- ✅ **DET Curve**: FAR vs FRR (log scale)
- ✅ **Training curves**: Loss, EER, AUC

---

## 🐛 TROUBLESHOOTING

### Issue 1: Modal volume không tìm thấy dataset

```bash
# Check volumes
modal volume list

# Upload dataset nếu chưa có
modal volume put nemo-dataset \
    local_dataset/callhome \
    /dataset/callhome

# Hoặc chạy script upload
modal run upload_callhome.py
modal run upload_voxconverse.py
```

### Issue 2: OOM (Out of Memory)

```python
# Giảm batch size
python finetune_nemo_speaker_triplet.py --batch_size 16

# Hoặc giảm window size
# Sửa trong Config:
max_duration = 2.0  # 3.0 → 2.0
```

### Issue 3: Training không converge

```python
# Thử các chiến lược mining khác
--mining semi-hard  # Thay vì hard

# Hoặc tăng margin
--margin 0.3  # Thay vì 0.2

# Hoặc giảm learning rate
--lr 5e-4  # Thay vì 1e-3
```

### Issue 4: Pretrained model vẫn tệ (DER > 50%)

```bash
# Check VAD có hoạt động không
--enable-vad  # Phải có flag này

# Thử không dùng auto-detection
# (dùng ground truth số speakers)
# Bỏ flag --auto-detect-speakers

# Check audio files có bị corrupt không
ffmpeg -i audio.wav -f null -
```

---

## 📈 EXPECTED TIMELINE

### Week 1: Quick Wins

- Day 1: Chạy Bước 1 (Quick Fix) → DER ~35%
- Day 2-3: Analyze results, tune parameters
- Day 4-5: Document findings

### Week 2: Triplet Loss Training

- Day 1-2: Setup training environment, prepare data
- Day 3-5: Train with triplet loss
- Day 6-7: Evaluate and compare

### Week 3-4: Domain Adaptation (Optional)

- Week 3: Pseudo-labeling and domain adaptation training
- Week 4: Ensemble, fine-tuning, final evaluation

---

## 🎯 SUCCESS METRICS

### Minimum Acceptable (Bước 1)

- ✅ Callhome DER < 40%
- ✅ Voxconverse DER < 35%

### Target (Bước 2)

- ✅ Callhome DER < 25%
- ✅ Voxconverse DER < 22%
- ✅ EER < 10%
- ✅ AUC > 0.90

### Stretch Goal (Bước 3)

- 🏆 Callhome DER < 15%
- 🏆 Voxconverse DER < 12%
- 🏆 EER < 5%
- 🏆 AUC > 0.95
- 🏆 Competitive với Pyannote 3.1

---

## 💡 NEXT STEPS

1. **Immediate (Ngay bây giờ):**

   ```bash
   cd d:\WORKSPACE\VJ\speaker-diarization\realtime
   modal run eval_finetuned_diarization_improved.py --dataset callhome --use-pretrained
   ```

2. **This Week:**
   - Train với triplet loss
   - Compare pretrained vs triplet-trained
3. **Next Week:**
   - Domain adaptation
   - Ensemble methods
4. **Future:**
   - End-to-end diarization (không cần clustering)
   - Real-time optimization
   - Multi-modal (audio + video)

---

## 📚 REFERENCES

- **Triplet Loss Paper:** [FaceNet (Schroff et al., 2015)](https://arxiv.org/abs/1503.03832)
- **TitaNet Model:** [NeMo TitaNet](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speaker_recognition/models.html)
- **Pyannote:** [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- **DER Metric:** [NIST RT Evaluation](https://catalog.ldc.upenn.edu/docs/LDC2004S04/rt03-spring-eval-plan-v4.pdf)

---

## ✅ CHECKLIST

Sau khi hoàn thành guide:

- [ ] Đã chạy Bước 1 (Quick Fix)
- [ ] DER giảm xuống dưới 40%
- [ ] Đã train model với Triplet Loss
- [ ] DER giảm xuống dưới 25%
- [ ] Đã thêm đầy đủ metrics (EER, AUC, F1, etc.)
- [ ] Đã tạo visualizations (ROC, DET curves)
- [ ] Đã document kết quả
- [ ] Đã compare với baseline/SOTA

---

**Created:** 2025-12-17  
**Author:** AI Assistant  
**Status:** ✅ Complete
