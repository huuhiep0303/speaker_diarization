# Kết Quả Đánh Giá Speaker Diarization

**Ngày đánh giá**: 5 tháng 12, 2025  
**Dataset**: JVS Corpus (Japanese speakers: jvs001-jvs050)  
**Số trials**: 3000 (genuine + impostor pairs)  
**Loại đánh giá**: Speaker Verification (khả năng phân biệt speaker embeddings)

---

## 📊 Tổng Quan Kết Quả

| Model                      | EER        | AUC        | Best F1    | Precision@F1 | Recall@F1  |
| -------------------------- | ---------- | ---------- | ---------- | ------------ | ---------- |
| **SpeechBrain ECAPA-TDNN** | 13.80%     | 0.9350     | 87.37%     | 97.42%       | 79.20%     |
| **NeMo TitaNet Large**     | **13.40%** | **0.9508** | **88.68%** | **97.83%**   | **81.10%** |

### 🏆 Kết Luận Nhanh

**NeMo TitaNet Large thắng nhẹ so với SpeechBrain ECAPA-TDNN:**

- ✅ EER thấp hơn: 13.40% vs 13.80% (giảm 0.4%)
- ✅ AUC cao hơn: 0.9508 vs 0.9350 (tăng 1.58%)
- ✅ F1 tốt hơn: 88.68% vs 87.37% (tăng 1.31%)

**Tuy nhiên, sự khác biệt không lớn** → Cả 2 model đều hoạt động tốt tương đương.

---

## 🔍 Phân Tích Chi Tiết

### 1. SpeechBrain ECAPA-TDNN

**Thông tin model:**

- Kiến trúc: ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation in TDNN)
- Embedding dimension: 192
- Pretrained: VoxCeleb dataset
- Sử dụng bởi: Whisper, SenseVoice, SenseVoice+SpeechBrain

**Kết quả:**

| Metric                  | Giá trị | Ý nghĩa                                                   |
| ----------------------- | ------- | --------------------------------------------------------- |
| **EER**                 | 13.80%  | Tại threshold 0.3950, FAR = FRR = 13.80%                  |
| **FAR @ EER**           | 13.80%  | 13.8% cặp impostor bị nhận nhầm là cùng speaker           |
| **FRR @ EER**           | 13.80%  | 13.8% cặp genuine bị từ chối nhầm                         |
| **Threshold @ EER**     | 0.3950  | Ngưỡng cosine similarity tối ưu                           |
| **Precision @ EER**     | 75.75%  | Trong số cặp dự đoán "cùng speaker", 75.75% đúng          |
| **Recall @ EER**        | 86.20%  | Trong số cặp thực sự "cùng speaker", 86.20% được tìm ra   |
| **F1 @ EER**            | 80.64%  | Điểm F1 tại threshold EER                                 |
| **Best F1**             | 87.37%  | F1 score cao nhất (tại threshold 0.5708)                  |
| **Precision @ Best F1** | 97.42%  | Precision cực cao khi tối ưu F1                           |
| **Recall @ Best F1**    | 79.20%  | Trade-off: Precision cao → Recall thấp hơn                |
| **AUC**                 | 0.9350  | Khả năng phân biệt tổng thể (93.5% so với ngẫu nhiên 50%) |

**Đánh giá:**

- ✅ **Điểm mạnh**:

  - AUC cao (0.9350) → Phân biệt tốt genuine/impostor
  - Precision@F1 rất cao (97.42%) → Ít false positive khi adjust threshold
  - Model nhẹ, inference nhanh
  - Đã được verify trên nhiều hệ thống (Whisper, SenseVoice)

- ⚠️ **Điểm yếu**:
  - EER 13.80% hơi cao cho speaker verification
  - Recall@F1 không cao (79.20%) → Miss 20.8% genuine pairs khi tối ưu precision

---

### 2. NeMo TitaNet Large

**Thông tin model:**

- Kiến trúc: TitaNet Large (Titan Network for Speaker Recognition)
- Embedding dimension: 192/512 (tùy config)
- Pretrained: VoxCeleb + proprietary data (NVIDIA)
- Sử dụng bởi: main_nemo.py

**Kết quả:**

| Metric                  | Giá trị | Ý nghĩa                                        |
| ----------------------- | ------- | ---------------------------------------------- |
| **EER**                 | 13.40%  | Tại threshold 0.4039, FAR = FRR = 13.40%       |
| **FAR @ EER**           | 13.40%  | 13.4% cặp impostor bị nhận nhầm                |
| **FRR @ EER**           | 13.40%  | 13.4% cặp genuine bị từ chối nhầm              |
| **Threshold @ EER**     | 0.4039  | Ngưỡng cao hơn SpeechBrain (0.3950)            |
| **Precision @ EER**     | 76.37%  | Cao hơn SpeechBrain (+0.62%)                   |
| **Recall @ EER**        | 86.60%  | Cao hơn SpeechBrain (+0.40%)                   |
| **F1 @ EER**            | 81.16%  | Cao hơn SpeechBrain (+0.52%)                   |
| **Best F1**             | 88.68%  | Cao nhất trong 2 models                        |
| **Precision @ Best F1** | 97.83%  | Precision tốt nhất                             |
| **Recall @ Best F1**    | 81.10%  | Recall cao hơn SpeechBrain (+1.90%)            |
| **AUC**                 | 0.9508  | Tốt nhất, cao hơn SpeechBrain đáng kể (+1.58%) |

**Đánh giá:**

- ✅ **Điểm mạnh**:

  - **AUC cao nhất (0.9508)** → Phân biệt tốt hơn SpeechBrain
  - **EER thấp hơn (13.40%)** → Cân bằng FAR/FRR tốt hơn
  - **Best F1 cao nhất (88.68%)** → Performance tổng thể tốt hơn
  - **Recall@F1 cao hơn (81.10%)** → Ít miss genuine pairs hơn
  - Threshold cao hơn (0.4039) → Có thể discriminative hơn

- ⚠️ **Điểm yếu**:
  - Model lớn hơn → Inference có thể chậm hơn
  - EER 13.40% vẫn chưa xuất sắc (baseline Pyannote 3.0: ~3-5%)

---

## 📈 So Sánh Trực Tiếp

### Performance Gap

| Metric       | SpeechBrain | NeMo TitaNet | Chênh lệch | Winner  |
| ------------ | ----------- | ------------ | ---------- | ------- |
| EER          | 13.80%      | **13.40%**   | -0.40%     | 🥇 NeMo |
| AUC          | 0.9350      | **0.9508**   | +1.58%     | 🥇 NeMo |
| Best F1      | 87.37%      | **88.68%**   | +1.31%     | 🥇 NeMo |
| Precision@F1 | 97.42%      | **97.83%**   | +0.41%     | 🥇 NeMo |
| Recall@F1    | 79.20%      | **81.10%**   | +1.90%     | 🥇 NeMo |
| F1@EER       | 80.64%      | **81.16%**   | +0.52%     | 🥇 NeMo |

**Kết luận**: NeMo TitaNet Large thắng **toàn bộ metrics**.

---

## ⚠️ Lưu Ý Quan Trọng

### 1. Đây KHÔNG phải đánh giá Diarization End-to-End

Metrics trên đánh giá **Speaker Verification** (khả năng phân biệt embeddings trên trials đơn giản), **KHÔNG phải Diarization Error Rate (DER)**.

**Tại sao EER ~13-14% (không quá ấn tượng)?**

1. **Dataset nhỏ**: Chỉ test trên 50 speakers (jvs001-jvs050)

   - Baseline VoxCeleb: 7000+ speakers
   - Dataset nhỏ → ít variability → kết quả có thể không đại diện

2. **Trials đơn giản**:

   - Positive pairs: Có thể từ cùng file/session → cosine similarity rất cao
   - Negative pairs: Speakers khác nhau rõ ràng → dễ phân biệt
   - Không có điều kiện thực tế: noise, overlap, cross-talk

3. **Thiếu hard trials**:

   - Không có same-gender hard pairs
   - Không có similar voice pairs
   - Không có short utterances

4. **JVS Corpus đặc thù**:
   - Audio chất lượng cao (24kHz 16bit)
   - Studio recording (ít nhiễu)
   - Read speech (không phải spontaneous)
   - Japanese corpus (có thể không optimal cho models trained on English)

### 2. Để đánh giá thực tế cần:

- ✅ **DER** (Diarization Error Rate) trên audio có nhiều speakers
- ✅ **JER** (Jaccard Error Rate)
- ✅ Test trên data có overlap, noise, reverb
- ✅ Test trên conversational speech (không phải read speech)
- ✅ Test với short utterances (<2s)
- ✅ Collar 0.25s (tolerance cho boundary errors)

---

## 🎯 Khuyến Nghị

### 1. Lựa Chọn Model

**Nếu ưu tiên Performance**:
→ Chọn **NeMo TitaNet Large**

- AUC cao nhất (0.9508)
- EER thấp nhất (13.40%)
- Recall tốt hơn (ít miss genuine pairs)

**Nếu ưu tiên Speed/Efficiency**:
→ Chọn **SpeechBrain ECAPA-TDNN**

- Performance chỉ kém hơn 1-2%
- Model nhẹ hơn, inference nhanh hơn
- Đã được verify trong nhiều hệ thống production

**Kết luận**: Chênh lệch không lớn (1-2%), có thể dùng cả 2 models tùy use case.

### 2. Cải Thiện Kết Quả

**Cải thiện EER từ 13-14% xuống <5%**:

1. **Augment training data**:

   - Add noise, reverb, codec artifacts
   - Mix speakers (simulate overlap)
   - Variable speech rate, pitch

2. **Fine-tune trên JVS**:

   - Fine-tune models trên JVS corpus
   - Japanese-specific optimization
   - Domain adaptation

3. **Ensemble methods**:

   - Combine SpeechBrain + NeMo predictions
   - Score fusion (weighted average)

4. **Better clustering**:

   - Thay AgglomerativeClustering bằng Spectral Clustering
   - Thử UMAP/t-SNE dimension reduction trước clustering
   - Optimize hyperparameters (linkage, metric, threshold)

5. **Post-processing**:
   - Smoothing (median filter trên speaker labels)
   - Minimum duration constraints
   - Re-segmentation with VAD

### 3. Đánh Giá Đầy Đủ Hơn

**Next steps**:

1. ✅ **Tăng số speakers**: Test trên 100 speakers (jvs001-jvs100)
2. ✅ **Tạo hard trials**: Same-gender pairs, short utterances
3. ✅ **Test end-to-end**: Đánh giá DER trên audio dài với multi-speaker conversations
4. ✅ **Cross-dataset**: Test trên dataset khác (AMI, CALLHOME, DIHARD)
5. ✅ **Real-world conditions**: Add noise, overlap, variable SNR

---

## 📚 Tham Khảo

### Baseline Performance (từ literature)

| System                  | EER        | AUC        | Dataset          | Note               |
| ----------------------- | ---------- | ---------- | ---------------- | ------------------ |
| Kaldi x-vectors         | 5-8%       | 0.97-0.98  | VoxCeleb         | Baseline cũ        |
| Pyannote 3.0            | 3-5%       | 0.98-0.99  | VoxCeleb         | SOTA 2023          |
| SpeechBrain ECAPA       | 2-4%       | 0.98-0.99  | VoxCeleb         | Official benchmark |
| **Ours (SpeechBrain)**  | **13.80%** | **0.9350** | **JVS (50 spk)** | Small dataset      |
| **Ours (NeMo TitaNet)** | **13.40%** | **0.9508** | **JVS (50 spk)** | Small dataset      |

**Gap analysis**: EER của chúng ta cao gấp ~3x benchmark

- Nguyên nhân: Dataset nhỏ, trials đơn giản, không fine-tune
- Giải pháp: Fine-tune + hard trials + larger dataset

### Papers

1. **ECAPA-TDNN**: Desplanques et al. (2020) - "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification"
2. **TitaNet**: Koluguri et al. (2022) - "TitaNet: Neural Model for speaker representation with 1D Depth-wise separable convolutions and global context"
3. **Pyannote 3.0**: Bredin et al. (2023) - "Pyannote.audio 2.1 speaker diarization pipeline"

---

## 📁 Generated Files

Kết quả đầy đủ được lưu tại:

- `eval_results/result.log` - Summary log
- `eval_results/roc_curves.png` - ROC curves comparison
- `eval_results/det_curves.png` - DET curves comparison
- `eval_results/precision_recall_curves.png` - PR curves comparison
- `eval_results/eval_diarization_speechbrain_results.json` - SpeechBrain detailed metrics
- `eval_results/eval_diarization_nemo_results.json` - NeMo detailed metrics
- `eval_cache/embeddings_cache_*.pkl` - Cached embeddings (reusable)

---

**Tóm lại**: NeMo TitaNet Large có performance tốt hơn nhẹ so với SpeechBrain ECAPA-TDNN trên JVS dataset (50 speakers), nhưng cả 2 đều chưa đạt SOTA. Cần fine-tune và test trên dataset lớn hơn để đánh giá chính xác.
