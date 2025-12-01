# Đánh giá Speaker Diarization Models

## Giải thích kết quả cao bất thường

File `eval_diarization.py` đánh giá **speaker verification** (khả năng phân biệt embeddings), **KHÔNG phải** diarization end-to-end.

### Vì sao EER ~0.3% (rất cao)?

1. **Trials đơn giản**:
   - Positive pairs: có thể từ cùng file/phiên → cosine similarity gần 1
   - Negative pairs: speakers rất khác nhau → dễ phân biệt
2. **Dataset JVS sạch**:
   - Audio chất lượng cao, ít nhiễu
   - Embeddings ECAPA-TDNN phân tách rất tốt
3. **Không có điều kiện thực tế**:
   - Không overlap speakers
   - Không nhiễu nền phức tạp
   - Không cross-talk

### Đánh giá thực tế cần:

1. **Trials khó hơn**:

   ```python
   # Positive: bắt buộc khác file, cách nhau >= X giây
   # Negative: cùng domain, cùng giới tính
   # Khử trùng lặp trial
   ```

2. **DER end-to-end**:

   ```
   DER = (Miss + FA + Confusion) / Total
   ```

   - Với collar 0.25s
   - Trên audio thực có overlap

3. **Test conditions thực tế**:
   - Nhiễu nền
   - Overlap speakers
   - Channel mismatch

## Cách chạy

### 1. Chuẩn bị

```bash
cd realtime/evaluation
pip install -r requirements.txt
```

Tải dataset JVS:

```bash
# Download từ https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus
# Giải nén vào ../dataset/jvs_ver1/
```

### 2. Chạy đánh giá

```bash
# Chạy với cache (nhanh hơn lần sau)
python eval_diarization.py --dataset ../dataset/jvs_ver1

# Force re-extract embeddings
python eval_diarization.py --dataset ../dataset/jvs_ver1 --no_cache

# Xóa cache và chạy lại
python eval_diarization.py --dataset ../dataset/jvs_ver1 --clear_cache
```

### 3. Tùy chỉnh trials

```bash
# Giảm số trials (chạy nhanh hơn)
python eval_diarization.py --max_genuine_per_spk 20 --impostor_per_spk 50

# Tăng số trials (đánh giá chính xác hơn)
python eval_diarization.py --max_genuine_per_spk 100 --impostor_per_spk 200
```

## Kết quả

File output trong `eval_results/`:

- `roc_curves.png`: ROC curves so sánh
- `det_curves.png`: DET curves so sánh
- `precision_recall_curves.png`: PR curves so sánh
- `eval_diarization_*_results.json`: Kết quả chi tiết từng model
- `result.log`: Summary log

## Hiểu metrics

| Metric    | Ý nghĩa        | Mục tiêu     |
| --------- | -------------- | ------------ |
| EER       | FAR = FRR      | Thấp hơn tốt |
| AUC       | Diện tích ROC  | Gần 1.0 tốt  |
| Precision | TP / (TP + FP) | Cao hơn tốt  |
| Recall    | TP / (TP + FN) | Cao hơn tốt  |
| F1        | 2PR / (P + R)  | Cao hơn tốt  |

## Cải thiện đánh giá

### 1. Tạo trials khó hơn

Sửa hàm `build_trials()`:

```python
def build_trials_harder(spk2utts, ...):
    # Positive: khác file + cách nhau >= 2s
    # Negative: cùng domain/giới tính
    # Khử trùng lặp
    ...
```

### 2. Đánh giá DER end-to-end

Tạo script mới `eval_diarization_der.py`:

```python
from pyannote.metrics.diarization import DiarizationErrorRate

metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)
# Chạy pipeline trên audio thực
# Tính DER = Miss + FA + Confusion
```

### 3. Test trên data khó

- AMI corpus (meetings với overlap)
- CHiME challenges (nhiễu nền)
- CallHome (telephone speech)

## So sánh với baseline

| System                     | EER   | AUC    | DER    |
| -------------------------- | ----- | ------ | ------ |
| Baseline (Kaldi x-vectors) | ~5-8% | 0.98   | 15-20% |
| Pyannote 3.0               | ~3-5% | 0.99   | 8-12%  |
| Ours (current)             | ~0.3% | 0.9999 | ?      |

→ EER quá thấp cho thấy trials quá dễ, cần đánh giá DER thực tế.

## Tham khảo

- Kaldi SRE evaluation: https://kaldi-asr.org/doc/sre.html
- Pyannote metrics: https://pyannote.github.io/pyannote-metrics/
- NIST SRE protocols: https://www.nist.gov/itl/iad/mig/speaker-recognition-evaluation
