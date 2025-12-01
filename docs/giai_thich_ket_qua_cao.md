# Giải thích kết quả đánh giá cao bất thường

## Vấn đề

File `realtime/evaluation/eval_diarization.py` cho kết quả:

- EER ≈ 0.32% (rất thấp)
- AUC ≈ 0.9999 (gần hoàn hảo)
- Precision/Recall > 99%

## Nguyên nhân

### 1. Đánh giá SAI loại bài toán

**File đang đánh giá**: Speaker Verification (so khớp embeddings)

- Input: 2 embeddings
- Output: Same speaker? (Yes/No)
- Metric: EER, AUC

**Cần đánh giá**: Speaker Diarization (phân đoạn end-to-end)

- Input: Audio file
- Output: Who spoke when?
- Metric: DER (Diarization Error Rate)

### 2. Trials quá dễ

```python
# Positive pairs (genuine):
# - Có thể từ CÙNG FILE → cosine ≈ 1.0
# - Không yêu cầu khác phiên/cách xa

# Negative pairs (impostor):
# - Speakers rất khác nhau → dễ phân biệt
# - Không control domain/giới tính
```

### 3. Dataset sạch

JVS Corpus:

- Audio chất lượng cao (studio)
- Không overlap speakers
- Không nhiễu nền phức tạp
- Embeddings ECAPA-TDNN phân tách tốt

### 4. Không có điều kiện thực tế

- Không có overlap
- Không có cross-talk
- Không có channel mismatch
- Không có time constraints

## So sánh với thực tế

| Condition | Current eval | Real diarization |
| --------- | ------------ | ---------------- |
| Overlap   | Không        | Có (10-30%)      |
| Noise     | Sạch         | SNR 10-20dB      |
| Duration  | Ngắn (3-5s)  | Dài (phút/giờ)   |
| Speakers  | 2 files      | Nhiều đồng thời  |
| Channel   | Cố định      | Thay đổi         |

## Kết quả thực tế nên như thế nào?

### Speaker Verification (đúng cách)

| System            | EER   | Conditions          |
| ----------------- | ----- | ------------------- |
| X-vectors (Kaldi) | 5-8%  | VoxCeleb test       |
| ECAPA-TDNN        | 2-4%  | VoxCeleb test       |
| Ours (current)    | 0.32% | **JVS easy trials** |

### Speaker Diarization (DER)

| System       | DER    | Conditions        |
| ------------ | ------ | ----------------- |
| Pyannote 3.0 | 8-12%  | AMI meetings      |
| Baseline     | 15-20% | CallHome          |
| Ours         | **?**  | **Chưa đánh giá** |

## Cách sửa

### 1. Tạo trials khó hơn (Verification)

```python
def build_trials_harder(spk2utts):
    # Positive: bắt buộc
    # - Khác file
    # - Cách nhau >= 2 giây
    # - Không overlap

    # Negative: control
    # - Cùng domain
    # - Cùng giới tính (nếu có meta)
    # - Tránh quá xa nhau

    # Khử trùng lặp
    trials = list(set(trials))
```

### 2. Đánh giá DER end-to-end

```python
from pyannote.metrics.diarization import DiarizationErrorRate

# Chạy pipeline trên audio thực
hypothesis = pipeline(audio_file)

# So với ground truth
reference = load_rttm(reference_file)

# Tính DER với collar 0.25s
metric = DiarizationErrorRate(collar=0.25)
der = metric(reference, hypothesis)
```

### 3. Test trên dataset khó

- **AMI corpus**: meetings với overlap
- **CHiME**: nhiễu nền phức tạp
- **CallHome**: telephone speech
- **DIHARD**: wild conditions

## Workflow đúng

```
1. Train embeddings → SpeechBrain ECAPA-TDNN
2. Test verification → Hard trials (EER 2-5%)
3. Build diarization → VAD + Clustering + Resegmentation
4. Test diarization → DER on real audio (8-15%)
```

## Tài liệu tham khảo

- [NIST SRE Evaluation Plan](https://www.nist.gov/itl/iad/mig/speaker-recognition-evaluation)
- [Pyannote Metrics](https://pyannote.github.io/pyannote-metrics/)
- [Kaldi SRE Recipe](https://kaldi-asr.org/doc/sre.html)
- [DIHARD Challenge](https://dihardchallenge.github.io/dihard3/)

## Tóm tắt

**Kết quả hiện tại (EER 0.32%) KHÔNG phản ánh chất lượng thực tế** vì:

1. ✗ Đánh giá verification thay vì diarization
2. ✗ Trials quá dễ (có thể cùng file)
3. ✗ Dataset sạch, không có điều kiện thực tế
4. ✗ Chưa có đánh giá DER end-to-end

**Cần làm**:

1. ✓ Tạo trials khó hơn (khác file, cách xa, không overlap)
2. ✓ Đánh giá DER trên audio thực (với collar 0.25s)
3. ✓ Test trên dataset có nhiễu, overlap (AMI, CHiME)
4. ✓ Benchmark với các hệ thống khác (Pyannote, Kaldi)
