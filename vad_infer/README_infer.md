# NeMo Speaker Diarization với VAD Model đã Finetune

Script integrate VAD model đã finetune vào NeMo diarization pipeline hoàn chỉnh.

## 🎯 Điểm khác biệt so với file cũ

| Aspect                | `infer.py` (cũ)           | `infer_diarization.py` (mới)            |
| --------------------- | ------------------------- | --------------------------------------- |
| **VAD Processing**    | Xử lý toàn bộ audio 1 lần | Sliding windows (1.0s window, 0.5s hop) |
| **Frame Output**      | 1 frame → 1 segment       | Hàng nghìn frames → nhiều segments      |
| **Speaker Detection** | Không có                  | Full pipeline với TitaNet embeddings    |
| **Clustering**        | Không có                  | Spectral clustering                     |
| **Output**            | Chỉ có "speech" label     | speaker_0, speaker_1, ...               |

## 🚀 Sử dụng

### 1. Chỉ định số speakers (Khuyến nghị)

```bash
python infer_diarization.py --audio audio.wav --vad-model best_vad_model.nemo --num-speakers 2
```

### 2. Auto-detect số speakers

```bash
python infer_diarization.py --audio audio.wav --vad-model best_vad_model.nemo
```

### 3. Với các tham số tùy chỉnh

```bash
python infer_diarization.py \
    --audio conversation.wav \
    --vad-model best_vad_model.nemo \
    --speaker-model titanet_large \
    --num-speakers 3 \
    --vad-threshold 0.5 \
    --vad-onset 0.5 \
    --vad-offset 0.3 \
    --min-speech 0.1 \
    --min-silence 0.3
```

## 📊 Pipeline hoạt động

```
Audio (16kHz mono)
    ↓
[STEP 1] Load & Preprocess
    ↓
[STEP 2] VAD với Sliding Windows
    - Window: 1.0s (matching training)
    - Hop: 0.5s
    - Output: Speech segments
    ↓
[STEP 3] Create Embedding Windows
    - Window: 1.5s
    - Shift: 0.75s
    - Chia nhỏ VAD segments
    ↓
[STEP 4] Extract Speaker Embeddings
    - TitaNet model
    - 192-dim embeddings
    ↓
[STEP 5] Spectral Clustering
    - Cosine similarity affinity
    - Cluster embeddings
    - Map labels về VAD segments
    ↓
[STEP 6] Generate RTTM
    - Format: SPEAKER {file} 1 {start} {dur} <NA> <NA> speaker_{id} <NA> <NA>
```

## 🔧 Tham số

| Tham số           | Mặc định              | Mô tả                                  |
| ----------------- | --------------------- | -------------------------------------- |
| `--audio`         | _required_            | Đường dẫn audio input (.wav)           |
| `--vad-model`     | `best_vad_model.nemo` | VAD model đã finetune                  |
| `--speaker-model` | `titanet_large`       | Speaker embedding model                |
| `--num-speakers`  | `None`                | Số speakers (auto nếu không chỉ định)  |
| `--vad-threshold` | `0.5`                 | VAD decision threshold                 |
| `--vad-onset`     | `0.5`                 | VAD onset threshold                    |
| `--vad-offset`    | `0.3`                 | VAD offset threshold                   |
| `--min-speech`    | `0.1`                 | Tối thiểu độ dài speech (s)            |
| `--min-silence`   | `0.3`                 | Tối thiểu silence để chia segments (s) |
| `--output-dir`    | `diar_output`         | Thư mục output                         |

## 📈 Output mẫu

```
================================================================================
🎙️  NeMo Speaker Diarization (Custom VAD)
================================================================================
Audio: conversation.wav
VAD Model: best_vad_model.nemo
Speaker Model: titanet_large
Num Speakers: 2
VAD Threshold: 0.5
================================================================================

🖥️  Device: cuda

================================================================================
STEP 1: Load Audio
================================================================================
✅ Audio loaded: 45.30s, 16000Hz

================================================================================
STEP 2: Voice Activity Detection (VAD)
================================================================================
   Loading VAD model: best_vad_model.nemo
   ✅ VAD model loaded
   Running VAD with sliding windows (window=1.0s, hop=0.5s)...
      Processing 90 windows...
      Speech probability range: [0.002, 0.998]
      Mean: 0.654
      Raw segments: 18
      After filtering (min 0.1s): 15
      After merging (min silence 0.3s): 12
✅ VAD completed: 12 speech segments detected
   Total speech: 42.50s (93.8%)
   Segments preview:
     #1: 0.50s - 4.30s (3.80s)
     #2: 5.00s - 8.20s (3.20s)
     #3: 9.00s - 12.50s (3.50s)
     ... and 9 more

================================================================================
STEP 3: Create Embedding Windows
================================================================================
✅ Created 65 embedding windows from 12 VAD segments

================================================================================
STEP 4: Extract Speaker Embeddings
================================================================================
   Loading speaker model: titanet_large
   ✅ Speaker model loaded
   Extracting embeddings for 65 windows...
      50/65 embeddings extracted
      65/65 embeddings extracted
✅ Embeddings extracted: shape (65, 192)

================================================================================
STEP 5: Speaker Clustering
================================================================================
   Using specified: 2 speakers
   ✅ Clustering completed
✅ Final result: 2 speakers detected

================================================================================
STEP 6: Generate RTTM Output
================================================================================
✅ RTTM saved: diar_output/conversation_20251224_105432.rttm

================================================================================
📈 SUMMARY
================================================================================
Audio duration:     45.30s
Speech segments:    12
Total speech:       42.50s (93.8%)
Speakers detected:  2
  speaker_0: 6 segments, 21.30s
  speaker_1: 6 segments, 21.20s
================================================================================

📋 SEGMENTS:
#    Start      End        Duration   Speaker
--------------------------------------------------
1    0.500      4.300      3.800      speaker_0
2    5.000      8.200      3.200      speaker_1
3    9.000      12.500     3.500      speaker_0
4    13.200     16.800     3.600      speaker_1
...
```

## ❓ Troubleshooting

### Vấn đề 1: Vẫn chỉ detect 1 speaker

**Nguyên nhân**:

- VAD threshold quá cao → toàn bộ audio thành 1 segment
- Hoặc VAD model chưa đủ tốt để phân biệt speech/silence

**Giải pháp**:

```bash
# Thử giảm threshold
python infer_diarization.py --audio audio.wav --vad-model best_vad_model.nemo \
    --num-speakers 2 \
    --vad-threshold 0.3 \
    --vad-onset 0.3 \
    --vad-offset 0.2

# Hoặc giảm min-silence để không merge quá nhiều segments
python infer_diarization.py --audio audio.wav --vad-model best_vad_model.nemo \
    --num-speakers 2 \
    --min-silence 0.5
```

### Vấn đề 2: Quá nhiều segments nhỏ

**Giải pháp**:

```bash
# Tăng min-speech và min-silence
python infer_diarization.py --audio audio.wav --vad-model best_vad_model.nemo \
    --num-speakers 2 \
    --min-speech 0.3 \
    --min-silence 0.5
```

### Vấn đề 3: Speakers bị nhầm lẫn

**Nguyên nhân**: VAD model tốt nhưng audio chất lượng kém hoặc speakers rất giống nhau

**Giải pháp**:

- Luôn chỉ định `--num-speakers` chính xác
- Thử speaker model khác: `--speaker-model ecapa_tdnn`
- Đảm bảo audio 16kHz mono, ít noise

## 🔬 Phân tích VAD Model

### Kiểm tra VAD model có hoạt động tốt không?

Xem output STEP 2:

```
Speech probability range: [0.002, 0.998]  ← Tốt! Range rộng
Mean: 0.654                                ← Hợp lý
Raw segments: 18                           ← Nhiều segments (tốt!)
After filtering: 15
After merging: 12                          ← Vẫn còn nhiều (tốt!)
```

**Dấu hiệu VAD model chưa tốt:**

```
Speech probability range: [0.48, 0.52]    ← Quá hẹp, model không tự tin
Mean: 0.95                                 ← Quá cao, predict hầu hết là speech
Raw segments: 1                            ← Chỉ 1 segment (tệ!)
```

### Cải thiện VAD model

Nếu VAD model chưa tốt, cần:

1. **Train thêm epochs**: Tăng từ 20 → 50 epochs
2. **Điều chỉnh data augmentation** trong `finetune_nemo_vad.py`:

   ```python
   "augmentor": {
       "shift": {"prob": 0.5, "min_shift_ms": -10.0, "max_shift_ms": 10.0},
       "white_noise": {"prob": 0.7, "min_level": -90, "max_level": -40},
       "speed": {"prob": 0.5, "min_speed_rate": 0.95, "max_speed_rate": 1.05},
   }
   ```

3. **Balance dataset tốt hơn**: Đảm bảo tỷ lệ speech/background ~ 1:1

4. **Thử learning rate khác**:
   ```bash
   modal run finetune_nemo_vad.py --epochs 50 --learning-rate 5e-5
   ```

## 💡 Tips

1. **Luôn chỉ định `--num-speakers`** nếu biết trước
2. **Kiểm tra STEP 2 output** để đánh giá VAD model
3. **Nếu có nhiều segments nhưng vẫn 1 speaker** → vấn đề ở clustering, không phải VAD
4. **Test với audio khác nhau** để đánh giá tính tổng quát

## 📝 So sánh với Pipeline NeMo gốc

| Component          | NeMo Pipeline        | Pipeline này                  |
| ------------------ | -------------------- | ----------------------------- |
| VAD                | MarbleNet pretrained | **VAD model đã finetune**     |
| Speaker Embeddings | TitaNet              | TitaNet (giữ nguyên)          |
| Clustering         | Spectral             | Spectral (giữ nguyên)         |
| Lợi ích            | -                    | VAD tốt hơn cho domain cụ thể |

Bạn đã finetune VAD thành công! Chỉ cần integrate đúng cách vào pipeline.
