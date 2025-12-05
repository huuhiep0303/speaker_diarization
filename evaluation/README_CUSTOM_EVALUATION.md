# Custom Evaluation Scripts for NeMo Diarization

## Tổng quan

Đã custom lại 2 file evaluation để đánh giá NeMo diarization (main_nemo.py) trên dataset JVS:

1. **eval_asr.py** - Đánh giá chất lượng ASR (transcription)
2. **eval_diarization.py** - Đánh giá khả năng phân biệt speaker (embeddings)

## Dataset: JVS Corpus

**Đường dẫn**: `D:\WORKSPACE\VJ\speaker-diarization\realtime\dataset\jvs_ver1\jvs_ver1`

**Cấu trúc**:

```
jvs_ver1/
├── jvs001/
│   ├── falset10/
│   │   ├── transcripts_utf8.txt
│   │   └── wav24kHz16bit/*.wav
│   ├── nonpara30/
│   ├── parallel100/
│   └── whisper10/
├── jvs002/
├── ...
└── jvs100/
```

**Transcripts format** (transcripts_utf8.txt):

```
VOICEACTRESS100_001:また、東寺のように、五大明王と呼ばれる、主要な明王の中央に配されることも多い。
VOICEACTRESS100_002:ニューイングランド風は、牛乳をベースとした、白いクリームスープであり、ボストンクラムチャウダーとも呼ばれる。
```

---

## 1. eval_asr.py - ASR Evaluation

### Các thay đổi chính:

#### 1.1. Thêm NeMoASR class

```python
class NeMoASR(BaseASR):
    """NeMo Speaker Diarization with Whisper ASR"""

    def __init__(self, whisper_model_name="base", device="cpu"):
        # Load SimpleSpeakerDiarization từ main_nemo.py
        from main_nemo import SimpleSpeakerDiarization

        self.diarizer = SimpleSpeakerDiarization(
            pretrained_speaker_model="titanet_large",
            whisper_model_name=whisper_model_name
        )

    def transcribe(self, audio_path):
        # Process audio với NeMo diarization
        result = self.diarizer.process_audio(audio_path, transcribe=True)
        # Merge tất cả transcripts từ tất cả speakers
        # Sort theo thời gian và concatenate
```

#### 1.2. Thêm load_jvs_dataset() function

- Tự động scan thư mục JVS dataset
- Đọc transcripts từ transcripts_utf8.txt
- Match với audio files (.wav)
- Trả về list of dicts: `{'wav_path', 'transcript', 'speaker', 'category'}`

#### 1.3. Update argument parser

```python
--model nemo              # Thêm option 'nemo'
--dataset ../dataset/jvs_ver1/jvs_ver1  # Default path
--max_samples N           # Giới hạn số samples để test nhanh
```

### Cách sử dụng:

#### Đánh giá NeMo trên toàn bộ JVS dataset:

```bash
cd D:\WORKSPACE\VJ\speaker-diarization\realtime\evaluation
python eval_asr.py --model nemo --device cpu
```

#### Test nhanh với 50 samples:

```bash
python eval_asr.py --model nemo --device cpu --max_samples 50
```

#### Đánh giá với Whisper model size khác:

```bash
python eval_asr.py --model nemo --whisper_size small --device cuda
```

#### So sánh với model khác:

```bash
# Whisper
python eval_asr.py --model whisper --whisper_size base --device cpu

# SenseVoice
python eval_asr.py --model sensevoice --device cpu

# SenseVoice + SpeechBrain
python eval_asr.py --model sensevoice-speechbrain --device cpu
```

### Kết quả:

**Output files** (trong `eval_results/`):

- `eval_asr_nemo-whisper-base_checkpoint.csv` - Chi tiết từng file
- `eval_asr_nemo-whisper-base_summary.json` - Tổng kết metrics

**Metrics đánh giá**:

- **WER** (Word Error Rate) - Sử dụng Sudachi tokenizer cho tiếng Nhật
- **CER** (Character Error Rate) - Đánh giá theo ký tự
- **RTF** (Real-Time Factor) - Tốc độ xử lý so với realtime
  - RTF < 1.0: Nhanh hơn realtime
  - RTF = 1.0: Bằng realtime
  - RTF > 1.0: Chậm hơn realtime

---

## 2. eval_diarization.py - Speaker Verification

### ⚠️ LƯU Ý QUAN TRỌNG VỀ SPEAKER EMBEDDINGS

**Vấn đề phát hiện**: Ba models (Whisper, SenseVoice, SenseVoice+SpeechBrain) cho kết quả **GIỐNG HỆT NHAU** vì chúng **ĐỀU DÙNG CÙNG 1 MODEL** cho speaker embeddings!

**Nguyên nhân**:

- **Whisper**: Không có speaker embedding model riêng → Dùng SpeechBrain ECAPA-TDNN
- **SenseVoice**: Không có speaker embedding model riêng → Dùng SpeechBrain ECAPA-TDNN
- **SenseVoice+SpeechBrain**: Dùng SpeechBrain ECAPA-TDNN
- **NeMo**: Dùng TitaNet Large (KI trúc khác hoàn toàn)

**Giải pháp**: Chỉ đánh giá **2 models thực sự khác nhau**:

1. **SpeechBrain ECAPA-TDNN** (dùng chung bởi Whisper/SenseVoice/SenseVoice+SpeechBrain)
2. **NeMo TitaNet Large** (kiến trúc hoàn toàn khác)

### Các thay đổi chính:

#### 2.1. Update MODELS config

```python
MODELS = {
    "speechbrain": {
        "name": "SpeechBrain ECAPA-TDNN",
        "note": "Used by Whisper, SenseVoice, SenseVoice+SpeechBrain"
    },
    "nemo": {
        "name": "NeMo TitaNet Large",
        "note": "Different architecture from SpeechBrain"
    }
}
```

#### 2.2. Chỉ extract 2 loại embeddings

- **speechbrain**: SpeechBrain ECAPA-TDNN embeddings
- **nemo**: NeMo TitaNet Large embeddings

#### 2.3. Chỉ evaluate 2 models

Code đã được sửa để:

- Chỉ evaluate "speechbrain" và "nemo"
- In ra NOTE để người dùng hiểu rõ
- Plot chỉ 2 curves thay vì 4

### Cách sử dụng:

#### 2.1. Thêm NeMo vào MODELS config

```python
MODELS = {
    "whisper": {...},
    "sensevoice": {...},
    "sensevoice-speechbrain": {...},
    "nemo": {
        "name": "NeMo TitaNet Large",
        "script": "main_nemo.py"
    }
}
```

#### 2.2. Thêm load_nemo_model() và extract_nemo_embedding()

```python
def load_nemo_model():
    """Load NeMo TitaNet Large speaker recognition model"""
    from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel

    model = EncDecSpeakerLabelModel.from_pretrained(
        model_name="titanet_large"
    )
    return model

def extract_nemo_embedding(audio_path, model):
    """Extract speaker embedding using NeMo TitaNet Large"""
    # Load audio
    # Resample to 16kHz
    # Extract embedding
    # Normalize
    return embedding
```

#### 2.3. Update extract_all_embeddings()

- Extract cả SpeechBrain và NeMo embeddings
- Cache structure: `{file_path: {'speechbrain': emb, 'nemo': emb}}`
- Tự động load từ cache nếu có

#### 2.4. Update evaluation và plotting

- Hỗ trợ đánh giá 4 models: whisper, sensevoice, sensevoice-speechbrain, **nemo**
- Plot ROC/DET/PR curves với màu purple cho NeMo
- Save kết quả riêng cho từng model

#### 2.5. Update default dataset path

```python
--dataset ../dataset/jvs_ver1/jvs_ver1  # Đúng với cấu trúc thực tế
```

### Cách sử dụng:

#### Đánh giá tất cả models (bao gồm NeMo):

```bash
cd D:\WORKSPACE\VJ\speaker-diarization\realtime\evaluation
python eval_diarization.py
```

#### Chỉ định dataset path:

```bash
python eval_diarization.py --dataset D:\WORKSPACE\VJ\speaker-diarization\realtime\dataset\jvs_ver1\jvs_ver1
```

#### Tùy chỉnh số trials:

```bash
python eval_diarization.py --max_genuine_per_spk 100 --impostor_per_spk 200
```

#### Xóa cache và re-extract embeddings:

```bash
python eval_diarization.py --clear_cache
```

#### Disable cache:

```bash
python eval_diarization.py --no_cache
```

### Kết quả:

**Output files** (trong `eval_results/`):

- `roc_curves.png` - ROC curves comparison (**2 models**: SpeechBrain vs NeMo)
- `det_curves.png` - DET curves comparison
- `precision_recall_curves.png` - PR curves comparison
- `eval_diarization_speechbrain_results.json` - Chi tiết SpeechBrain model
- `eval_diarization_nemo_results.json` - Chi tiết NeMo model
- `result.log` - Tổng kết 2 models

**Sample result.log**:

```
======================================================================
Speaker Embedding Comparison
NOTE: SpeechBrain used by Whisper/SenseVoice/SenseVoice+SpeechBrain
======================================================================

=== Evaluating speechbrain embeddings ===
Computing metrics on 15000 valid trials
EER: 15.57% | FAR@EER: 15.59% | FRR@EER: 15.56% | Thr(EER): 0.3790
...

=== Evaluating nemo embeddings ===
Computing metrics on 15000 valid trials
EER: 12.34% | FAR@EER: 12.35% | FRR@EER: 12.33% | Thr(EER): 0.4123
...
```

**Lưu ý**: SpeechBrain và NeMo sẽ cho kết quả **KHÁC NHAU** vì dùng 2 kiến trúc embedding khác nhau.

**Cached embeddings** (trong `eval_cache/`):

- `embeddings_cache_*.pkl` - Cached embeddings (tránh re-extract)

**Metrics đánh giá**:

- **EER** (Equal Error Rate): FAR = FRR (càng thấp càng tốt)
- **FAR** (False Acceptance Rate): Nhận nhầm người khác
- **FRR** (False Rejection Rate): Từ chối người đúng
- **AUC** (Area Under Curve): Diện tích dưới ROC (càng cao càng tốt)
- **Precision, Recall, F1-score**

---

## Lưu ý quan trọng

### Dependencies cần cài đặt:

```bash
# Cho eval_asr.py
pip install jiwer sudachipy sudachidict_core
pip install faster-whisper  # Cho Whisper
pip install funasr  # Cho SenseVoice
pip install speechbrain  # Cho SpeechBrain

# Cho eval_diarization.py
pip install torch torchaudio
pip install speechbrain
pip install scikit-learn matplotlib
pip install soundfile librosa

# Cho NeMo
pip install nemo_toolkit[asr]
pip install openai-whisper
```

### Troubleshooting:

1. **Import error từ main_nemo.py**:

   - Đảm bảo main_nemo.py trong folder `realtime/`
   - Script tự động thêm parent directory vào sys.path

2. **Dataset not found**:

   - Kiểm tra đường dẫn: `dataset/jvs_ver1/jvs_ver1/jvsXXX/`
   - Script tự động thử các alternative paths

3. **Model download**:

   - NeMo TitaNet Large sẽ tự động download lần đầu (~400MB)
   - Whisper models cũng tự động download

4. **Memory issues**:

   - Dùng `--max_samples` để giới hạn số samples
   - Dùng smaller Whisper model (tiny, base)
   - Process từng speaker một (modify code)

5. **Cache issues**:
   - Dùng `--clear_cache` để xóa cache cũ
   - Cache lưu trong `eval_cache/`

---

## Kết quả mong đợi

### ASR Evaluation (eval_asr.py):

- NeMo với Whisper base: WER ~5-15%, CER ~3-10%
- So sánh với Whisper standalone, SenseVoice

### Diarization Evaluation (eval_diarization.py):

- **SpeechBrain ECAPA-TDNN**: EER ~15%, AUC ~0.93
- **NeMo TitaNet Large**: EER ~10-15%, AUC ~0.93-0.95
- Kết quả sẽ **KHÁC NHAU** vì 2 models dùng kiến trúc embedding khác nhau

### Thời gian chạy:

- **eval_asr.py**: ~2-5 giây/file với NeMo (depends on audio length)
- **eval_diarization.py**:
  - Extract embeddings: ~0.5-1 giây/file
  - Evaluation: vài giây (sau khi có embeddings)
  - Toàn bộ JVS (~15,000 files): ~2-4 giờ lần đầu (có cache)
  - Lần sau: vài phút (dùng cache)

---

## Ví dụ workflow hoàn chỉnh

```bash
# 1. Chuyển đến thư mục evaluation
cd D:\WORKSPACE\VJ\speaker-diarization\realtime\evaluation

# 2. Test nhanh với 50 samples
python eval_asr.py --model nemo --max_samples 50

# 3. Đánh giá ASR đầy đủ
python eval_asr.py --model nemo --device cpu

# 4. Đánh giá speaker verification
python eval_diarization.py --dataset ../dataset/jvs_ver1/jvs_ver1

# 5. Xem kết quả
# - eval_results/eval_asr_nemo-whisper-base_summary.json
# - eval_results/roc_curves.png
# - eval_results/det_curves.png
# - eval_results/result.log
```

---

## Summary

✅ **eval_asr.py**:

- Thêm NeMoASR class
- Thêm load_jvs_dataset() để load trực tiếp từ JVS directory
- Support --model nemo option
- Default dataset: ../dataset/jvs_ver1/jvs_ver1

✅ **eval_diarization.py**:

- Thêm NeMo TitaNet Large vào models
- Thêm extract_nemo_embedding() function
- Update extract_all_embeddings() để extract cả NeMo và SpeechBrain
- Update plots để show cả 4 models
- Default dataset: ../dataset/jvs_ver1/jvs_ver1

🎯 **Mục đích**: So sánh NeMo diarization với các models khác về:

- Chất lượng transcription (WER, CER, RTF)
- Khả năng phân biệt speaker (EER, AUC, F1)
