# Real-time Speaker Diarization System

Hệ thống nhận diện người nói và chuyển đổi giọng nói thành văn bản thời gian thực với 4 models khác nhau.

---

## 🚀 Quick Start (Cho người mới)

### Bước 1: Cài đặt môi trường

```bash
# Clone repo và cd vào thư mục
cd d:\WORKSPACE\VJ\speaker-diarization\realtime

# Tạo virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Cài đặt dependencies cơ bản
pip install -r requirements.txt
```

### Bước 2: Chọn model và chạy test

**Recommended cho người mới:** SenseVoice + SpeechBrain (nhanh nhất, accurate nhất)

```bash
pip install -r requirements_sen_voice.txt
python senvoi_spebrai_fixed.py
```

**Hoặc thử NeMo model** (flexible nhất, có transcription):

```bash
pip install nemo_toolkit['all'] openai-whisper
python main_nemo.py
```

### Bước 3: Test với file audio của bạn

Sửa tên file trong code:

```python
# Trong main_nemo.py hoặc test_nemo.py, dòng cuối:
audio_file = "your_audio.wav"  # Thay đổi tên file ở đây
```

Chạy lại:

```bash
python main_nemo.py
```

### Output mong đợi:

```
📊 DIARIZATION RESULTS:
   Number of speakers detected: 2
   Total segments: 63

⏱️  SPEAKER TIME DISTRIBUTION:
   SPEAKER_00: 25.30s (52.9%)
   SPEAKER_01: 22.50s (47.1%)

📝 TRANSCRIPTS BY SPEAKER:
   SPEAKER_00:
   [  0.00s -   3.50s] こんにちは、元気ですか？
   ...
```

---

## 📁 Cấu trúc thư mục

```
realtime/
├── README.md                          # Tài liệu này
├── requirements.txt                   # Dependencies chung
├── requirements_sen_voice.txt         # Dependencies cho SenseVoice
│
├── realtime_diarization_improved.py   # Model 1: Whisper + SpeechBrain
├── sen_voice.py                      # Model 2: SenseVoice
├── senvoi_spebrai_fixed.py           # Model 3: SenseVoice + SpeechBrain
├── main_nemo.py                      # Model 4: NeMo TitaNet + Whisper (Full)
├── test_nemo.py                      # Model 4: Simplified version
├── diar_infer_config.yaml            # Config cho NeMo ClusteringDiarizer
│
├── dataset/                          # Dataset JVS để đánh giá
│   └── jvs_ver1/
├── pretrained_models/                # Models đã tải về
├── evaluation/                       # Scripts đánh giá và so sánh
│   ├── eval_asr.py                  # Đánh giá ASR (Speech Recognition)
│   ├── eval_diarization.py          # Đánh giá Diarization (Real Evaluation)
│   ├── model_wrappers.py            # Wrapper classes cho các models
│   ├── test_eval.py                 # Test script cho evaluation
│   ├── compared.py                  # So sánh kết quả ASR
│   ├── compare_diarization.py       # So sánh kết quả Diarization
│   ├── eval_results/                # Kết quả đánh giá
│   └── *.bat                        # Batch files để chạy dễ dàng
│
├── tmp_model/                        # Models tạm thời
├── venv/                            # Virtual environment
└── *.json                           # Output files từ các lần chạy
```

## 🚀 Cài đặt và Khởi tạo

### 1. Tạo Virtual Environment

```bash
cd realtime
python -m venv venv

# Kích hoạt virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. Cài đặt Dependencies

```bash
# Cài đặt packages cơ bản
pip install -r requirements.txt

# Cài đặt packages cho SenseVoice
pip install -r requirements_sen_voice.txt

# Packages bổ sung cho evaluation
pip install pandas matplotlib seaborn scikit-learn

# Cài đặt NeMo cho Model 4 (NeMo TitaNet + Whisper)
pip install nemo_toolkit['all']
# hoặc cài đặt từ source nếu cần
pip install Cython
pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[all]
```

**Dependencies chính cho NeMo model:**

- `nemo_toolkit` - NeMo framework
- `torch`, `torchaudio` - PyTorch
- `openai-whisper` - Whisper ASR
- `soundfile`, `librosa` - Audio processing
- `scikit-learn`, `scipy` - Clustering và metrics
- `numpy` - Numerical computing

## 🎯 Các Models Available

### 1. **Whisper + SpeechBrain** (`realtime_diarization_improved.py`)

- **ASR**: Whisper Small
- **Speaker Diarization**: SpeechBrain ECAPA-TDNN
- **Ưu điểm**: Độ chính xác ASR cao, diarization tốt
- **Nhược điểm**: Tốc độ chậm nhất (RTF ~2.7)

### 2. **SenseVoice** (`sen_voice.py`)

- **ASR**: SenseVoice Small (FunAudioLLM)
- **Speaker Diarization**: Không có
- **Ưu điểm**: Tốc độ nhanh, độ chính xác ASR cao
- **Nhược điểm**: Không phân biệt được người nói

### 3. **SenseVoice + SpeechBrain** (`senvoi_spebrai_fixed.py`)

- **ASR**: SenseVoice Small
- **Speaker Diarization**: SpeechBrain ECAPA-TDNN
- **Ưu điểm**: **Tốt nhất** - Tốc độ nhanh nhất (RTF ~0.35), độ chính xác cao nhất
- **Nhược điểm**: Cài đặt phức tạp hơn

### 4. **NeMo TitaNet + Whisper** (`main_nemo.py`, `test_nemo.py`) 🆕

- **ASR**: Whisper (base/small/medium)
- **Speaker Diarization**: NeMo TitaNet-Large (192-dim embeddings)
- **Approach**: Segmentation-based clustering
  - Sliding window segmentation (1.5s window, 0.75s shift, 50% overlap)
  - Extract speaker embeddings per segment
  - AgglomerativeClustering với cosine distance
  - Auto-detect số speakers hoặc cố định trước
- **Ưu điểm**:
  - Embedding-based approach linh hoạt
  - Có thể sử dụng speaker memory cho tracking
  - Transcription kèm diarization
  - Kết quả chi tiết với timestamps
- **Nhược điểm**:
  - Cần cài đặt NeMo framework
  - Tốc độ phụ thuộc vào Whisper model size

**Files liên quan:**

- `main_nemo.py`: Full implementation với transcription
- `test_nemo.py`: Simplified version
- `diar_infer_config.yaml`: Config cho NeMo ClusteringDiarizer (alternative approach)

## 🎤 Chạy Real-time Recognition

### Chạy từng model riêng lẻ:

```bash
# Model 1: Whisper + SpeechBrain
python realtime_diarization_improved.py

# Model 2: SenseVoice only
python sen_voice.py

# Model 3: SenseVoice + SpeechBrain (recommended)
python senvoi_spebrai_fixed.py

# Model 4: NeMo TitaNet + Whisper (với transcription)
python main_nemo.py
# hoặc simplified version
python test_nemo.py
```

### Output:

- Console: Hiển thị real-time transcript
- JSON file: Lưu chi tiết với timestamp (format: `[model]_output_YYYYMMDD_HHMMSS.json`)

---

## 📖 Chi tiết Model NeMo TitaNet + Whisper

### Kiến trúc và Pipeline

File `main_nemo.py` (và `test_nemo.py`) implement một approach khác biệt so với các models trước:

#### **1. Segmentation-based Approach**

Thay vì xử lý toàn bộ audio, chia audio thành các segments nhỏ:

```python
# Sliding window parameters
window_length_sec = 1.5   # Độ dài mỗi segment
shift_length_sec = 0.75   # Shift giữa các segments (50% overlap)
```

**Ví dụ**: Audio 47.8s → ~63 segments

#### **2. Speaker Embedding Extraction**

Sử dụng NeMo TitaNet-Large model để extract embeddings:

- Input: Audio segment (1.5s)
- Output: 192-dimensional embedding vector
- Model: `titanet_large` (pretrained from NeMo)

```python
speaker_model = EncDecSpeakerLabelModel.from_pretrained(
    model_name="titanet_large"
)
```

#### **3. Clustering cho Speaker Detection**

Sử dụng AgglomerativeClustering để nhóm embeddings:

```python
clusterer = AgglomerativeClustering(
    n_clusters=num_speakers,  # Auto-detect hoặc cố định
    metric='cosine',           # Cosine distance
    linkage='average'          # Average linkage
)
```

**Auto-detection**: Thử từ 1-8 speakers, chọn số tốt nhất dựa trên silhouette score

#### **4. Transcription với Whisper**

Sau khi có speaker labels, transcribe từng speaker segment:

```python
whisper_model = whisper.load_model("base")  # tiny/base/small/medium/large
result = whisper_model.transcribe(segment_audio, language='ja')
```

**Merge consecutive segments**: Gộp các segments liên tiếp của cùng speaker trước khi transcribe

### Cách chạy và Output

#### **Khởi tạo:**

```python
diarizer = SimpleSpeakerDiarization(
    pretrained_speaker_model="titanet_large",
    window_length_sec=1.5,
    shift_length_sec=0.75,
    similarity_threshold=0.7,
    whisper_model_name="base"  # tiny, base, small, medium, large
)
```

#### **Process audio:**

```python
result = diarizer.process_audio(
    "audio.wav",
    num_speakers=None,    # Auto-detect hoặc cố định (e.g., 2)
    max_speakers=8,       # Số speakers tối đa khi auto-detect
    transcribe=True       # Enable transcription
)
```

#### **Output Structure:**

```python
{
    'num_speakers': 2,                    # Số speakers detected
    'speaker_labels': ['SPEAKER_00', ...], # Labels cho mỗi segment
    'time_ranges': [(0.0, 1.5), ...],     # Time range của mỗi segment
    'embeddings': np.ndarray,              # Speaker embeddings
    'speaker_times': {                     # Tổng thời gian của mỗi speaker
        'SPEAKER_00': 25.3,
        'SPEAKER_01': 22.5
    },
    'transcripts': {                       # Transcripts theo speaker
        'SPEAKER_00': [
            {
                'start': 0.0,
                'end': 3.5,
                'duration': 3.5,
                'text': 'こんにちは、元気ですか？'
            },
            ...
        ],
        'SPEAKER_01': [...]
    }
}
```

#### **Console Output Example:**

```
🚀 Initializing NeMo Speaker Embedding Model...
   Device: cuda
   Model: titanet_large
   Window: 1.5s, Shift: 0.75s
✅ Model loaded successfully!

🎯 Loading Whisper ASR model (base)...
✅ Whisper model loaded successfully!

🎤 Processing audio...
  Audio duration: 47.80s
  🔪 Segmented into 63 segments
  📊 Extracted 63 embeddings, shape: (63, 192)
  🎯 Auto-detected 2 speakers (max tried: 8)
  👥 Detected 2 speakers in audio

🎤 Transcribing audio segments...
  Found 15 merged speaker segments
  [1/15] SPEAKER_00 (0.00s-3.50s): こんにちは、元気ですか？...
  [2/15] SPEAKER_01 (3.75s-7.20s): はい、元気です。ありがとう...
  ✅ Transcription completed

📊 DIARIZATION RESULTS:
   Number of speakers detected: 2
   Total segments: 63

⏱️  SPEAKER TIME DISTRIBUTION:
   SPEAKER_00: 25.30s (52.9%)
   SPEAKER_01: 22.50s (47.1%)

📝 TRANSCRIPTS BY SPEAKER:

   SPEAKER_00:
   ────────────────────────────────────────────────────────────────────────────────
   [  0.00s -   3.50s] こんにちは、元気ですか？
   [  7.50s -  11.20s] 今日は天気がいいですね。
   ...

   SPEAKER_01:
   ────────────────────────────────────────────────────────────────────────────────
   [  3.75s -   7.20s] はい、元気です。ありがとう。
   [ 11.50s -  15.80s] そうですね、散歩に行きましょう。
   ...
```

### So sánh với NeMo ClusteringDiarizer

Project cũng có sẵn config cho NeMo ClusteringDiarizer (`diar_infer_config.yaml`), nhưng approach trong `main_nemo.py` khác:

| Aspect                  | NeMo ClusteringDiarizer   | main_nemo.py (Custom)              |
| ----------------------- | ------------------------- | ---------------------------------- |
| **Approach**            | End-to-end pipeline       | Modular: Segment → Embed → Cluster |
| **Flexibility**         | Cố định theo config       | Linh hoạt, dễ customize            |
| **Speaker Memory**      | Không có                  | Có (optional)                      |
| **Transcription**       | Tách riêng                | Tích hợp sẵn                       |
| **Config**              | Nhiều parameters phức tạp | Đơn giản, intuitive                |
| **Auto-detect Speaker** | Dựa trên threshold        | Clustering-based với scoring       |

### Parameters Tuning

#### **Segmentation:**

```python
window_length_sec = 1.5   # Tăng → ít segments hơn, embeddings stable hơn
shift_length_sec = 0.75   # Giảm → overlap nhiều hơn, accurate hơn nhưng chậm hơn
```

#### **Clustering:**

```python
max_speakers = 8          # Giới hạn số speakers để thử khi auto-detect
num_speakers = 2          # Cố định nếu biết trước (bỏ qua auto-detect)
```

#### **Speaker Matching:**

```python
similarity_threshold = 0.7       # Ngưỡng cosine similarity để match speaker
min_similarity_gap = 0.15        # Gap tối thiểu giữa best và 2nd-best
embedding_update_weight = 0.3    # Trọng số EMA khi update embedding
```

#### **Whisper:**

```python
whisper_model_name = "base"  # tiny (fastest) → large (most accurate)
language = 'ja'              # hoặc None để auto-detect
```

### Khi nào dùng Model này?

**Sử dụng NeMo TitaNet + Whisper khi:**

✅ Cần flexibility cao trong pipeline  
✅ Muốn customize clustering algorithm  
✅ Cần speaker memory/tracking across multiple audio files  
✅ Muốn control chi tiết từng bước (segmentation, embedding, clustering)  
✅ Research hoặc experiment với speaker diarization

**Không nên dùng khi:**

❌ Cần solution đơn giản, plug-and-play  
❌ Ưu tiên tốc độ tối đa (SenseVoice + SpeechBrain nhanh hơn)  
❌ Không cần customize pipeline  
❌ Production environment cần stability cao

---

## 📊 Đánh giá và So sánh Models

### 1. Đánh giá ASR (Speech Recognition)

```bash
cd evaluation

# Chạy đánh giá ASR cho cả 3 models
eval_asr.bat

# Hoặc Python trực tiếp
python eval_asr.py --max_files 100  # Test nhanh với 100 files
python eval_asr.py                  # Full dataset (~14,000+ files)
```

### 2. Đánh giá Diarization (Real Evaluation)

```bash
cd evaluation

# Test với vài files trước
python test_eval.py

# Chạy đánh giá diarization cho 1 model
python eval_diarization.py --models sensevoice --max_files 50

# Chạy full evaluation cho cả 3 models
python eval_diarization.py --max_files 100  # Test nhanh
python eval_diarization.py                  # Full dataset (~14,000+ files)

# Chạy cho models cụ thể
python eval_diarization.py --models whisper-speechbrain sensevoice-speechbrain
```

### 3. So sánh kết quả

```bash
# So sánh kết quả ASR
python compared.py

# So sánh kết quả Diarization
python compare_diarization.py
```

## 📈 Kết quả Đánh giá

### ASR Performance (trên JVS dataset):

| Model                        | WER (%)   | CER (%)  | RTF       | Real-time |
| ---------------------------- | --------- | -------- | --------- | --------- |
| **SenseVoice + SpeechBrain** | **11.70** | **8.32** | **0.355** | ✅        |
| SenseVoice                   | 13.89     | 10.08    | 0.805     | ✅        |
| Whisper + SpeechBrain        | 16.12     | 12.58    | 2.749     | ❌        |

### Diarization Performance:

| Model                        | DER (%)  | F1 (%)    | RTF       | Real-time |
| ---------------------------- | -------- | --------- | --------- | --------- |
| **SenseVoice + SpeechBrain** | **5.33** | **91.84** | **0.049** | ✅        |
| Whisper + SpeechBrain        | 10.36    | 93.54     | 0.168     | ✅        |
| SenseVoice                   | 87.27    | 76.58     | 0.084     | ✅        |

### 🏆 **Kết luận**:

**SenseVoice + SpeechBrain** là model tốt nhất với:

- WER thấp nhất (11.70%)
- DER thấp nhất (5.33%)
- RTF nhanh nhất (0.355 cho ASR, 0.049 cho diarization)
- Khả năng real-time tốt nhất

## 🔧 Troubleshooting

### Lỗi thường gặp:

1. **ImportError với SpeechBrain/FunASR**:

```bash
pip install --upgrade speechbrain funasr
pip install soundfile librosa torch torchaudio
```

2. **ImportError với NeMo**:

```bash
pip install nemo_toolkit['all']
# Nếu lỗi Cython
pip install Cython
# Nếu lỗi hydra-core
pip install hydra-core omegaconf
```

3. **ConfigAttributeError trong NeMo ClusteringDiarizer**:

Kiểm tra file `diar_infer_config.yaml` có đầy đủ các parameters:

```yaml
num_workers: 0 # Windows compatibility
batch_size: 32
sample_rate: 16000
window_length_in_sec: [1.5, 1.25, 1.0, 0.75, 0.5]
shift_length_in_sec: [0.75, 0.625, 0.5, 0.375, 0.25]
multiscale_weights: [1, 1, 1, 1, 1]
verbose: true
smoothing: "median" # Phải là string, không phải boolean
overlap: 0.5
collar: 0.25
ignore_overlap: true
max_rp_threshold: 0.15 # Giảm để tránh over-segmentation
```

4. **CUDA not available**:

- Models sẽ tự động chuyển về CPU
- Tốc độ chậm hơn nhưng vẫn hoạt động

5. **Microphone không hoạt động**:

```bash
pip install sounddevice
# Kiểm tra device available
python -c "import sounddevice as sd; print(sd.query_devices())"
```

6. **Memory errors**:

- Giảm `CHUNK_SEC` trong config
- Sử dụng CPU thay vì CUDA
- Giảm `window_length_sec` trong NeMo model
- Dùng Whisper model nhỏ hơn (tiny/base thay vì large)

7. **Whisper transcription chậm**:

- Dùng model nhỏ hơn: `whisper_model_name="tiny"` hoặc `"base"`
- Enable FP16 nếu có GPU: `fp16=True`
- Giảm số merged segments (tăng threshold để merge nhiều hơn)

### Cấu hình tùy chỉnh:

Sửa các thông số trong file Python:

**Cho models cũ:**

```python
SAMPLE_RATE = 16000    # Sample rate
CHUNK_SEC = 3.0        # Độ dài mỗi chunk (giây)
OVERLAP_SEC = 0.3      # Overlap giữa các chunk
DEVICE = "cpu"         # hoặc "cuda"
```

**Cho NeMo model (main_nemo.py):**

```python
diarizer = SimpleSpeakerDiarization(
    pretrained_speaker_model="titanet_large",  # hoặc "ecapa_tdnn", "speakerverification_speakernet"
    window_length_sec=1.5,          # Độ dài segment
    shift_length_sec=0.75,          # Shift giữa segments
    similarity_threshold=0.7,        # Ngưỡng match speaker
    embedding_update_weight=0.3,     # Trọng số EMA
    min_similarity_gap=0.15,         # Gap tối thiểu
    whisper_model_name="base"        # tiny/base/small/medium/large
)

# Process audio
result = diarizer.process_audio(
    "audio.wav",
    num_speakers=None,    # None = auto-detect, hoặc cố định (e.g., 2)
    max_speakers=8,       # Số speakers tối đa khi auto-detect
    transcribe=True       # True = có transcription, False = chỉ diarization
)
```

## 📝 Output Format

### JSON Output Structure:

```json
{
  "start_time": "2025-11-28T10:30:00.000000",
  "model": "SenseVoice + SpeechBrain",
  "device": "cpu",
  "sample_rate": 16000,
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 3.2,
      "duration": 3.2,
      "text": "Xin chào, tôi là người nói số một",
      "speaker": "speaker_1",
      "confidence": 0.95
    },
    {
      "start_time": 3.5,
      "end_time": 6.8,
      "duration": 3.3,
      "text": "Và tôi là người nói số hai",
      "speaker": "speaker_2",
      "confidence": 0.92
    }
  ]
}
```

## 🛠 Development

### Thêm model mới:

1. Tạo file Python mới theo template
2. Implement interface tương tự các model có sẵn
3. Thêm vào `MODELS` dict trong evaluation scripts
4. Chạy evaluation để so sánh

### Customize evaluation:

- Sửa `eval_asr.py` và `eval_diarization.py`
- Thêm metrics mới vào comparison scripts
- Tùy chỉnh visualizations trong plot functions

## 📚 References

- **Whisper**: OpenAI Whisper ASR model - https://github.com/openai/whisper
- **SenseVoice**: FunAudioLLM SenseVoice Small - https://github.com/FunAudioLLM/SenseVoice
- **SpeechBrain**: ECAPA-TDNN speaker recognition - https://speechbrain.github.io/
- **NeMo**: NVIDIA NeMo Toolkit - https://github.com/NVIDIA/NeMo
  - TitaNet-Large: Speaker verification model với 192-dim embeddings
  - ClusteringDiarizer: End-to-end diarization pipeline
- **JVS Dataset**: Japanese Versatile Speech corpus - https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus
- **Evaluation Metrics**:
  - WER (Word Error Rate), CER (Character Error Rate) - ASR metrics
  - DER (Diarization Error Rate) - Speaker diarization metric
  - RTF (Real-Time Factor) - Speed metric
  - F1-score - Classification metric

### Papers & Resources:

- **TitaNet**: "TitaNet: Neural Model for speaker representation with 1D Depth-wise separable convolutions and global context" (NVIDIA)
- **ECAPA-TDNN**: "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification"
- **Whisper**: "Robust Speech Recognition via Large-Scale Weak Supervision" (OpenAI, 2022)
- **AgglomerativeClustering**: Scikit-learn hierarchical clustering - https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering

---

## 🔍 So sánh chi tiết các Models

| Feature                      | Whisper + SpeechBrain | SenseVoice | SenseVoice + SpeechBrain | **NeMo TitaNet + Whisper** |
| ---------------------------- | --------------------- | ---------- | ------------------------ | -------------------------- |
| **ASR Model**                | Whisper Small         | SenseVoice | SenseVoice               | Whisper (configurable)     |
| **Diarization Model**        | ECAPA-TDNN            | None       | ECAPA-TDNN               | TitaNet-Large              |
| **Embedding Dimension**      | 192                   | -          | 192                      | **192**                    |
| **Speaker Detection**        | Threshold-based       | N/A        | Threshold-based          | **Clustering-based**       |
| **Auto-detect Speakers**     | ❌                    | ❌         | ❌                       | **✅**                     |
| **Speaker Memory/Tracking**  | ❌                    | ❌         | ❌                       | **✅**                     |
| **Transcription Integrated** | ✅                    | ✅         | ✅                       | **✅**                     |
| **Customizable Pipeline**    | ⚠️                    | ⚠️         | ⚠️                       | **✅✅**                   |
| **Speed (RTF - ASR)**        | 2.7 (slow)            | 0.8 (fast) | 0.35 (fastest)           | **~1.5-2.0 (medium)**      |
| **Accuracy (WER)**           | 16.12%                | 13.89%     | 11.70% (best)            | **~12-15% (est.)**         |
| **Ease of Use**              | ✅                    | ✅         | ⚠️                       | **⚠️**                     |
| **Documentation**            | ✅                    | ⚠️         | ✅                       | **✅✅**                   |

### 🎯 Khi nào dùng model nào?

#### **Production / Real-world Application**

→ **SenseVoice + SpeechBrain**

- Nhanh nhất + accurate nhất
- Đã được evaluate thoroughly trên dataset lớn
- Ổn định và đáng tin cậy

#### **Research / Development / Experimentation**

→ **NeMo TitaNet + Whisper**

- Flexibility cao, dễ customize mọi bước
- Có speaker memory cho tracking qua nhiều files
- Auto-detect số speakers
- Ideal cho prototype và testing các approaches mới

#### **Quick Prototype / Simple Task**

→ **SenseVoice**

- Đơn giản nhất (chỉ ASR, không diarization)
- Setup nhanh
- Phù hợp khi không cần phân biệt người nói

#### **Learning / Education**

→ **Whisper + SpeechBrain** hoặc **NeMo TitaNet + Whisper**

- Well-documented với comments chi tiết
- Dễ hiểu cách hoạt động từng component
- NeMo model có docs đầy đủ về approach và parameters

---

## ✅ Checklist cho người mới bắt đầu

- [ ] Clone repo và cd vào `realtime/`
- [ ] Tạo virtual environment: `python -m venv venv`
- [ ] Activate venv: `venv\Scripts\activate` (Windows)
- [ ] Cài đặt dependencies: `pip install -r requirements.txt`
- [ ] Chọn 1 model để test (recommend: SenseVoice + SpeechBrain hoặc NeMo)
- [ ] Cài đặt dependencies cho model đó
- [ ] Chuẩn bị file audio test (16kHz, WAV format)
- [ ] Chạy file Python của model
- [ ] Kiểm tra output trên console và JSON file
- [ ] Đọc docs để hiểu parameters và customize

**Hỗ trợ thêm?** Đọc phần [Troubleshooting](#-troubleshooting) hoặc issues trên GitHub repo.

---
