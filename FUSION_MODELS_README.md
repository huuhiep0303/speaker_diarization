# Fusion Speaker Diarization Models

## Tổng quan

Project này triển khai các **Fusion Diarization Models** - kết hợp các ASR models với các speaker embedding models khác nhau để tạo ra hệ thống speaker diarization hoàn chỉnh.

## Kiến trúc Fusion

Mỗi fusion model bao gồm 2 components chính:

1. **ASR Component**: Chuyển đổi giọng nói thành văn bản + phân đoạn audio (VAD)
2. **Speaker Embedding Component**: Trích xuất speaker embeddings để nhận diện người nói

## Các Fusion Models

### 1. **Whisper + SpeechBrain** ✅ (Baseline)

- **File**: `realtime_diarization_improved.py`
- **ASR**: faster-whisper (multilingual, 99 languages)
- **Speaker**: SpeechBrain ECAPA-TDNN (EER: 15.57%, AUC: 0.9353)
- **Ưu điểm**:
  - Đã được test kỹ, stable
  - Tốt cho English và European languages
  - Whisper có VAD tốt
- **Nhược điểm**: SpeechBrain không tốt bằng NeMo

### 2. **SenseVoice + SpeechBrain** ✅

- **File**: `senvoi_spebrai_fixed.py`
- **ASR**: FunAudioLLM/SenseVoiceSmall (Chinese, English, Japanese, Korean, Cantonese)
- **Speaker**: SpeechBrain ECAPA-TDNN (EER: 15.57%, AUC: 0.9353)
- **Ưu điểm**:
  - Tốt cho Asian languages
  - Có emotion detection (Happy, Sad, Angry, Neutral)
  - Có event detection (Speech, Music, Applause)
- **Nhược điểm**: SpeechBrain không tốt bằng NeMo

### 3. **Whisper + PyAnnote** ✨ NEW

- **File**: `whisper_pyannote.py`
- **ASR**: faster-whisper (multilingual, 99 languages)
- **Speaker**: PyAnnote WeSpeaker-ResNet34 (đang đánh giá)
- **Ưu điểm**:
  - Whisper VAD tốt
  - PyAnnote là state-of-the-art cho diarization
- **Nhược điểm**: Cần đánh giá để biết performance

### 4. **SenseVoice + PyAnnote** ✨ NEW

- **File**: `sensevoice_pyannote.py`
- **ASR**: FunAudioLLM/SenseVoiceSmall (Asian languages + emotion)
- **Speaker**: PyAnnote WeSpeaker-ResNet34 (đang đánh giá)
- **Ưu điểm**:
  - Tốt cho Asian languages
  - Có emotion + event detection
  - PyAnnote speaker embedding
- **Nhược điểm**: Cần đánh giá để biết performance

### 5. **Whisper + NeMo** ✨ NEW ⭐ RECOMMENDED

- **File**: `whisper_nemo.py`
- **ASR**: faster-whisper (multilingual, 99 languages)
- **Speaker**: NeMo TitaNet Large (EER: 14.89%, AUC: 0.9403) ← **BEST**
- **Ưu điểm**:
  - Whisper VAD tốt
  - **NeMo có speaker embedding tốt nhất**
  - Suitable cho production
- **Nhược điểm**: Cần GPU để tốc độ tối ưu

### 6. **SenseVoice + NeMo** ✨ NEW ⭐ RECOMMENDED

- **File**: `sensevoice_nemo.py`
- **ASR**: FunAudioLLM/SenseVoiceSmall (Asian languages + emotion)
- **Speaker**: NeMo TitaNet Large (EER: 14.89%, AUC: 0.9403) ← **BEST**
- **Ưu điểm**:
  - **Tốt nhất cho Asian languages**
  - **NeMo speaker embedding tốt nhất**
  - Có emotion + event detection
  - Recommended cho tiếng Việt, tiếng Trung, tiếng Nhật
- **Nhược điểm**: Cần GPU để tốc độ tối ưu

## So sánh Speaker Embeddings

| Model                       | EER (↓)    | AUC (↑)    | Best F1 (↑) | Note          |
| --------------------------- | ---------- | ---------- | ----------- | ------------- |
| **NeMo TitaNet Large**      | **14.89%** | **0.9403** | **87.02%**  | 🏆 **BEST**   |
| SpeechBrain ECAPA-TDNN      | 15.57%     | 0.9353     | 86.38%      | Good baseline |
| PyAnnote WeSpeaker-ResNet34 | TBD        | TBD        | TBD         | Đang đánh giá |

_Nguồn: `evaluation/eval_results/result.log`_

## Cách sử dụng

### 1. Xử lý single audio file

```bash
# Whisper + NeMo (recommended for English)
python whisper_nemo.py --audio_file audio.wav

# SenseVoice + NeMo (recommended for Asian languages)
python sensevoice_nemo.py --audio_file audio.wav

# Whisper + PyAnnote
python whisper_pyannote.py --audio_file audio.wav --output result.json

# SenseVoice + SpeechBrain (baseline, có emotion)
python senvoi_spebrai_fixed.py --audio_file audio.wav
```

### 2. Đánh giá speaker embeddings

```bash
cd evaluation
python eval_diarization.py --dataset /path/to/jvs_ver1 --max_speakers 10
```

### 3. Xem tổng quan fusion models

```bash
cd evaluation
python eval_fusion_models.py --all
```

## Output Format

Tất cả các fusion models đều output JSON với format giống nhau:

```json
{
  "audio_file": "audio.wav",
  "timestamp": "2025-12-15T10:30:00",
  "model": {
    "asr": "faster-whisper-small",
    "speaker": "nvidia/speakerverification_en_titanet_large"
  },
  "device": "cuda",
  "language": "en",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "duration": 2.5,
      "text": "Hello, how are you?",
      "speaker": "spk_01"
    },
    {
      "start": 2.8,
      "end": 5.0,
      "duration": 2.2,
      "text": "I'm fine, thank you.",
      "speaker": "spk_02"
    }
  ],
  "speaker_stats": {
    "spk_01": {
      "count": 3,
      "duration": 8.5,
      "text_length": 150
    },
    "spk_02": {
      "count": 2,
      "duration": 5.2,
      "text_length": 80
    }
  },
  "total_segments": 5,
  "total_speakers": 2
}
```

## Khuyến nghị sử dụng

### Theo ngôn ngữ:

- **English, European languages**: `whisper_nemo.py`
- **Chinese, Japanese, Korean, Vietnamese**: `sensevoice_nemo.py`
- **Multilingual (mixed)**: `whisper_nemo.py`

### Theo use case:

- **Best overall performance**: `whisper_nemo.py` hoặc `sensevoice_nemo.py`
- **Need emotion detection**: `sensevoice_nemo.py` hoặc `sensevoice_pyannote.py`
- **Most stable/tested**: `realtime_diarization_improved.py` (Whisper + SpeechBrain)
- **Experimental**: `*_pyannote.py` models (đang test)

### Theo resource:

- **GPU available**: Bất kỳ model nào, ưu tiên NeMo-based
- **CPU only**: SenseVoice-based models (SenseVoice chạy tốt trên CPU)

## Installation

```bash
# Core dependencies
pip install torch torchaudio soundfile

# For Whisper models
pip install faster-whisper

# For SenseVoice models
pip install funasr

# For SpeechBrain speaker embedding
pip install speechbrain

# For NeMo speaker embedding
pip install nemo_toolkit[asr]

# For PyAnnote speaker embedding
pip install pyannote.audio
```

## Performance Metrics

### Speaker Diarization (từ eval_diarization.py):

- **NeMo**: EER = 14.89%, AUC = 0.9403 ⭐
- **SpeechBrain**: EER = 15.57%, AUC = 0.9353
- **PyAnnote**: Đang đánh giá...

### ASR Performance:

- **Whisper**: WER varies by language (3-10% for English)
- **SenseVoice**: Optimized for Asian languages

_Lưu ý: ASR performance phụ thuộc vào language, domain, và audio quality_

## Files Structure

```
realtime/
├── realtime_diarization_improved.py  # Whisper + SpeechBrain (baseline)
├── senvoi_spebrai_fixed.py          # SenseVoice + SpeechBrain
├── whisper_pyannote.py               # Whisper + PyAnnote (new)
├── sensevoice_pyannote.py            # SenseVoice + PyAnnote (new)
├── whisper_nemo.py                   # Whisper + NeMo (new, recommended)
├── sensevoice_nemo.py                # SenseVoice + NeMo (new, recommended)
├── main_nemo.py                      # NeMo standalone diarization
├── main_pyannote.py                  # PyAnnote standalone diarization
└── evaluation/
    ├── eval_diarization.py           # Evaluate speaker embeddings
    ├── eval_fusion_models.py         # Compare fusion models
    └── eval_results/
        ├── result.log                # Speaker embedding results
        └── fusion_models_summary.txt # Comprehensive report
```

## Contributing

Để thêm fusion model mới:

1. Tạo file mới (e.g., `new_asr_new_speaker.py`)
2. Implement class với 2 components: ASR + Speaker
3. Sử dụng `SpeakerManager` class để quản lý speakers
4. Follow output format giống các models khác
5. Update `eval_fusion_models.py` để thêm vào comparison

## References

- **Whisper**: https://github.com/openai/whisper
- **SenseVoice**: https://github.com/FunAudioLLM/SenseVoice
- **SpeechBrain**: https://speechbrain.github.io/
- **NeMo**: https://github.com/NVIDIA/NeMo
- **PyAnnote**: https://github.com/pyannote/pyannote-audio

## License

See LICENSE file for details.

---

**Last Updated**: December 15, 2025

**Version**: 1.0

**Status**:

- ✅ Whisper + SpeechBrain (Production ready)
- ✅ SenseVoice + SpeechBrain (Production ready)
- ✨ Whisper + NeMo (New, recommended)
- ✨ SenseVoice + NeMo (New, recommended)
- 🧪 Whisper + PyAnnote (Experimental)
- 🧪 SenseVoice + PyAnnote (Experimental)
