# Kết quả đánh giá ASR (Automatic Speech Recognition)

Đánh giá 3 models chuyển đổi giọng nói thành văn bản trên **JVS Dataset** (398 audio samples).

---

## 📊 Tóm tắt kết quả

| Model                      | WER (%)    | CER (%)   | RTF  | Xếp hạng          |
| -------------------------- | ---------- | --------- | ---- | ----------------- |
| **SenseVoice+SpeechBrain** | **11.70%** | **8.32%** | 1.37 | 🥇 **Tốt nhất**   |
| **SenseVoice**             | 13.89%     | 10.08%    | 1.47 | 🥈 Khá tốt        |
| **Whisper-small**          | 16.12%     | 12.58%    | 2.75 | 🥉 Chấp nhận được |

---

## 🎯 Giải thích Metrics

### 1. WER (Word Error Rate) - Tỷ lệ lỗi từ

- **Định nghĩa**: Tỷ lệ từ bị nhận dạng sai so với tổng số từ
- **Công thức**: `WER = (Substitutions + Deletions + Insertions) / Total Words`
- **Càng thấp càng tốt**: 0% = hoàn hảo, 100% = sai hoàn toàn

**Ví dụ**:

```
Ground Truth: "これは テスト です" (4 từ)
Prediction:   "これは デスト です" (1 từ sai)
WER = 1/4 = 25%
```

### 2. CER (Character Error Rate) - Tỷ lệ lỗi ký tự

- **Định nghĩa**: Tỷ lệ ký tự bị nhận dạng sai so với tổng số ký tự
- **Phù hợp cho tiếng Nhật**: Vì tiếng Nhật không có khoảng trắng rõ ràng giữa các từ
- **Càng thấp càng tốt**

**Ví dụ**:

```
Ground Truth: "これはテストです" (8 ký tự)
Prediction:   "これはデストです" (1 ký tự sai: テ → デ)
CER = 1/8 = 12.5%
```

### 3. RTF (Real-Time Factor) - Hệ số thời gian thực

- **Định nghĩa**: Tỷ lệ giữa thời gian xử lý và độ dài audio
- **Công thức**: `RTF = Processing Time / Audio Duration`
- **Càng thấp càng tốt**:
  - RTF < 1.0: **Nhanh hơn real-time** ✅ (xử lý nhanh hơn độ dài audio)
  - RTF = 1.0: **Bằng real-time** (xử lý đúng bằng độ dài audio)
  - RTF > 1.0: **Chậm hơn real-time** ⚠️ (cần nhiều thời gian hơn để xử lý)

**Ví dụ**:

```
Audio: 10 giây
Processing: 13.7 giây
RTF = 13.7/10 = 1.37
→ Chậm hơn real-time 37%
```

---

## 📈 Phân tích chi tiết

### 🥇 **SenseVoice+SpeechBrain** (Best Overall)

**Điểm mạnh**:

- ✅ **WER thấp nhất: 11.70%** - Nhận dạng chính xác nhất
- ✅ **CER thấp nhất: 8.32%** - Ít lỗi ký tự nhất
- ✅ **RTF tốt: 1.37** - Tốc độ chấp nhận được
- ✅ **Kết hợp tốt nhất**: SenseVoice (ASR) + SpeechBrain (enhancement)

**Khi nào dùng**:

- ✅ Khi cần độ chính xác cao nhất
- ✅ Transcription cho mục đích chuyên nghiệp
- ✅ Ứng dụng yêu cầu chất lượng văn bản cao
- ✅ Có thể chấp nhận xử lý chậm hơn một chút để đổi lấy độ chính xác

**Performance**:

```
Số samples đánh giá: 398
WER trung bình:      11.70%
CER trung bình:      8.32%
RTF trung bình:      1.37
RTF median:          0.95 (nhanh hơn real-time!)
RTF min/max:         0.46 - 10.75
```

---

### 🥈 **SenseVoice** (Good Balance)

**Điểm mạnh**:

- ✅ **WER tốt: 13.89%** - Độ chính xác cao
- ✅ **CER tốt: 10.08%** - Lỗi ký tự ít
- ✅ **RTF chấp nhận được: 1.47** - Tốc độ hợp lý
- ✅ **Cân bằng tốt** giữa chất lượng và tốc độ

**So sánh với SenseVoice+SpeechBrain**:

- WER cao hơn 2.19% (13.89% vs 11.70%)
- CER cao hơn 1.76% (10.08% vs 8.32%)
- RTF chậm hơn chút ít: 1.47 vs 1.37

**Khi nào dùng**:

- ✅ Cần độ chính xác cao nhưng không cần "perfect"
- ✅ Ứng dụng real-time với yêu cầu chất lượng tốt
- ✅ Transcription cho mục đích thông thường
- ⚠️ Không cần SpeechBrain enhancement

**Performance**:

```
Số samples đánh giá: 398
WER trung bình:      13.89%
CER trung bình:      10.08%
RTF trung bình:      1.47
RTF median:          1.07
RTF min/max:         0.31 - 18.41
```

---

### 🥉 **Whisper-small** (Acceptable)

**Điểm mạnh**:

- ✅ **Model phổ biến**: Được sử dụng rộng rãi
- ✅ **Đa ngôn ngữ**: Support nhiều ngôn ngữ
- ✅ **Cộng đồng lớn**: Nhiều tài liệu, hỗ trợ

**Điểm yếu**:

- ⚠️ **WER cao nhất: 16.12%** - Kém chính xác nhất trong 3 model
- ⚠️ **CER cao nhất: 12.58%** - Nhiều lỗi ký tự nhất
- ⚠️ **RTF chậm nhất: 2.75** - Chậm hơn real-time gần 3 lần!
- ⚠️ **Không tối ưu cho tiếng Nhật**

**So sánh với SenseVoice+SpeechBrain**:

- WER cao hơn 4.42% (16.12% vs 11.70%)
- CER cao hơn 4.26% (12.58% vs 8.32%)
- RTF chậm hơn gấp đôi: 2.75 vs 1.37

**Khi nào dùng**:

- ✅ Cần model đa ngôn ngữ (không chỉ tiếng Nhật)
- ✅ Đã có infrastructure sử dụng Whisper
- ✅ Không yêu cầu độ chính xác cao
- ⚠️ Có thể chấp nhận tốc độ chậm

**Performance**:

```
Số samples đánh giá: 398
WER trung bình:      16.12%
CER trung bình:      12.58%
RTF trung bình:      2.75
RTF median:          2.50
RTF min/max:         0.97 - 8.11
```

---

## 🎓 So sánh trực quan

### WER - Word Error Rate (thấp hơn = tốt hơn)

```
SenseVoice+SpeechBrain: ████████████ 11.70%  ⭐⭐⭐⭐⭐
SenseVoice:             ██████████████ 13.89%  ⭐⭐⭐⭐
Whisper-small:          ████████████████ 16.12%  ⭐⭐⭐
```

### CER - Character Error Rate (thấp hơn = tốt hơn)

```
SenseVoice+SpeechBrain: ████████ 8.32%   ⭐⭐⭐⭐⭐
SenseVoice:             ██████████ 10.08%   ⭐⭐⭐⭐
Whisper-small:          ████████████ 12.58%   ⭐⭐⭐
```

### RTF - Real-Time Factor (thấp hơn = nhanh hơn)

```
SenseVoice+SpeechBrain: ██████████████ 1.37x  ⭐⭐⭐⭐
SenseVoice:             ███████████████ 1.47x  ⭐⭐⭐⭐
Whisper-small:          ███████████████████████████ 2.75x  ⭐⭐
```

---

## 💡 Khuyến nghị sử dụng

### Scenario 1: Transcription chuyên nghiệp (Meeting, Interview)

**→ Dùng SenseVoice+SpeechBrain** 🥇

- Yêu cầu độ chính xác cao nhất
- Có thể chấp nhận xử lý chậm hơn một chút
- Văn bản cần chính xác cho mục đích lưu trữ/phân tích

### Scenario 2: Real-time Subtitle (Live stream, Video call)

**→ Dùng SenseVoice** 🥈

- Cân bằng tốt giữa chất lượng và tốc độ
- RTF gần real-time (median 1.07x)
- Độ chính xác chấp nhận được cho subtitle

### Scenario 3: Multi-language Application

**→ Dùng Whisper-small** 🥉

- Cần support nhiều ngôn ngữ
- Có infrastructure sẵn với Whisper
- Chấp nhận độ chính xác thấp hơn

### Scenario 4: High-accuracy Japanese ASR

**→ Dùng SenseVoice+SpeechBrain** 🥇

- Tối ưu hóa cho tiếng Nhật
- WER thấp nhất: 11.70%
- CER thấp nhất: 8.32%

---

## 📝 Kết luận

### Rankings:

**1️⃣ Độ chính xác (Accuracy)**:

1. SenseVoice+SpeechBrain (11.70% WER) 🥇
2. SenseVoice (13.89% WER) 🥈
3. Whisper-small (16.12% WER) 🥉

**2️⃣ Tốc độ (Speed)**:

1. SenseVoice+SpeechBrain (1.37 RTF) 🥇
2. SenseVoice (1.47 RTF) 🥈
3. Whisper-small (2.75 RTF) 🥉

**3️⃣ Tổng thể (Overall)**:

1. **SenseVoice+SpeechBrain** - Best choice cho tiếng Nhật 🥇
2. **SenseVoice** - Good balance 🥈
3. **Whisper-small** - Acceptable cho multi-language 🥉

