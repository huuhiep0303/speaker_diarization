# Hướng dẫn Deploy NeMo VAD API trên Modal Cloud

## 📋 Tổng quan

Hệ thống API cho phép người dùng gọi inference NeMo VAD models từ xa thông qua HTTP REST API. Models được host trên Modal cloud với GPU acceleration.

**Ưu điểm:**

- ✅ GPU inference nhanh (T4 GPU)
- ✅ Pay-per-use pricing (chỉ trả khi có request)
- ✅ Auto-scaling
- ✅ Không cần maintain infrastructure
- ✅ HTTPS endpoints tự động

---

## 🚀 Bước 1: Setup Modal Account

### 1.1. Tạo tài khoản Modal

1. Truy cập: https://modal.com
2. Sign up với GitHub hoặc email
3. Verify email

### 1.2. Install Modal CLI

```bash
pip install modal
```

### 1.3. Authenticate

```bash
modal token new
```

Lệnh này sẽ:

- Mở browser để login
- Tạo token và lưu vào `~/.modal.toml`

### 1.4. Verify setup

```bash
modal profile current
```

---

## 📦 Bước 2: Upload Models

Bạn có 2 cách upload models:

### Option A: Sử dụng script Python

```bash
# Upload single model
python upload_models.py --model best_vad_nemo.nemo

# Upload multiple models
python upload_models.py --model best_vad_nemo.nemo --model vad_model.ckpt

# Upload all models in directory
python upload_models.py --dir pretrained_models/nemo_vad
```

### Option B: Sử dụng Modal CLI trực tiếp

```bash
# Create volume if not exists
modal volume create nemo-vad-models

# Upload models
modal volume put nemo-vad-models best_vad_nemo.nemo best_vad_nemo.nemo
modal volume put nemo-vad-models vad_model.ckpt vad_model.ckpt

# List models in volume
modal volume ls nemo-vad-models
```

### Verify uploads

```bash
python upload_models.py
# Sẽ list tất cả models trong volume
```

---

## 🌐 Bước 3: Deploy API

### 3.1. Test API locally (recommended)

```bash
modal serve vad_api.py
```

**Output:**

```
✓ Created web function infer_api => https://username--nemo-vad-api-infer-api-dev.modal.run
✓ Created web function health => https://username--nemo-vad-api-health-dev.modal.run
✓ Created web function list_models => https://username--nemo-vad-api-list-models-dev.modal.run
```

**Test endpoints:**

```bash
# Health check
curl https://username--nemo-vad-api-health-dev.modal.run

# List models
curl https://username--nemo-vad-api-list-models-dev.modal.run
```

Khi test xong, nhấn `Ctrl+C` để stop.

### 3.2. Deploy to production

```bash
modal deploy vad_api.py
```

**Output:**

```
✓ Created web function infer_api => https://username--nemo-vad-api-infer-api.modal.run
✓ Created web function health => https://username--nemo-vad-api-health.modal.run
✓ Created web function list_models => https://username--nemo-vad-api-list-models.modal.run
```

**Lưu lại API URL!** Bạn sẽ cần nó để gọi API.

---

## 🔧 Bước 4: Test API

### 4.1. Health check

```bash
python vad_client.py \
    --api-url https://username--nemo-vad-api-infer-api.modal.run \
    --health
```

**Output:**

```
✅ API is healthy
   Service: NeMo VAD API
   Version: 1.0.0
   Status: healthy
```

### 4.2. List models

```bash
python vad_client.py \
    --api-url https://username--nemo-vad-api-infer-api.modal.run \
    --list-models
```

**Output:**

```
📦 Available models (2):
   1. best_vad_nemo.nemo (480.00 MB) - nemo
   2. vad_model.ckpt (1.20 MB) - ckpt
```

### 4.3. Single file inference

```bash
python vad_client.py \
    --api-url https://username--nemo-vad-api-infer-api.modal.run \
    --audio audio.wav
```

**Output:**

```
🎵 Processing: audio.wav
   Size: 1024.50 KB
   🌐 Calling API...
   ✅ Success!
   Audio duration: 10.50s
   Speech segments: 12
   Total speech: 6.30s (60.0%)
   💾 Saved RTTM: results/audio.rttm
```

### 4.4. Batch inference

```bash
python vad_client.py \
    --api-url https://username--nemo-vad-api-infer-api.modal.run \
    --audio_dir dataset/test \
    --output_dir results
```

### 4.5. Custom parameters

```bash
python vad_client.py \
    --api-url https://username--nemo-vad-api-infer-api.modal.run \
    --audio audio.wav \
    --model vad_model.ckpt \
    --threshold 0.6 \
    --output custom_output.rttm \
    --save-json
```

---

## 📊 Bước 5: Monitor và Manage

### 5.1. View logs

```bash
modal app logs nemo-vad-api
```

### 5.2. View metrics

Truy cập Modal Dashboard:

- https://modal.com/apps

Xem:

- Number of requests
- Execution time
- GPU utilization
- Costs

### 5.3. Stop API

```bash
modal app stop nemo-vad-api
```

**Note:** Với deployment, API vẫn tồn tại nhưng không tốn cost khi không có requests.

---

## 💰 Pricing

### Modal Pricing (tính đến Dec 2024)

**Free tier:**

- $30/month credits
- Enough cho ~1000 requests (mỗi request ~3s GPU time)

**T4 GPU pricing:**

- $0.000133/second (~$0.48/hour)
- Example: 10s audio inference = ~3s GPU time = $0.0004

**Ví dụ monthly cost:**

- 1,000 requests/month: ~$0.40
- 10,000 requests/month: ~$4.00
- 100,000 requests/month: ~$40.00

**Storage:**

- Volume storage: $0.05/GB/month
- Models (~1GB): $0.05/month

---

## 🔒 Security Best Practices

### 1. Add authentication (recommended)

Thêm vào `vad_api.py`:

```python
@app.function(image=image)
@modal.web_endpoint(method="POST", docs=True)
def infer_api(request: dict):
    # Check API key
    api_key = request.get("api_key")
    if api_key != "YOUR_SECRET_KEY":
        return {"error": "Invalid API key"}

    # ... rest of code
```

### 2. Rate limiting

Dùng Modal secrets để store API keys:

```bash
modal secret create vad-api-key API_KEY=your_secret_key
```

Update code:

```python
import os

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("vad-api-key")]
)
@modal.web_endpoint(method="POST")
def infer_api(request: dict):
    api_key = request.get("api_key")
    if api_key != os.environ["API_KEY"]:
        return {"error": "Unauthorized"}
    # ...
```

### 3. Input validation

Đã implement trong code:

- Max file size check
- Audio format validation
- Parameter bounds checking

---

## 🐛 Troubleshooting

### Issue: Model not found

**Lỗi:**

```
{"error": "Model not found: best_vad_nemo.nemo"}
```

**Giải pháp:**

```bash
# Check models in volume
modal volume ls nemo-vad-models

# Re-upload if needed
python upload_models.py --model best_vad_nemo.nemo
```

### Issue: API timeout

**Lỗi:**

```
requests.exceptions.Timeout: Request timed out
```

**Giải pháp:**

- Tăng timeout trong client: `timeout=600`
- Audio file quá lớn → split thành chunks nhỏ hơn
- Check Modal dashboard for cold start times

### Issue: CUDA out of memory

**Giải pháp:**

Sửa `vad_api.py`, đổi GPU:

```python
@app.function(
    gpu="A10G",  # Thay vì T4 (more memory)
    ...
)
```

### Issue: High latency on first request

**Nguyên nhân:** Cold start (Modal cần download image và khởi động container)

**Giải pháp:**

- Keep-warm function (advanced):

```python
@app.function(
    image=image,
    gpu="T4",
    schedule=modal.Cron("*/5 * * * *")  # Every 5 minutes
)
def keep_warm():
    """Keep container warm"""
    pass
```

---

## 📚 API Reference

### POST /infer_api

**Request:**

```json
{
  "audio": "base64_encoded_audio_bytes",
  "model": "best_vad_nemo.nemo",
  "threshold": 0.5,
  "min_speech_duration": 0.2,
  "min_silence_duration": 0.3,
  "filename": "audio"
}
```

**Response:**

```json
{
  "success": true,
  "audio_duration": 10.5,
  "num_segments": 12,
  "total_speech_duration": 6.3,
  "speech_ratio": 0.6,
  "segments": [{ "start": 0.5, "end": 1.75, "duration": 1.25 }],
  "rttm": "SPEAKER audio 1 0.500 1.250 <NA> <NA> speech <NA> <NA>\n...",
  "model": "best_vad_nemo.nemo",
  "threshold": 0.5,
  "device": "cuda"
}
```

### GET /health

**Response:**

```json
{
  "status": "healthy",
  "service": "NeMo VAD API",
  "version": "1.0.0",
  "timestamp": "2025-12-22T10:30:00"
}
```

### GET /list_models

**Response:**

```json
{
  "models": [
    {
      "name": "best_vad_nemo.nemo",
      "size_mb": 480.0,
      "format": "nemo"
    }
  ],
  "count": 1
}
```

---

## 🔄 Update Models

### Update existing model:

```bash
# Upload new version
python upload_models.py --model best_vad_nemo.nemo

# Redeploy API (để load model mới)
modal deploy vad_api.py
```

### Add new model:

```bash
# Upload new model
python upload_models.py --model new_model.nemo

# No need to redeploy, model sẽ available ngay
```

---

## 📞 Support

Nếu gặp vấn đề:

1. **Check Modal Dashboard:** https://modal.com/apps
2. **View logs:** `modal app logs nemo-vad-api`
3. **Modal docs:** https://modal.com/docs
4. **Modal Discord:** https://discord.gg/modal

---

## ✅ Quick Start Checklist

- [ ] Install Modal: `pip install modal`
- [ ] Authenticate: `modal token new`
- [ ] Upload models: `python upload_models.py --model best_vad_nemo.nemo`
- [ ] Test locally: `modal serve vad_api.py`
- [ ] Deploy to prod: `modal deploy vad_api.py`
- [ ] Test API: `python vad_client.py --api-url <URL> --audio audio.wav`
- [ ] Monitor dashboard: https://modal.com/apps

---

## 🎉 Hoàn tất!

API của bạn đã ready để sử dụng! Chia sẻ API URL với team members để họ có thể inference VAD models.

**Example usage for team:**

```bash
python vad_client.py \
    --api-url https://your-username--nemo-vad-api-infer-api.modal.run \
    --audio my_audio.wav
```

---

**Happy API-ing! 🚀**
