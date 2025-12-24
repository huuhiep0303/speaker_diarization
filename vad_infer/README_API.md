# NeMo VAD API on Modal Cloud

API inference cho NeMo VAD models trên Modal cloud với GPU acceleration.

## 📦 Files trong folder này:

1. **vad_api.py** - Modal API server code
2. **vad_client.py** - Python client để call API
3. **upload_models.py** - Script upload models lên Modal volume
4. **requirements.txt** - Dependencies cho client
5. **MODAL_GUIDE.md** - Hướng dẫn chi tiết setup và deploy
6. **README_API.md** - File này

---

## 🚀 Quick Start (5 phút)

### Bước 1: Install và Authenticate

```bash
# Install dependencies
pip install -r requirements.txt

# Authenticate with Modal
modal token new
```

### Bước 2: Upload Models

```bash
# Upload your fine-tuned models
python upload_models.py --model best_vad_nemo.nemo --model vad_model.ckpt
```

### Bước 3: Deploy API

```bash
# Deploy to Modal cloud
modal deploy vad_api.py
```

**Lưu lại API URL từ output!**

### Bước 4: Test API

```bash
# Test với audio file
python vad_client.py --api-url <YOUR_API_URL> --audio audio.wav
```

**Done!** 🎉

---

## 📚 Hướng dẫn chi tiết

Xem file **[MODAL_GUIDE.md](MODAL_GUIDE.md)** để biết:

- Setup Modal account
- Upload models chi tiết
- API deployment options
- Authentication setup
- Monitoring và troubleshooting
- Pricing information
- API reference

---

## 💻 Cách sử dụng

### Health check

```bash
python vad_client.py --api-url <YOUR_API_URL> --health
```

### List models available

```bash
python vad_client.py --api-url <YOUR_API_URL> --list-models
```

### Single file inference

```bash
python vad_client.py \
    --api-url <YOUR_API_URL> \
    --audio audio.wav
```

### Batch inference

```bash
python vad_client.py \
    --api-url <YOUR_API_URL> \
    --audio_dir dataset/test \
    --output_dir results
```

### Custom parameters

```bash
python vad_client.py \
    --api-url <YOUR_API_URL> \
    --audio audio.wav \
    --model vad_model.ckpt \
    --threshold 0.6 \
    --output output.rttm \
    --save-json
```

---

## 🔧 API Endpoints

### POST /infer_api

Run VAD inference trên audio file

**Request:**

```json
{
  "audio": "base64_encoded_audio",
  "model": "best_vad_nemo.nemo",
  "threshold": 0.5
}
```

**Response:**

```json
{
  "success": true,
  "num_segments": 12,
  "total_speech_duration": 6.3,
  "rttm": "SPEAKER audio 1 0.500 1.250 ..."
}
```

### GET /health

Check API status

### GET /list_models

List available models in volume

---

## 📊 Features

✅ **GPU Inference** - T4 GPU on Modal (nhanh ~100x realtime)  
✅ **Auto-scaling** - Scale to zero when idle  
✅ **Pay-per-use** - Chỉ pay khi có requests  
✅ **Multiple models** - Support cả .nemo và .ckpt  
✅ **Batch processing** - Process nhiều files  
✅ **JSON + RTTM output** - Flexible output formats

---

## 💰 Cost Estimate

**Free tier:** $30/month credits  
**T4 GPU:** ~$0.48/hour (~$0.0004/request với 10s audio)

**Example monthly costs:**

- 1,000 requests: ~$0.40
- 10,000 requests: ~$4.00
- 100,000 requests: ~$40.00

_(Chỉ pay khi có requests, không pay khi idle)_

---

## 🔒 Security

Để add authentication:

1. Tạo Modal secret:

```bash
modal secret create vad-api-key API_KEY=your_secret_key
```

2. Update `vad_api.py` để check API key

3. Client gửi API key trong request:

```python
payload = {
    "audio": audio_b64,
    "api_key": "your_secret_key"
}
```

---

## 🐛 Common Issues

### Model not found

```bash
# Re-upload model
python upload_models.py --model best_vad_nemo.nemo
```

### API timeout

```bash
# Increase timeout in client
# or split audio into smaller chunks
```

### High latency on first request

- **Normal:** Cold start (~10-30s first time)
- Subsequent requests: ~3-5s for 10s audio

---

## 📞 Support

- **Modal Docs:** https://modal.com/docs
- **Modal Discord:** https://discord.gg/modal
- **Issues:** Check logs với `modal app logs nemo-vad-api`

---

## 🎯 Next Steps

1. ✅ Deploy API lên Modal
2. ✅ Share API URL với team
3. ✅ Monitor usage trên Modal dashboard
4. 📈 Scale as needed (auto-handled by Modal)

---

**Chúc bạn deploy thành công! 🚀**

Nếu có vấn đề, xem **MODAL_GUIDE.md** hoặc hỏi lại tôi.
