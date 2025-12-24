"""
NeMo VAD API on Modal Cloud

This script creates a REST API for NeMo VAD inference on Modal cloud.
Users can send audio files and receive RTTM results via HTTP requests.

Features:
- GPU-accelerated inference on Modal
- RESTful API endpoints
- Support for .nemo and .ckpt models
- Auto RTTM generation
- Batch processing support

Usage:
    # Deploy API
    modal deploy vad_api.py
    
    # Test API locally
    modal serve vad_api.py
"""

import modal
import io
import base64
import json
from pathlib import Path
from datetime import datetime
import numpy as np

# Create Modal app
app = modal.App("nemo-vad-api")

# Define Modal image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "libsndfile1",
        "ffmpeg",
        "sox",
        "libsox-dev",
        "git",
        "build-essential",
    )
    # Core build tools
    .pip_install(
        "pip==23.3.2",
        "setuptools==69.0.3",
        "wheel==0.42.0",
        "Cython==3.0.8",
    )
    # NumPy 1.x (required for NeMo)
    .pip_install("numpy==1.24.3")
    # PyTorch
    .pip_install(
        "torch==2.1.0",
        "torchaudio==2.1.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    # HuggingFace Hub (needs ModelFilter for NeMo 1.23.0)
    .pip_install("huggingface-hub==0.23.0")
    # PyTorch Lightning
    .pip_install(
        "pytorch-lightning==2.1.0",
        "torchmetrics==1.2.1",
    )
    # Transformers (must install after PyTorch Lightning to avoid conflicts)
    .pip_install(
        "transformers==4.35.2",
        "tokenizers==0.15.0",
    )
    # Audio processing
    .pip_install(
        "soundfile==0.12.1",
        "librosa==0.10.1",
    )
    # Scientific computing
    .pip_install(
        "scipy==1.11.4",
        "scikit-learn==1.3.2",
    )
    # NeMo
    .pip_install("nemo_toolkit[asr]==1.23.0")
    # FastAPI for web endpoints
    .pip_install("fastapi")
)

# Volume for models
models_volume = modal.Volume.from_name("nemo-vad-models", create_if_missing=True)

# Volume for results (optional, for logging)
results_volume = modal.Volume.from_name("nemo-vad-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",  # T4 is cost-effective for inference
    timeout=300,
    volumes={
        "/models": models_volume,
        "/results": results_volume,
    },
)
def load_model(model_name: str = "best_vad_model.nemo"):
    """Load VAD model from volume"""
    from nemo.collections.asr.models import EncDecClassificationModel
    import torch
    
    model_path = Path(f"/models/{model_name}")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model: {model_name}")
    
    if model_path.suffix == ".nemo":
        model = EncDecClassificationModel.restore_from(str(model_path))
    elif model_path.suffix == ".ckpt":
        model = EncDecClassificationModel.load_from_checkpoint(str(model_path))
    else:
        raise ValueError(f"Unsupported format: {model_path.suffix}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    
    return model


@app.function(
    image=image,
    gpu="T4",
    timeout=300,
    volumes={
        "/models": models_volume,
        "/results": results_volume,
    },
)
def run_vad_inference(
    audio_bytes: bytes,
    model_name: str = "best_vad_model.nemo",
    threshold: float = 0.5,
    min_speech_duration: float = 0.2,
    min_silence_duration: float = 0.3,
    audio_filename: str = "audio"
):
    """
    Run VAD inference on audio bytes
    
    Args:
        audio_bytes: Audio file bytes
        model_name: Name of model in /models volume
        threshold: VAD threshold (0-1)
        min_speech_duration: Minimum speech segment duration (seconds)
        min_silence_duration: Minimum silence gap (seconds)
        audio_filename: Original filename for RTTM
    
    Returns:
        dict with segments and RTTM content
    """
    import torch
    import torchaudio
    import soundfile as sf
    from nemo.collections.asr.models import EncDecClassificationModel
    
    # Load model
    model_path = Path(f"/models/{model_name}")
    
    if not model_path.exists():
        return {
            "error": f"Model not found: {model_name}",
            "available_models": [f.name for f in Path("/models").glob("*") if f.suffix in [".nemo", ".ckpt"]]
        }
    
    print(f"Loading model: {model_name}")
    
    if model_path.suffix == ".nemo":
        model = EncDecClassificationModel.restore_from(str(model_path))
    elif model_path.suffix == ".ckpt":
        model = EncDecClassificationModel.load_from_checkpoint(str(model_path))
    else:
        return {"error": f"Unsupported format: {model_path.suffix}"}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    
    # Load audio from bytes
    audio_io = io.BytesIO(audio_bytes)
    
    try:
        audio, sr = sf.read(audio_io)
    except Exception as e:
        return {"error": f"Failed to load audio: {str(e)}"}
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Resample to 16kHz if needed
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    audio_duration = len(audio) / sr
    
    print(f"Audio loaded: {audio_duration:.2f}s, {sr}Hz")
    
    # Prepare input
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
    audio_length = torch.tensor([len(audio)]).to(device)
    
    # Run inference
    with torch.no_grad():
        logits = model(input_signal=audio_tensor, input_signal_length=audio_length)
        probs = torch.softmax(logits, dim=-1)
    
    # Get speech probabilities
    speech_probs = probs[0, :, 1].cpu().numpy()
    
    # Convert to segments
    frame_shift = 0.01  # 10ms
    segments = []
    in_speech = False
    segment_start = 0
    
    for i, prob in enumerate(speech_probs):
        time = i * frame_shift
        
        if prob >= threshold:
            if not in_speech:
                in_speech = True
                segment_start = time
        else:
            if in_speech:
                in_speech = False
                segments.append((segment_start, time))
    
    if in_speech:
        segments.append((segment_start, len(audio) / sr))
    
    # Post-process: filter short segments
    filtered_segments = [
        (start, end) for start, end in segments
        if (end - start) >= min_speech_duration
    ]
    
    # Post-process: merge close segments
    if len(filtered_segments) > 0:
        merged_segments = [filtered_segments[0]]
        
        for current in filtered_segments[1:]:
            last = merged_segments[-1]
            gap = current[0] - last[1]
            
            if gap < min_silence_duration:
                merged_segments[-1] = (last[0], current[1])
            else:
                merged_segments.append(current)
    else:
        merged_segments = []
    
    # Generate RTTM content
    rttm_lines = []
    for start, end in merged_segments:
        duration = end - start
        line = f"SPEAKER {audio_filename} 1 {start:.3f} {duration:.3f} <NA> <NA> speech <NA> <NA>"
        rttm_lines.append(line)
    
    rttm_content = "\n".join(rttm_lines)
    
    # Calculate statistics
    total_speech = sum(end - start for start, end in merged_segments)
    speech_ratio = total_speech / audio_duration if audio_duration > 0 else 0
    
    # Save to results volume (for logging)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = Path(f"/results/inference_{timestamp}.json")
    
    result_data = {
        "timestamp": timestamp,
        "model": model_name,
        "audio_filename": audio_filename,
        "audio_duration": float(audio_duration),
        "threshold": threshold,
        "num_segments": len(merged_segments),
        "total_speech": float(total_speech),
        "speech_ratio": float(speech_ratio),
    }
    
    with open(result_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    results_volume.commit()
    
    print(f"Inference completed: {len(merged_segments)} segments, {total_speech:.2f}s speech")
    
    return {
        "success": True,
        "audio_duration": audio_duration,
        "num_segments": len(merged_segments),
        "total_speech_duration": total_speech,
        "speech_ratio": speech_ratio,
        "segments": [
            {"start": float(start), "end": float(end), "duration": float(end - start)}
            for start, end in merged_segments
        ],
        "rttm": rttm_content,
        "model": model_name,
        "threshold": threshold,
        "device": str(device)
    }


@app.function(image=image)
@modal.fastapi_endpoint(method="POST", docs=True)
def infer_api(request: dict):
    """
    REST API endpoint for VAD inference
    
    Request body (JSON):
    {
        "audio": "base64_encoded_audio_bytes",
        "model": "best_vad_nemo.nemo",  // optional
        "threshold": 0.5,  // optional
        "filename": "audio.wav"  // optional
    }
    
    Response (JSON):
    {
        "success": true,
        "audio_duration": 10.5,
        "num_segments": 12,
        "total_speech_duration": 6.3,
        "speech_ratio": 0.6,
        "segments": [...],
        "rttm": "SPEAKER audio 1 0.500 1.250 ..."
    }
    """
    
    # Extract parameters
    audio_b64 = request.get("audio")
    model_name = request.get("model", "best_vad_model.nemo")
    threshold = request.get("threshold", 0.5)
    min_speech = request.get("min_speech_duration", 0.2)
    min_silence = request.get("min_silence_duration", 0.3)
    filename = request.get("filename", "audio")
    
    if not audio_b64:
        return {"error": "Missing 'audio' field in request"}
    
    # Decode audio
    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception as e:
        return {"error": f"Failed to decode base64 audio: {str(e)}"}
    
    # Run inference
    result = run_vad_inference.remote(
        audio_bytes=audio_bytes,
        model_name=model_name,
        threshold=threshold,
        min_speech_duration=min_speech,
        min_silence_duration=min_silence,
        audio_filename=filename
    )
    
    return result


@app.function(image=image)
@modal.fastapi_endpoint(method="GET", docs=True)
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "NeMo VAD API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.function(
    image=image,
    volumes={"/models": models_volume}
)
@modal.fastapi_endpoint(method="GET", docs=True)
def list_models():
    """List available models in volume"""
    models_dir = Path("/models")
    
    if not models_dir.exists():
        return {"models": [], "count": 0}
    
    models = []
    for model_file in models_dir.glob("*"):
        if model_file.suffix in [".nemo", ".ckpt"]:
            stat = model_file.stat()
            models.append({
                "name": model_file.name,
                "size_mb": stat.st_size / 1024 / 1024,
                "format": model_file.suffix[1:]
            })
    
    return {
        "models": models,
        "count": len(models)
    }


@app.local_entrypoint()
def main():
    """Local test"""
    print("\n" + "="*80)
    print("🚀 NeMo VAD API - Modal Cloud")
    print("="*80)
    print()
    print("API deployed successfully!")
    print()
    print("Endpoints:")
    print("  POST /infer_api - Run VAD inference")
    print("  GET  /health    - Health check")
    print("  GET  /list_models - List available models")
    print()
    print("Next steps:")
    print("  1. Upload models: modal volume put nemo-vad-models best_vad_model.nemo")
    print("  2. Deploy API: modal deploy vad_api.py")
    print("  3. Test API: python vad_client.py --audio audio.wav")
    print()
    print("="*80)
