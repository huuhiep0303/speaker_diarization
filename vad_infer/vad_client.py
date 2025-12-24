"""
Client script to call NeMo VAD API on Modal

Usage:
    # Basic usage
    python vad_client.py --audio audio.wav
    
    # With custom model
    python vad_client.py --audio audio.wav --model vad_model.ckpt
    
    # With custom threshold
    python vad_client.py --audio audio.wav --threshold 0.6
    
    # Save RTTM to file
    python vad_client.py --audio audio.wav --output results/output.rttm
    
    # Batch processing
    python vad_client.py --audio_dir dataset/test --output_dir results
"""

import argparse
import base64
import json
import sys
from pathlib import Path
import requests
from datetime import datetime
from tqdm import tqdm


def read_audio_file(audio_path: str) -> bytes:
    """Read audio file as bytes"""
    with open(audio_path, 'rb') as f:
        return f.read()


def call_vad_api(
    api_url: str,
    audio_bytes: bytes,
    model: str = "best_vad_model.nemo",
    threshold: float = 0.5,
    filename: str = "audio"
):
    """
    Call VAD API with audio
    
    Args:
        api_url: API endpoint URL
        audio_bytes: Audio file bytes
        model: Model name
        threshold: VAD threshold
        filename: Audio filename
    
    Returns:
        API response dict
    """
    
    # Encode audio to base64
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    # Prepare request
    payload = {
        "audio": audio_b64,
        "model": model,
        "threshold": threshold,
        "filename": filename
    }
    
    # Call API
    try:
        response = requests.post(api_url, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}


def save_rttm(rttm_content: str, output_path: str):
    """Save RTTM content to file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(rttm_content)
    
    print(f"   💾 Saved RTTM: {output_path}")


def save_json(result: dict, output_path: str):
    """Save JSON results"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"   💾 Saved JSON: {output_path}")


def process_single_file(
    api_url: str,
    audio_path: str,
    model: str,
    threshold: float,
    output_dir: str = None,
    save_json_flag: bool = False
):
    """Process single audio file"""
    
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        return None
    
    print(f"\n🎵 Processing: {audio_path.name}")
    
    # Read audio
    audio_bytes = read_audio_file(str(audio_path))
    print(f"   Size: {len(audio_bytes) / 1024:.2f} KB")
    
    # Call API
    print(f"   🌐 Calling API...")
    result = call_vad_api(
        api_url=api_url,
        audio_bytes=audio_bytes,
        model=model,
        threshold=threshold,
        filename=audio_path.stem
    )
    
    # Check for errors
    if "error" in result:
        print(f"   ❌ Error: {result['error']}")
        return None
    
    # Print results
    if result.get("success"):
        print(f"   ✅ Success!")
        print(f"   Audio duration: {result['audio_duration']:.2f}s")
        print(f"   Speech segments: {result['num_segments']}")
        print(f"   Total speech: {result['total_speech_duration']:.2f}s ({result['speech_ratio']*100:.1f}%)")
        
        # Save outputs
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save RTTM
            rttm_path = output_dir / f"{audio_path.stem}.rttm"
            save_rttm(result['rttm'], str(rttm_path))
            
            # Save JSON if requested
            if save_json_flag:
                json_path = output_dir / f"{audio_path.stem}.json"
                save_json(result, str(json_path))
        
        return result
    
    else:
        print(f"   ❌ Inference failed")
        return None


def process_batch(
    api_url: str,
    audio_dir: str,
    model: str,
    threshold: float,
    output_dir: str,
    save_json_flag: bool = False
):
    """Process multiple audio files"""
    
    audio_dir = Path(audio_dir)
    
    if not audio_dir.exists():
        print(f"❌ Directory not found: {audio_dir}")
        return
    
    # Find audio files
    audio_files = []
    for ext in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
        audio_files.extend(audio_dir.glob(f"**/*{ext}"))
    
    if len(audio_files) == 0:
        print(f"❌ No audio files found in {audio_dir}")
        return
    
    print(f"\n📂 Found {len(audio_files)} audio files")
    print(f"📤 Processing with API: {api_url}")
    print()
    
    # Process each file
    results_summary = []
    
    for audio_path in tqdm(audio_files, desc="Processing"):
        result = process_single_file(
            api_url=api_url,
            audio_path=str(audio_path),
            model=model,
            threshold=threshold,
            output_dir=output_dir,
            save_json_flag=save_json_flag
        )
        
        if result:
            results_summary.append({
                "file": audio_path.name,
                "status": "success",
                "num_segments": result.get("num_segments", 0),
                "speech_duration": result.get("total_speech_duration", 0)
            })
        else:
            results_summary.append({
                "file": audio_path.name,
                "status": "error"
            })
    
    # Save batch summary
    summary_path = Path(output_dir) / "batch_summary.json"
    batch_summary = {
        "total_files": len(audio_files),
        "successful": sum(1 for r in results_summary if r["status"] == "success"),
        "failed": sum(1 for r in results_summary if r["status"] == "error"),
        "results": results_summary,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(summary_path, 'w') as f:
        json.dump(batch_summary, f, indent=2)
    
    print(f"\n💾 Batch summary saved: {summary_path}")
    print(f"\n📊 Summary:")
    print(f"   Total: {batch_summary['total_files']}")
    print(f"   Success: {batch_summary['successful']}")
    print(f"   Failed: {batch_summary['failed']}")


def check_health(api_url: str):
    """Check API health"""
    # Modal creates separate URLs for each endpoint
    # Convert infer_api URL to health URL
    if "infer-api" in api_url:
        health_url = api_url.replace("infer-api", "health")
    else:
        health_url = api_url.replace("/infer_api", "/health")
    
    try:
        response = requests.get(health_url, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        print(f"✅ API is healthy")
        print(f"   Service: {result.get('service', 'Unknown')}")
        print(f"   Version: {result.get('version', 'Unknown')}")
        print(f"   Status: {result.get('status', 'Unknown')}")
        
        return True
    
    except Exception as e:
        print(f"⚠️  Health check skipped (expected with Modal URLs)")
        print(f"   Will proceed with inference request")
        return True  # Don't block on health check for Modal


def list_models(api_url: str):
    """List available models"""
    # Modal creates separate URLs for each endpoint
    if "infer-api" in api_url:
        models_url = api_url.replace("infer-api", "list-models")
    else:
        models_url = api_url.replace("/infer_api", "/list_models")
    
    try:
        response = requests.get(models_url, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        models = result.get("models", [])
        
        if len(models) == 0:
            print("⚠️  No models found in volume")
        else:
            print(f"📦 Available models ({len(models)}):")
            for i, model in enumerate(models, 1):
                print(f"   {i}. {model['name']} ({model['size_mb']:.2f} MB) - {model['format']}")
        
        return models
    
    except Exception as e:
        print(f"⚠️  Could not list models: {e}")
        print(f"   Proceeding with default model")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Client for NeMo VAD API on Modal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python vad_client.py --audio audio.wav
  
  # With custom model
  python vad_client.py --audio audio.wav --model vad_model.ckpt
  
  # Batch processing
  python vad_client.py --audio_dir dataset/test --output_dir results
  
  # Check API health
  python vad_client.py --health
  
  # List available models
  python vad_client.py --list-models
        """
    )
    
    # API endpoint
    parser.add_argument("--api-url", type=str, required=True,
                       help="Modal API endpoint URL (get from 'modal deploy' output)")
    
    # Input
    parser.add_argument("--audio", type=str,
                       help="Path to single audio file")
    parser.add_argument("--audio_dir", type=str,
                       help="Path to directory with audio files (batch mode)")
    
    # Model parameters
    parser.add_argument("--model", type=str, default="best_vad_model.nemo",
                       help="Model name (default: best_vad_model.nemo)")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="VAD threshold (default: 0.5)")
    
    # Output
    parser.add_argument("--output", type=str,
                       help="Output RTTM file path (single file mode)")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory (default: results)")
    parser.add_argument("--save-json", action="store_true",
                       help="Save JSON results along with RTTM")
    
    # Utility commands
    parser.add_argument("--health", action="store_true",
                       help="Check API health")
    parser.add_argument("--list-models", action="store_true",
                       help="List available models")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🌐 NeMo VAD API Client")
    print("="*80)
    print(f"API URL: {args.api_url}")
    print("="*80)
    print()
    
    # Health check
    if args.health:
        check_health(args.api_url)
        return
    
    # List models
    if args.list_models:
        list_models(args.api_url)
        return
    
    # Check API is accessible (optional for Modal)
    print("🔍 Checking API...")
    check_health(args.api_url)
    print()
    
    # List available models (optional for Modal)
    print("📦 Checking available models...")
    models = list_models(args.api_url)
    print()
    
    # Validate model exists
    model_names = [m['name'] for m in models]
    if args.model not in model_names and len(models) > 0:
        print(f"⚠️  Warning: Model '{args.model}' not found in volume")
        print(f"   Available models: {', '.join(model_names)}")
        print()
    
    # Process audio
    if args.audio:
        # Single file mode
        output_dir = Path(args.output).parent if args.output else args.output_dir
        
        result = process_single_file(
            api_url=args.api_url,
            audio_path=args.audio,
            model=args.model,
            threshold=args.threshold,
            output_dir=output_dir,
            save_json_flag=args.save_json
        )
        
        if result and args.output:
            # Save to specific path
            save_rttm(result['rttm'], args.output)
    
    elif args.audio_dir:
        # Batch mode
        process_batch(
            api_url=args.api_url,
            audio_dir=args.audio_dir,
            model=args.model,
            threshold=args.threshold,
            output_dir=args.output_dir,
            save_json_flag=args.save_json
        )
    
    else:
        parser.error("Either --audio or --audio_dir must be specified (or use --health/--list-models)")
    
    print("\n" + "="*80)
    print("✅ COMPLETED")
    print("="*80)
    print()


if __name__ == "__main__":
    main()
