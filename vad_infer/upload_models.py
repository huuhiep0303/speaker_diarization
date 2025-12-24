"""
Upload models to Modal volume

This script uploads your fine-tuned VAD models to Modal volume
so they can be used by the API.

Usage:
    # With Modal CLI (recommended)
    modal run upload_models.py --model best_vad_nemo.nemo
    modal run upload_models.py --model best_vad_nemo.nemo --model vad_model.ckpt
    modal run upload_models.py --dir pretrained_models/nemo_vad
    
    # Or direct Python (with argparse)
    python upload_models.py --model best_vad_nemo.nemo
"""

import modal
import sys
from pathlib import Path
from typing import Optional


app = modal.App("upload-vad-models")


@app.function(
    volumes={
        "/models": modal.Volume.from_name("nemo-vad-models", create_if_missing=True)
    }
)
def upload_model(model_name: str, model_bytes: bytes):
    """
    Upload a single model to volume
    
    Args:
        model_name: Name of the model file
        model_bytes: Model file content as bytes
    """
    from pathlib import Path
    
    # Write to volume
    dest_path = Path(f"/models/{model_name}")
    
    try:
        with open(dest_path, 'wb') as f:
            f.write(model_bytes)
        
        size_mb = len(model_bytes) / 1024 / 1024
        
        print(f"✅ Uploaded: {model_name} ({size_mb:.2f} MB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False


@app.function(
    volumes={
        "/models": modal.Volume.from_name("nemo-vad-models", create_if_missing=True)
    }
)
def list_models_in_volume():
    """List all models in volume"""
    from pathlib import Path
    
    models_dir = Path("/models")
    
    if not models_dir.exists():
        return []
    
    models = []
    for model_file in models_dir.glob("*"):
        if model_file.suffix in [".nemo", ".ckpt"]:
            stat = model_file.stat()
            models.append({
                "name": model_file.name,
                "size_mb": stat.st_size / 1024 / 1024,
                "format": model_file.suffix[1:]
            })
    
    return models


def do_upload(model_files: list = None, directory: str = None):
    """Main upload logic"""
    
    print("\n" + "="*80)
    print("📤 UPLOAD MODELS TO MODAL VOLUME")
    print("="*80)
    print(f"📂 Current directory: {Path.cwd()}")
    print()
    
    # Collect model paths
    model_paths = []
    
    if directory:
        dir_path = Path(directory)
        if dir_path.exists():
            for ext in ['.nemo', '.ckpt']:
                found = list(dir_path.glob(f"*{ext}"))
                model_paths.extend(found)
            print(f"📁 Found {len(model_paths)} models in {directory}")
        else:
            print(f"❌ Directory not found: {directory}")
            print(f"   Looking for: {dir_path.absolute()}")
    
    if model_files:
        for m in model_files:
            p = Path(m)
            if p.exists():
                model_paths.append(p)
                print(f"✓ Found: {p.name}")
            else:
                print(f"⚠️  Not found: {p.name}")
                print(f"   Looking at: {p.absolute()}")
    
    if len(model_paths) == 0:
        print("\n❌ No valid model files found!")
        print("\n💡 Tips:")
        print("  • Check if files exist in current directory")
        print("  • Use full path: --model /path/to/best_vad_nemo.nemo")
        print("  • Or use --dir to specify directory")
        print("\nUsage:")
        print("  modal run upload_models.py --model best_vad_nemo.nemo")
        print("  modal run upload_models.py --dir ../../checkpoints/vad")
        return
    
    print(f"\n📦 Uploading {len(model_paths)} model(s)...")
    print()
    
    # Upload each model
    success_count = 0
    for model_path in model_paths:
        print(f"📤 Uploading {model_path.name} ({model_path.stat().st_size / 1024 / 1024:.2f} MB)...")
        try:
            # Read file bytes locally
            with open(model_path, 'rb') as f:
                model_bytes = f.read()
            
            # Upload to Modal volume
            result = upload_model.remote(model_path.name, model_bytes)
            if result:
                success_count += 1
        except Exception as e:
            print(f"   ❌ Upload failed: {e}")
    
    print()
    print(f"✅ Successfully uploaded {success_count}/{len(model_paths)} models")
    print()
    
    # List models in volume
    print("📋 Models in volume:")
    models = list_models_in_volume.remote()
    
    if len(models) == 0:
        print("   (empty)")
    else:
        for i, model_info in enumerate(models, 1):
            print(f"   {i}. {model_info['name']} ({model_info['size_mb']:.2f} MB) - {model_info['format']}")
    
    print()
    print("="*80)
    print("✅ UPLOAD COMPLETED")
    print("="*80)
    print()
    print("Next steps:")
    print("  1. Deploy API: modal deploy vad_api.py")
    print("  2. Test API: python vad_client.py --api-url <YOUR_API_URL> --audio audio.wav")
    print()


@app.local_entrypoint()
def main(model: str = None, dir: str = None):
    """
    Modal entrypoint for uploading models
    
    Note: Modal CLI doesn't support list arguments well,
    so we only accept single --model at a time with modal run.
    For multiple models, use Python directly or --dir option.
    """
    
    model_files = [model] if model else []
    do_upload(model_files=model_files, directory=dir)


if __name__ == "__main__":
    # Python argparse entrypoint (when running with python directly)
    import argparse
    import subprocess
    
    parser = argparse.ArgumentParser(
        description="Upload NeMo VAD models to Modal volume",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Recommended: Use Modal CLI
  modal run upload_models.py --model best_vad_nemo.nemo
  modal run upload_models.py --dir /path/to/models
  
  # Python direct (will call Modal CLI automatically)
  python upload_models.py --model best_vad_nemo.nemo
  python upload_models.py --dir /path/to/models
        """
    )
    
    parser.add_argument("--model", action="append", help="Model file to upload (can be specified multiple times)")
    parser.add_argument("--dir", type=str, help="Directory containing models to upload")
    
    args = parser.parse_args()
    
    if not args.model and not args.dir:
        parser.print_help()
        print("\n❌ Error: Either --model or --dir must be specified")
        sys.exit(1)
    
    # When running with python directly, we need to use Modal CLI
    # because .remote() calls require Modal context
    print("\n💡 Detected Python direct execution")
    print("   Calling Modal CLI automatically...\n")
    
    # Build modal command
    cmd = ["modal", "run", __file__]
    
    if args.model:
        # For multiple models with Python, we need to run modal multiple times
        # or use directory approach
        if len(args.model) > 1:
            print(f"📦 Uploading {len(args.model)} models...")
            print("   Running Modal for each model...\n")
            
            for model_file in args.model:
                model_path = Path(model_file)
                if model_path.exists():
                    print(f"📤 Uploading {model_path.name}...")
                    result = subprocess.run(
                        ["modal", "run", __file__, "--model", str(model_path.absolute())],
                        capture_output=False
                    )
                    if result.returncode != 0:
                        print(f"   ❌ Failed to upload {model_path.name}")
                else:
                    print(f"⚠️  File not found: {model_file}")
                    print(f"   Looking at: {model_path.absolute()}")
                print()
        else:
            # Single model
            model_path = Path(args.model[0])
            if not model_path.exists():
                print(f"❌ Model file not found: {args.model[0]}")
                print(f"   Looking at: {model_path.absolute()}")
                print(f"\n💡 Current directory: {Path.cwd()}")
                sys.exit(1)
            
            cmd.extend(["--model", str(model_path.absolute())])
            subprocess.run(cmd)
    
    elif args.dir:
        dir_path = Path(args.dir)
        if not dir_path.exists():
            print(f"❌ Directory not found: {args.dir}")
            print(f"   Looking at: {dir_path.absolute()}")
            sys.exit(1)
        
        cmd.extend(["--dir", str(dir_path.absolute())])
        subprocess.run(cmd)
