# """
# NeMo VAD Model Inference

# This script performs Voice Activity Detection inference using fine-tuned NeMo VAD models.
# Supports both .nemo and .ckpt checkpoint formats.

# Input:  audio.wav (or any audio file)
# Output: output.rttm (Rich Transcription Time Marked format)

# Usage:
#     # Inference with .nemo model
#     python vad_infer.py --model best_vad_nemo.nemo --audio audio.wav --output output.rttm
    
#     # Inference with .ckpt model  
#     python vad_infer.py --model vad_model.ckpt --audio audio.wav --output output.rttm
    
#     # With visualization
#     python vad_infer.py --model best_vad_nemo.nemo --audio audio.wav --visualize
    
#     # Adjust threshold
#     python vad_infer.py --model best_vad_nemo.nemo --audio audio.wav --threshold 0.6
    
#     # Compare both models
#     python vad_infer.py --model1 best_vad_nemo.nemo --model2 vad_model.ckpt --audio audio.wav --compare
# """

# import argparse
# import sys
# from pathlib import Path
# import numpy as np
# import torch
# import soundfile as sf
# from datetime import datetime
# import json


# def load_vad_model(model_path: str, device: str = "cuda"):
#     """
#     Load NeMo VAD model from checkpoint
    
#     Supports:
#     - .nemo format (NeMo exported model)
#     - .ckpt format (PyTorch Lightning checkpoint)
    
#     Args:
#         model_path: Path to model checkpoint
#         device: Device to load model on ('cuda' or 'cpu')
    
#     Returns:
#         Loaded VAD model
#     """
#     from nemo.collections.asr.models import EncDecClassificationModel
    
#     model_path = Path(model_path)
    
#     if not model_path.exists():
#         raise FileNotFoundError(f"Model file not found: {model_path}")
    
#     print(f"📦 Loading VAD model: {model_path.name}")
#     print(f"   Format: {model_path.suffix}")
#     print(f"   Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    
#     try:
#         if model_path.suffix == ".nemo":
#             # Load .nemo format (NeMo's native format)
#             model = EncDecClassificationModel.restore_from(str(model_path))
#             print(f"   ✅ Loaded .nemo model successfully")
            
#         elif model_path.suffix == ".ckpt":
#             # Load .ckpt format (PyTorch Lightning checkpoint)
#             model = EncDecClassificationModel.load_from_checkpoint(str(model_path))
#             print(f"   ✅ Loaded .ckpt model successfully")
            
#         else:
#             raise ValueError(f"Unsupported model format: {model_path.suffix}. Expected .nemo or .ckpt")
        
#         # Move to device and set eval mode
#         model.to(device)
#         model.eval()
        
#         print(f"   Device: {device}")
#         print()
        
#         return model
        
#     except Exception as e:
#         print(f"   ❌ Error loading model: {e}")
#         raise


# def load_audio(audio_path: str, target_sr: int = 16000):
#     """
#     Load audio file and resample to target sample rate
    
#     Args:
#         audio_path: Path to audio file
#         target_sr: Target sample rate (default: 16000 Hz for NeMo)
    
#     Returns:
#         audio_data: numpy array of audio samples
#         sample_rate: sample rate after resampling
#     """
#     print(f"🎵 Loading audio: {audio_path}")
    
#     audio_path = Path(audio_path)
    
#     if not audio_path.exists():
#         raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
#     # Load audio
#     audio, sr = sf.read(str(audio_path))
    
#     # Convert stereo to mono if needed
#     if len(audio.shape) > 1:
#         audio = audio.mean(axis=1)
#         print(f"   Converted stereo to mono")
    
#     # Resample if needed
#     if sr != target_sr:
#         import librosa
#         audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
#         print(f"   Resampled: {sr} Hz → {target_sr} Hz")
#         sr = target_sr
    
#     duration = len(audio) / sr
    
#     print(f"   Duration: {duration:.2f} seconds")
#     print(f"   Samples: {len(audio)}")
#     print(f"   Sample rate: {sr} Hz")
#     print()
    
#     return audio, sr


# def run_vad_inference(model, audio_data: np.ndarray, sample_rate: int, 
#                      threshold: float = 0.5, frame_length: float = 0.02, 
#                      frame_shift: float = 0.01):
#     """
#     Run VAD inference on audio
    
#     Args:
#         model: NeMo VAD model
#         audio_data: Audio samples (numpy array)
#         sample_rate: Sample rate of audio
#         threshold: VAD threshold (0-1)
#         frame_length: Frame length in seconds (default: 20ms)
#         frame_shift: Frame shift/stride in seconds (default: 10ms)
    
#     Returns:
#         speech_segments: List of (start, end) tuples in seconds
#         speech_probs: Frame-level speech probabilities
#     """
#     print(f"🔍 Running VAD inference...")
#     print(f"   Threshold: {threshold}")
#     print(f"   Frame length: {frame_length}s ({frame_length*1000:.0f}ms)")
#     print(f"   Frame shift: {frame_shift}s ({frame_shift*1000:.0f}ms)")
    
#     # Prepare input tensor
#     audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
#     audio_length = torch.tensor([len(audio_data)])
    
#     # Move to same device as model
#     device = next(model.parameters()).device
#     audio_tensor = audio_tensor.to(device)
#     audio_length = audio_length.to(device)
    
#     # Run inference
#     with torch.no_grad():
#         logits = model(input_signal=audio_tensor, input_signal_length=audio_length)
        
#         # Convert logits to probabilities
#         probs = torch.softmax(logits, dim=-1)
    
#     # Extract speech probability (class 1 = speech, class 0 = non-speech)
#     speech_probs = probs[0, :, 1].cpu().numpy()
    
#     print(f"   Output frames: {len(speech_probs)}")
#     print(f"   Speech probability range: [{speech_probs.min():.3f}, {speech_probs.max():.3f}]")
    
#     # Convert frame-level predictions to segments
#     speech_segments = frames_to_segments(speech_probs, threshold, frame_shift)
    
#     print(f"   ✅ Detected {len(speech_segments)} speech segments")
#     print()
    
#     return speech_segments, speech_probs


# def frames_to_segments(speech_probs: np.ndarray, threshold: float, frame_shift: float):
#     """
#     Convert frame-level speech probabilities to segments
    
#     Args:
#         speech_probs: Frame-level speech probabilities
#         threshold: Threshold for speech detection
#         frame_shift: Time between frames in seconds
    
#     Returns:
#         segments: List of (start, end) tuples in seconds
#     """
#     segments = []
#     in_speech = False
#     segment_start = 0
    
#     for i, prob in enumerate(speech_probs):
#         time = i * frame_shift
        
#         if prob >= threshold:
#             if not in_speech:
#                 # Start of speech segment
#                 in_speech = True
#                 segment_start = time
#         else:
#             if in_speech:
#                 # End of speech segment
#                 in_speech = False
#                 segments.append((segment_start, time))
    
#     # Handle case where audio ends during speech
#     if in_speech:
#         final_time = len(speech_probs) * frame_shift
#         segments.append((segment_start, final_time))
    
#     return segments


# def post_process_segments(segments, min_speech_duration: float = 0.2, 
#                          min_silence_duration: float = 0.3):
#     """
#     Post-process VAD segments
    
#     Operations:
#     1. Filter out very short speech segments (likely false alarms)
#     2. Merge segments separated by short silence (reduce fragmentation)
    
#     Args:
#         segments: List of (start, end) tuples
#         min_speech_duration: Minimum duration for speech segment (seconds)
#         min_silence_duration: Minimum gap to keep segments separate (seconds)
    
#     Returns:
#         processed_segments: Filtered and merged segments
#     """
#     if len(segments) == 0:
#         return []
    
#     print(f"📝 Post-processing segments...")
#     print(f"   Initial segments: {len(segments)}")
    
#     # Step 1: Filter short segments
#     filtered = []
#     for start, end in segments:
#         duration = end - start
#         if duration >= min_speech_duration:
#             filtered.append((start, end))
    
#     print(f"   After filtering (min {min_speech_duration}s): {len(filtered)}")
    
#     if len(filtered) == 0:
#         return []
    
#     # Step 2: Merge close segments
#     merged = [filtered[0]]
    
#     for current in filtered[1:]:
#         last = merged[-1]
#         gap = current[0] - last[1]
        
#         if gap < min_silence_duration:
#             # Merge segments
#             merged[-1] = (last[0], current[1])
#         else:
#             merged.append(current)
    
#     print(f"   After merging (min gap {min_silence_duration}s): {len(merged)}")
    
#     # Calculate statistics
#     total_speech = sum(end - start for start, end in merged)
#     print(f"   Total speech duration: {total_speech:.2f}s")
#     print()
    
#     return merged


# def save_rttm(segments, output_path: str, audio_id: str = "audio"):
#     """
#     Save segments in RTTM format
    
#     RTTM format (Rich Transcription Time Marked):
#     SPEAKER <file-id> 1 <start-time> <duration> <NA> <NA> <speaker-id> <NA> <NA>
    
#     For VAD, we use a single speaker label since we're only detecting speech vs non-speech
    
#     Args:
#         segments: List of (start, end) tuples in seconds
#         output_path: Path to output RTTM file
#         audio_id: Audio file identifier (default: "audio")
#     """
#     output_path = Path(output_path)
#     output_path.parent.mkdir(parents=True, exist_ok=True)
    
#     print(f"💾 Saving RTTM: {output_path}")
    
#     with open(output_path, 'w') as f:
#         for start, end in segments:
#             duration = end - start
            
#             # RTTM format line
#             # Type=SPEAKER, file_id, channel=1, start, duration, ortho, type, speaker, conf, slat
#             line = f"SPEAKER {audio_id} 1 {start:.3f} {duration:.3f} <NA> <NA> speech <NA> <NA>\n"
#             f.write(line)
    
#     print(f"   ✅ Saved {len(segments)} segments")
#     print(f"   File: {output_path}")
#     print()


# def save_json_results(segments, output_path: str, audio_path: str, 
#                      model_path: str, threshold: float, audio_duration: float):
#     """
#     Save detailed results in JSON format
    
#     Args:
#         segments: Speech segments
#         output_path: Path to output JSON file
#         audio_path: Original audio file path
#         model_path: Model checkpoint path
#         threshold: VAD threshold used
#         audio_duration: Total audio duration
#     """
#     output_path = Path(output_path)
    
#     total_speech = sum(end - start for start, end in segments)
#     speech_ratio = total_speech / audio_duration if audio_duration > 0 else 0
    
#     result = {
#         "audio_file": str(audio_path),
#         "audio_duration": float(audio_duration),
#         "model": str(model_path),
#         "threshold": float(threshold),
#         "total_speech_duration": float(total_speech),
#         "speech_ratio": float(speech_ratio),
#         "num_segments": len(segments),
#         "segments": [
#             {
#                 "start": float(start),
#                 "end": float(end),
#                 "duration": float(end - start)
#             }
#             for start, end in segments
#         ],
#         "timestamp": datetime.now().isoformat()
#     }
    
#     with open(output_path, 'w') as f:
#         json.dump(result, f, indent=2)
    
#     print(f"💾 Saved JSON: {output_path}")
#     print()


# def visualize_vad(audio_data: np.ndarray, sample_rate: int, segments, 
#                  speech_probs: np.ndarray, output_path: str, threshold: float):
#     """
#     Create visualization of VAD results
    
#     Args:
#         audio_data: Audio waveform
#         sample_rate: Sample rate
#         segments: Speech segments
#         speech_probs: Frame-level probabilities
#         output_path: Path to save plot
#         threshold: VAD threshold
#     """
#     import matplotlib.pyplot as plt
    
#     print(f"📊 Creating visualization...")
    
#     # Time axes
#     time_audio = np.arange(len(audio_data)) / sample_rate
#     time_probs = np.arange(len(speech_probs)) * 0.01  # 10ms frame shift
    
#     # Create figure
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    
#     # Plot 1: Waveform with speech segments highlighted
#     ax1.plot(time_audio, audio_data, linewidth=0.5, alpha=0.7, color='steelblue')
#     ax1.set_ylabel("Amplitude", fontsize=11)
#     ax1.set_title(f"Audio Waveform with VAD Segments", fontsize=13, fontweight='bold')
#     ax1.grid(True, alpha=0.3)
    
#     # Highlight speech segments
#     for i, (start, end) in enumerate(segments):
#         label = 'Speech' if i == 0 else ''
#         ax1.axvspan(start, end, alpha=0.3, color='green', label=label)
    
#     if len(segments) > 0:
#         ax1.legend(loc='upper right', fontsize=10)
    
#     # Plot 2: Speech probabilities
#     ax2.plot(time_probs, speech_probs, linewidth=1.5, color='darkblue', label='Speech Probability')
#     ax2.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, 
#                 label=f'Threshold ({threshold})')
#     ax2.fill_between(time_probs, 0, speech_probs, alpha=0.3, color='steelblue')
    
#     ax2.set_xlabel("Time (seconds)", fontsize=11)
#     ax2.set_ylabel("Speech Probability", fontsize=11)
#     ax2.set_title("VAD Predictions (Frame-level)", fontsize=13, fontweight='bold')
#     ax2.set_ylim([0, 1])
#     ax2.grid(True, alpha=0.3)
#     ax2.legend(loc='upper right', fontsize=10)
    
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=150, bbox_inches='tight')
#     plt.close()
    
#     print(f"   ✅ Visualization saved: {output_path}")
#     print()


# def compare_models(model1_path: str, model2_path: str, audio_path: str, 
#                   output_dir: str = "comparison_results", threshold: float = 0.5):
#     """
#     Compare results from two different models
    
#     Args:
#         model1_path: Path to first model
#         model2_path: Path to second model
#         audio_path: Path to audio file
#         output_dir: Directory to save comparison results
#         threshold: VAD threshold
#     """
#     print("\n" + "="*80)
#     print("🔬 COMPARING TWO MODELS")
#     print("="*80)
#     print(f"Model 1: {model1_path}")
#     print(f"Model 2: {model2_path}")
#     print("="*80)
#     print()
    
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     # Load audio once
#     audio_data, sample_rate = load_audio(audio_path)
#     audio_duration = len(audio_data) / sample_rate
    
#     results = {}
    
#     # Run inference with both models
#     for i, model_path in enumerate([model1_path, model2_path], 1):
#         print(f"\n{'='*80}")
#         print(f"MODEL {i}: {Path(model_path).name}")
#         print(f"{'='*80}\n")
        
#         # Load model
#         model = load_vad_model(model_path, device)
        
#         # Run inference
#         segments, speech_probs = run_vad_inference(
#             model, audio_data, sample_rate, threshold
#         )
        
#         # Post-process
#         segments = post_process_segments(segments)
        
#         # Save results
#         model_name = Path(model_path).stem
#         rttm_path = output_dir / f"{model_name}.rttm"
#         json_path = output_dir / f"{model_name}.json"
        
#         save_rttm(segments, str(rttm_path), Path(audio_path).stem)
#         save_json_results(segments, str(json_path), audio_path, 
#                          model_path, threshold, audio_duration)
        
#         results[f"model{i}"] = {
#             "name": model_name,
#             "path": model_path,
#             "num_segments": len(segments),
#             "total_speech": sum(end - start for start, end in segments),
#             "segments": segments,
#             "probs": speech_probs
#         }
    
#     # Generate comparison report
#     print("\n" + "="*80)
#     print("📊 COMPARISON SUMMARY")
#     print("="*80)
    
#     for i in [1, 2]:
#         r = results[f"model{i}"]
#         speech_ratio = r["total_speech"] / audio_duration * 100
        
#         print(f"\nModel {i}: {r['name']}")
#         print(f"  Segments: {r['num_segments']}")
#         print(f"  Total speech: {r['total_speech']:.2f}s ({speech_ratio:.1f}%)")
    
#     # Calculate differences
#     diff_segments = abs(results['model1']['num_segments'] - results['model2']['num_segments'])
#     diff_speech = abs(results['model1']['total_speech'] - results['model2']['total_speech'])
    
#     print(f"\n📈 Differences:")
#     print(f"  Segments: ±{diff_segments}")
#     print(f"  Total speech: ±{diff_speech:.2f}s")
    
#     print("\n" + "="*80)
#     print()
    
#     # Save comparison JSON
#     comparison_path = output_dir / "comparison.json"
#     comparison_data = {
#         "audio_file": str(audio_path),
#         "audio_duration": audio_duration,
#         "threshold": threshold,
#         "model1": {
#             "name": results['model1']['name'],
#             "path": str(results['model1']['path']),
#             "num_segments": results['model1']['num_segments'],
#             "total_speech": results['model1']['total_speech']
#         },
#         "model2": {
#             "name": results['model2']['name'],
#             "path": str(results['model2']['path']),
#             "num_segments": results['model2']['num_segments'],
#             "total_speech": results['model2']['total_speech']
#         },
#         "differences": {
#             "segments": diff_segments,
#             "total_speech": diff_speech
#         },
#         "timestamp": datetime.now().isoformat()
#     }
    
#     with open(comparison_path, 'w') as f:
#         json.dump(comparison_data, f, indent=2)
    
#     print(f"💾 Comparison saved: {comparison_path}")


# def main():
#     parser = argparse.ArgumentParser(
#         description="NeMo VAD Model Inference",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   # Basic inference
#   python vad_infer.py --model best_vad_nemo.nemo --audio audio.wav
  
#   # Custom output path
#   python vad_infer.py --model vad_model.ckpt --audio audio.wav --output results/output.rttm
  
#   # With visualization
#   python vad_infer.py --model best_vad_nemo.nemo --audio audio.wav --visualize
  
#   # Adjust threshold
#   python vad_infer.py --model best_vad_nemo.nemo --audio audio.wav --threshold 0.6
  
#   # Compare two models
#   python vad_infer.py --model1 best_vad_nemo.nemo --model2 vad_model.ckpt --audio audio.wav --compare
#         """
#     )
    
#     # Input arguments
#     parser.add_argument("--model", type=str, 
#                        help="Path to VAD model (.nemo or .ckpt)")
#     parser.add_argument("--audio", type=str, required=True,
#                        help="Path to input audio file")
#     parser.add_argument("--output", type=str, 
#                        help="Path to output RTTM file (default: auto-generated)")
    
#     # Model comparison mode
#     parser.add_argument("--model1", type=str,
#                        help="Path to first model for comparison")
#     parser.add_argument("--model2", type=str,
#                        help="Path to second model for comparison")
#     parser.add_argument("--compare", action="store_true",
#                        help="Enable comparison mode")
    
#     # VAD parameters
#     parser.add_argument("--threshold", type=float, default=0.5,
#                        help="VAD threshold (0-1, default: 0.5)")
#     parser.add_argument("--min-speech", type=float, default=0.2,
#                        help="Minimum speech duration in seconds (default: 0.2)")
#     parser.add_argument("--min-silence", type=float, default=0.3,
#                        help="Minimum silence gap in seconds (default: 0.3)")
    
#     # Device
#     parser.add_argument("--device", type=str, default="cuda",
#                        choices=["cuda", "cpu"],
#                        help="Device to use (default: cuda)")
    
#     # Output options
#     parser.add_argument("--visualize", action="store_true",
#                        help="Generate visualization plot")
#     parser.add_argument("--save-json", action="store_true",
#                        help="Save detailed results in JSON format")
    
#     args = parser.parse_args()
    
#     # Validate arguments
#     if args.compare:
#         if not args.model1 or not args.model2:
#             parser.error("--compare requires both --model1 and --model2")
        
#         compare_models(args.model1, args.model2, args.audio, 
#                       threshold=args.threshold)
#         return
    
#     if not args.model:
#         parser.error("--model is required (unless using --compare mode)")
    
#     # Check device availability
#     if args.device == "cuda" and not torch.cuda.is_available():
#         print("⚠️  CUDA not available, falling back to CPU")
#         args.device = "cpu"
    
#     # Print header
#     print("\n" + "="*80)
#     print("🎙️  NeMo VAD MODEL INFERENCE")
#     print("="*80)
#     print(f"Model: {args.model}")
#     print(f"Audio: {args.audio}")
#     print(f"Device: {args.device}")
#     print(f"Threshold: {args.threshold}")
#     print("="*80)
#     print()
    
#     try:
#         # Load model
#         model = load_vad_model(args.model, args.device)
        
#         # Load audio
#         audio_data, sample_rate = load_audio(args.audio)
#         audio_duration = len(audio_data) / sample_rate
        
#         # Run inference
#         segments, speech_probs = run_vad_inference(
#             model, audio_data, sample_rate, args.threshold
#         )
        
#         # Post-process segments
#         segments = post_process_segments(
#             segments, 
#             min_speech_duration=args.min_speech,
#             min_silence_duration=args.min_silence
#         )
        
#         # Determine output paths
#         if args.output:
#             output_rttm = args.output
#         else:
#             # Auto-generate output filename
#             audio_stem = Path(args.audio).stem
#             model_stem = Path(args.model).stem
#             output_rttm = f"{audio_stem}_{model_stem}_output.rttm"
        
#         output_base = Path(output_rttm).stem
#         output_dir = Path(output_rttm).parent
        
#         # Save RTTM
#         save_rttm(segments, output_rttm, Path(args.audio).stem)
        
#         # Save JSON if requested
#         if args.save_json:
#             json_path = output_dir / f"{output_base}.json"
#             save_json_results(segments, str(json_path), args.audio, 
#                             args.model, args.threshold, audio_duration)
        
#         # Visualize if requested
#         if args.visualize:
#             vis_path = output_dir / f"{output_base}_visualization.png"
#             visualize_vad(audio_data, sample_rate, segments, 
#                          speech_probs, str(vis_path), args.threshold)
        
#         # Print summary
#         print("="*80)
#         print("✅ INFERENCE COMPLETED SUCCESSFULLY")
#         print("="*80)
#         print(f"\n📊 Results:")
#         print(f"   Audio duration: {audio_duration:.2f}s")
#         print(f"   Speech segments: {len(segments)}")
        
#         total_speech = sum(end - start for start, end in segments)
#         speech_ratio = total_speech / audio_duration * 100 if audio_duration > 0 else 0
        
#         print(f"   Total speech: {total_speech:.2f}s ({speech_ratio:.1f}%)")
#         print(f"\n📁 Output files:")
#         print(f"   RTTM: {output_rttm}")
        
#         if args.save_json:
#             print(f"   JSON: {output_dir / f'{output_base}.json'}")
        
#         if args.visualize:
#             print(f"   Plot: {output_dir / f'{output_base}_visualization.png'}")
        
#         print("\n" + "="*80)
#         print()
        
#     except Exception as e:
#         print(f"\n❌ ERROR: {e}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)


# if __name__ == "__main__":
#     main()















import argparse
from pathlib import Path
import torch
import numpy as np
import soundfile as sf

from nemo.collections.asr.models import EncDecClassificationModel


# =========================
# Load model
# =========================
def load_model(model_path, device):
    model_path = Path(model_path)

    if model_path.suffix == ".nemo":
        model = EncDecClassificationModel.restore_from(str(model_path))
    elif model_path.suffix == ".ckpt":
        model = EncDecClassificationModel.load_from_checkpoint(str(model_path))
    else:
        raise ValueError("Model must be .nemo or .ckpt")

    model.to(device)
    model.eval()
    return model


# =========================
# Load audio
# =========================
def load_audio(audio_path, target_sr=16000):
    audio, sr = sf.read(audio_path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, sr, target_sr)

    return audio, target_sr


# =========================
# VAD inference
# =========================
def vad_infer(model, audio, sr, threshold=0.5, frame_shift=0.01):
    device = next(model.parameters()).device

    audio_tensor = torch.tensor(audio).float().unsqueeze(0).to(device)
    length_tensor = torch.tensor([len(audio)]).to(device)

    with torch.no_grad():
        logits = model(
            input_signal=audio_tensor,
            input_signal_length=length_tensor
        )
        probs = torch.softmax(logits, dim=-1)

    speech_probs = probs[0, :, 1].cpu().numpy()

    segments = []
    in_speech = False
    start = 0.0

    for i, p in enumerate(speech_probs):
        t = i * frame_shift
        if p >= threshold and not in_speech:
            in_speech = True
            start = t
        elif p < threshold and in_speech:
            in_speech = False
            segments.append((start, t))

    if in_speech:
        segments.append((start, len(speech_probs) * frame_shift))

    return segments


# =========================
# Save RTTM
# =========================
def save_rttm(segments, out_path, audio_id):
    with open(out_path, "w") as f:
        for s, e in segments:
            dur = e - s
            f.write(
                f"SPEAKER {audio_id} 1 {s:.3f} {dur:.3f} <NA> <NA> speech <NA> <NA>\n"
            )


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser("Simple NeMo VAD inference")
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", default="output.rttm")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available → CPU")
        args.device = "cpu"

    print("🔹 Loading model...")
    model = load_model(args.model, args.device)

    print("🔹 Loading audio...")
    audio, sr = load_audio(args.audio)

    print("🔹 Running VAD inference...")
    segments = vad_infer(model, audio, sr, args.threshold)

    print("🔹 Saving RTTM...")
    save_rttm(segments, args.output, Path(args.audio).stem)

    print(f"✅ Done. Speech segments: {len(segments)}")
    print(f"📄 RTTM saved to: {args.output}")


if __name__ == "__main__":
    main()
