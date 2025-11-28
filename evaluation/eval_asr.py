"""
Evaluation script for ASR/Transcription Quality
Đánh giá chất lượng transcription của các model:
- realtime_diarization_improved.py (Whisper)
- sen_voice.py (SenseVoice) 
- senvoi_spebrai_fixed.py (SenseVoice + SpeechBrain)

Metrics:
- WER (Word Error Rate) - cho tiếng Nhật sử dụng Sudachi tokenizer
- CER (Character Error Rate)
- RTF (Real-Time Factor)

Dataset: JVS Corpus (Japanese audio)
"""

import os
import sys
import csv
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from jiwer import wer, cer
    import jiwer.transforms as tr
    import regex as re
    from sudachipy import dictionary, tokenizer
except ImportError:
    print("ERROR: Missing required packages. Install with:")
    print("  pip install jiwer regex sudachipy")
    sys.exit(1)

# Configuration
RESULTS_DIR = Path(__file__).parent / "eval_results"
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================
#   JAPANESE TOKENIZER FOR WER
# ============================================
class JPTokenizer:
    """Tokenizer for Japanese text using Sudachi"""
    def __init__(self):
        self.tok = dictionary.Dictionary().create()
        self.mode = tokenizer.Tokenizer.SplitMode.A

    def _tokenize_str(self, text: str):
        return [m.surface() for m in self.tok.tokenize(text, self.mode)]

    def __call__(self, text):
        # Case 1: string
        if isinstance(text, str):
            return [self._tokenize_str(text)]

        # Case 2: list-of-strings
        if isinstance(text, list):
            result = []
            for t in text:
                if isinstance(t, str):
                    result.append(self._tokenize_str(t))
                elif isinstance(t, list):
                    result.append([self._tokenize_str(x) for x in t])
            return result

        raise TypeError(f"Unsupported type for JPTokenizer: {type(text)}")


# Transform for Japanese WER calculation
wer_ja = tr.Compose([
    tr.RemoveMultipleSpaces(),
    tr.Strip(),
    JPTokenizer(),
])


def get_audio_duration(file_path):
    """Get audio duration in seconds"""
    try:
        import librosa
        duration = librosa.get_duration(path=file_path)
        return duration
    except:
        try:
            import soundfile as sf
            info = sf.info(file_path)
            return info.duration
        except Exception as e:
            print(f"Warning: Could not get duration for {file_path}: {e}")
            return 0.0


def normalize_text_japanese(text):
    """
    Normalize Japanese text for evaluation.
    Loại bỏ punctuation, tags, và normalize spaces.
    """
    # Lowercase
    text = text.lower()
    
    # Remove punctuation (Japanese + common symbols)
    pattern = r"[\p{P}～~＋＝＄|]+"
    text = re.sub(pattern, "", text)
    
    # Remove tags (e.g., <|ja|>, <|nospeech|>)
    text = re.sub(r"<\|[^|>]+\|>", "", text)
    text = re.sub(r"<[^>]*>", "", text)
    
    # Remove English letters and spaces for Japanese evaluation
    # (Keep only Japanese characters)
    text = re.sub(r"[A-Za-z\s]+", "", text).strip()
    
    return text


def eval_score(ground_truth, prediction):
    """
    Calculate WER and CER scores for Japanese text.
    
    Args:
        ground_truth: Ground truth transcript (Japanese)
        prediction: Model prediction (Japanese)
    
    Returns:
        (wer_score, cer_score)
    """
    # Normalize texts
    ground_truth = normalize_text_japanese(ground_truth)
    prediction = normalize_text_japanese(prediction)
    
    # Calculate WER with Japanese tokenization
    try:
        wer_score = wer(
            ground_truth, 
            prediction, 
            reference_transform=wer_ja, 
            hypothesis_transform=wer_ja
        )
    except Exception as e:
        print(f"Warning: WER calculation failed: {e}")
        wer_score = 1.0
    
    # Calculate CER (character-level)
    try:
        cer_score = cer(ground_truth, prediction)
    except Exception as e:
        print(f"Warning: CER calculation failed: {e}")
        cer_score = 1.0
    
    return wer_score, cer_score


# ============================================
#        ASR MODEL WRAPPERS
# ============================================
class BaseASR:
    """Base class for ASR models"""
    
    def __init__(self, model_name):
        self.model_name = model_name
    
    def transcribe(self, audio_path):
        """Transcribe audio file. Override in subclass."""
        raise NotImplementedError


class WhisperASR(BaseASR):
    """Whisper ASR using faster-whisper"""
    
    def __init__(self, model_name="small", device="cpu", compute_type="int8"):
        super().__init__(f"whisper-{model_name}")
        self.device = device
        self.compute_type = compute_type
        
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(
                model_name, 
                device=device, 
                compute_type=compute_type
            )
            print(f"✓ Loaded Whisper model: {model_name} ({device}, {compute_type})")
        except ImportError:
            print("ERROR: faster-whisper not installed")
            print("Install with: pip install faster-whisper")
            sys.exit(1)
    
    def transcribe(self, audio_path):
        """Transcribe audio file"""
        try:
            segments, info = self.model.transcribe(
                audio_path,
                language="ja",  # Japanese
                vad_filter=False,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                condition_on_previous_text=True,
                without_timestamps=True
            )
            
            texts = [seg.text.strip() for seg in segments if seg.text.strip()]
            return "".join(texts)  # Join without spaces for Japanese
            
        except Exception as e:
            print(f"Error transcribing {audio_path}: {e}")
            return ""


class SenseVoiceASR(BaseASR):
    """SenseVoice ASR using FunASR"""
    
    def __init__(self, model_name="FunAudioLLM/SenseVoiceSmall", device="cpu"):
        super().__init__("sensevoice")
        self.device = device
        
        try:
            from funasr import AutoModel
            self.model = AutoModel(
                model=model_name,
                device=device,
                hub="hf",
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000}
            )
            print(f"✓ Loaded SenseVoice model ({device})")
        except ImportError:
            print("ERROR: funasr not installed")
            print("Install with: pip install funasr")
            sys.exit(1)
    
    def _clean_text(self, text):
        """Clean SenseVoice output - remove special tokens"""
        # Remove language tags like <|ja|>, <|en|>, etc.
        text = re.sub(r"<\|[^|>]+\|>", "", text).strip()
        # Remove spaces (for Japanese)
        text = text.replace(" ", "")
        return text
    
    def transcribe(self, audio_path):
        """Transcribe audio file"""
        try:
            res = self.model.generate(
                input=audio_path,
                language="ja",  # Japanese
                use_itn=False,
                batch_size_s=20
            )
            
            if res and len(res) > 0:
                raw = res[0]["text"]
                return self._clean_text(raw)
            else:
                return ""
            
        except Exception as e:
            print(f"Error transcribing {audio_path}: {e}")
            return ""


class SenseVoiceSpeechBrainASR(BaseASR):
    """SenseVoice + SpeechBrain combined model (from senvoi_spebrai_fixed.py)"""
    
    def __init__(self, device="cpu"):
        super().__init__("sensevoice-speechbrain")
        self.device = device
        
        print(f"Loading SenseVoice + SpeechBrain on {device}...")
        
        # Fix huggingface_hub compatibility
        try:
            import huggingface_hub
            _original_hf_download = huggingface_hub.hf_hub_download
            
            def _patched_hf_download(*args, use_auth_token=None, token=None, **kwargs):
                """Convert use_auth_token to token for compatibility"""
                if token is None and use_auth_token is not None:
                    token = use_auth_token
                return _original_hf_download(*args, token=token, **kwargs)
            
            huggingface_hub.hf_hub_download = _patched_hf_download
        except Exception as e:
            print(f"Warning: Could not apply huggingface_hub patch: {e}")
        
        # Import required modules
        try:
            from funasr import AutoModel
            from speechbrain.pretrained import EncoderClassifier
            import torch
        except ImportError as e:
            raise ImportError(f"Failed to import required modules: {e}")
        
        # Load SenseVoice ASR model
        print("  Loading SenseVoice ASR...")
        self.asr_model = AutoModel(
            model="FunAudioLLM/SenseVoiceSmall",
            hub="hf",
            device=device,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            disable_pbar=True,
            disable_log=True,
        )
        
        # Load SpeechBrain speaker model
        print("  Loading SpeechBrain speaker model...")
        pretrained_dir = Path(__file__).parent.parent / "pretrained_models" / "spkrec-ecapa-voxceleb"
        
        try:
            # Try to load from local directory first
            if pretrained_dir.exists():
                self.speaker_model = EncoderClassifier.from_hparams(
                    source=str(pretrained_dir),
                    run_opts={"device": device}
                )
            else:
                # Download from HuggingFace if not exists locally
                self.speaker_model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=str(pretrained_dir),
                    run_opts={"device": device}
                )
        except Exception as e:
            print(f"Warning: Error loading with savedir: {e}")
            print("Trying without savedir...")
            # Fallback: try without savedir
            self.speaker_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": device}
            )
        
        print("✓ Models loaded successfully")
    
    def _clean_text(self, text):
        """Remove speaker labels and clean text"""
        # Remove speaker tags like <spk_01>, <spk_02>
        text = re.sub(r'<spk_\d+>\s*', '', text)
        # Remove language tags
        text = re.sub(r"<\|[^|>]+\|>", "", text)
        # Remove spaces (for Japanese)
        text = text.replace(" ", "").strip()
        return text
    
    def transcribe(self, audio_path):
        """Transcribe audio file (speaker labels removed for ASR evaluation)"""
        try:
            res = self.asr_model.generate(
                input=str(audio_path),
                cache={},
                language="ja",
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )
            
            if res and len(res) > 0:
                result_item = res[0]
                if isinstance(result_item, dict):
                    text = result_item.get("text", "")
                else:
                    text = str(result_item)
                
                # Clean text: remove speaker labels for ASR evaluation
                return self._clean_text(text)
            else:
                return ""
            
        except Exception as e:
            print(f"Error transcribing {audio_path}: {e}")
            return ""


# ============================================
#   DATASET AND EVALUATION
# ============================================
def load_dataset(csv_file):
    """Load dataset from CSV file (JVS format)"""
    dataset = []
    
    csv_path = Path(csv_file)
    if not csv_path.exists():
        print(f"Error: Dataset file not found: {csv_file}")
        return dataset
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset.append(row)
    
    return dataset


def evaluate_model_with_checkpoint(model, dataset, checkpoint_file="eval_results_checkpoint.csv"):
    """
    Evaluate ASR model on dataset with checkpoint mechanism.
    
    Args:
        model: ASR model instance
        dataset: List of dict with audio file info
        checkpoint_file: Path to checkpoint CSV file
    
    Returns:
        (summary dict, results list)
    """
    checkpoint_path = RESULTS_DIR / checkpoint_file
    completed_files = set()
    
    # Load checkpoint if exists
    if checkpoint_path.exists():
        print(f"✓ Found checkpoint file: {checkpoint_path}")
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                if len(row) >= 1:
                    completed_files.add(row[0])  # file_path
        print(f"  Already completed: {len(completed_files)} samples")
    else:
        # Create new checkpoint file with header
        with open(checkpoint_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "file_path", "ground_truth", "prediction", 
                "wer", "cer", "rtf", "audio_duration", "processing_time"
            ])
        print(f"✓ Created new checkpoint file: {checkpoint_path}")
    
    # Evaluate dataset
    print(f"\nEvaluating {model.model_name} on {len(dataset)} samples...")
    print(f"Remaining: {len(dataset) - len(completed_files)} samples\n")
    
    for item in tqdm(dataset, desc="Evaluating"):
        wav_path = item.get('wav_path')
        ground_truth = item.get('transcript', '')
        
        if not wav_path or not ground_truth:
            continue
        
        # Skip if already processed
        if wav_path in completed_files:
            continue
        
        # Check if file exists
        if not Path(wav_path).exists():
            print(f"\nWarning: File not found: {wav_path}")
            continue
        
        try:
            # Get audio duration
            audio_duration = get_audio_duration(wav_path)
            
            # Transcribe and measure time
            start_time = time.time()
            prediction = model.transcribe(wav_path)
            processing_time = time.time() - start_time
            
            # Calculate RTF
            rtf = processing_time / audio_duration if audio_duration > 0 else 0.0
            
            # Skip if no prediction
            if not prediction:
                print(f"\nWarning: No prediction for {wav_path}")
                continue
            
            # Calculate scores
            wer_score, cer_score = eval_score(ground_truth, prediction)
            
            # Save result immediately (checkpoint)
            with open(checkpoint_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    wav_path, ground_truth, prediction,
                    wer_score, cer_score, rtf, audio_duration, processing_time
                ])
            
        except Exception as e:
            print(f"\nError processing {wav_path}: {e}")
            print("Progress saved in checkpoint. You can resume by running again.")
            raise
    
    # Calculate summary from checkpoint file
    print("\nCalculating final statistics...")
    
    wer_scores = []
    cer_scores = []
    rtf_scores = []
    
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 8:
                try:
                    wer_scores.append(float(row[3]))
                    cer_scores.append(float(row[4]))
                    rtf_scores.append(float(row[5]))
                except ValueError:
                    continue
    
    # Calculate averages
    summary = {
        'model': model.model_name,
        'num_samples': len(wer_scores),
        'avg_wer': float(np.mean(wer_scores)) if wer_scores else 0.0,
        'avg_cer': float(np.mean(cer_scores)) if cer_scores else 0.0,
        'avg_rtf': float(np.mean(rtf_scores)) if rtf_scores else 0.0,
        'median_rtf': float(np.median(rtf_scores)) if rtf_scores else 0.0,
        'min_rtf': float(np.min(rtf_scores)) if rtf_scores else 0.0,
        'max_rtf': float(np.max(rtf_scores)) if rtf_scores else 0.0,
        'timestamp': datetime.now().isoformat(),
        'checkpoint_file': str(checkpoint_path)
    }
    
    return summary, checkpoint_path


def print_summary(summary):
    """Print evaluation summary"""
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Model:              {summary['model']}")
    print(f"Samples evaluated:  {summary['num_samples']}")
    print(f"Average WER:        {summary['avg_wer']:.4f} ({summary['avg_wer']*100:.2f}%)")
    print(f"Average CER:        {summary['avg_cer']:.4f} ({summary['avg_cer']*100:.2f}%)")
    print(f"Average RTF:        {summary['avg_rtf']:.4f}")
    print(f"Median RTF:         {summary['median_rtf']:.4f}")
    print(f"RTF Range:          {summary['min_rtf']:.4f} - {summary['max_rtf']:.4f}")
    
    if summary['avg_rtf'] < 1.0:
        speedup = 1.0 / summary['avg_rtf']
        print(f"  → {speedup:.2f}x faster than realtime ✓")
    elif summary['avg_rtf'] == 1.0:
        print(f"  → Exactly realtime")
    else:
        slowdown = summary['avg_rtf']
        print(f"  → {slowdown:.2f}x slower than realtime ✗")
    
    print(f"\nCheckpoint file:    {summary['checkpoint_file']}")
    print("="*70 + "\n")


def save_summary_json(summary, output_file):
    """Save summary to JSON file"""
    output_path = RESULTS_DIR / output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved summary JSON: {output_path}")


# ============================================
#   MAIN EVALUATION
# ============================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluate ASR Models on JVS Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate Whisper Small on CPU
  python eval_asr.py --dataset dataset_400_testcases.csv --model whisper --whisper_size small --device cpu
  
  # Evaluate SenseVoice on CUDA
  python eval_asr.py --dataset dataset_400_testcases.csv --model sensevoice --device cuda
  
  # Evaluate SenseVoice + SpeechBrain (combined model)
  python eval_asr.py --dataset dataset_400_testcases.csv --model sensevoice-speechbrain --device cpu
  
  # Resume from checkpoint
  python eval_asr.py --dataset dataset_400_testcases.csv --model whisper --resume
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='dataset_400_testcases.csv',
        help='Path to dataset CSV file (default: dataset_400_testcases.csv)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='whisper',
        choices=['whisper', 'sensevoice', 'sensevoice-speechbrain'],
        help='Model to evaluate: whisper, sensevoice, or sensevoice-speechbrain (default: whisper)'
    )
    parser.add_argument(
        '--whisper_size',
        type=str,
        default='small',
        choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v1', 'large-v2', 'large-v3', 'turbo'],
        help='Whisper model size (default: small)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use (default: cpu)'
    )
    parser.add_argument(
        '--compute_type',
        type=str,
        default='int8',
        choices=['int8', 'float16', 'float32'],
        help='Compute type for Whisper (default: int8)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint if exists'
    )
    
    args = parser.parse_args()
    
    # Print header
    print(f"\n{'='*70}")
    print("ASR EVALUATION FOR JAPANESE (JVS DATASET)")
    print(f"{'='*70}")
    print(f"Model:          {args.model}")
    print(f"Dataset:        {args.dataset}")
    print(f"Device:         {args.device}")
    if args.model == 'whisper':
        print(f"Whisper Size:   {args.whisper_size}")
        print(f"Compute Type:   {args.compute_type}")
    print(f"Resume:         {args.resume}")
    print(f"{'='*70}\n")
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args.dataset)
    
    if not dataset:
        print("Error: No data loaded!")
        return
    
    print(f"✓ Loaded {len(dataset)} samples from {args.dataset}")
    
    # Show sample
    sample = dataset[0]
    print(f"\nSample entry:")
    print(f"  Speaker:    {sample.get('speaker', 'N/A')}")
    print(f"  Category:   {sample.get('category', 'N/A')}")
    print(f"  File:       {sample.get('wav_path', 'N/A')}")
    print(f"  Transcript: {sample.get('transcript', 'N/A')[:50]}...")
    
    # Initialize model
    print(f"\nInitializing {args.model} model...")
    if args.model == 'whisper':
        model = WhisperASR(
            model_name=args.whisper_size,
            device=args.device,
            compute_type=args.compute_type
        )
    elif args.model == 'sensevoice':
        model = SenseVoiceASR(device=args.device)
    elif args.model == 'sensevoice-speechbrain':
        model = SenseVoiceSpeechBrainASR(device=args.device)
    else:
        print(f"Error: Unknown model {args.model}")
        return
    
    # Setup checkpoint file
    checkpoint_name = f"eval_{model.model_name}_checkpoint.csv"
    
    # Evaluate
    print("\nStarting evaluation...")
    print("Note: Progress is auto-saved. You can stop and resume anytime.\n")
    
    summary, checkpoint_path = evaluate_model_with_checkpoint(
        model, 
        dataset, 
        checkpoint_file=checkpoint_name
    )
    
    # Print summary
    print_summary(summary)
    
    # Save summary JSON
    summary_json_name = f"eval_{model.model_name}_summary.json"
    save_summary_json(summary, summary_json_name)
    
    print("\n✓ Evaluation completed successfully!")
    print(f"\nResults:")
    print(f"  - Checkpoint CSV: {checkpoint_path}")
    print(f"  - Summary JSON:   {RESULTS_DIR / summary_json_name}")
    print(f"\nTo view results:")
    print(f"  - Open CSV in Excel: {checkpoint_path}")
    print(f"  - Check summary JSON: {RESULTS_DIR / summary_json_name}")
    print()


if __name__ == "__main__":
    main()
