"""
Evaluation Script for Fusion Speaker Diarization Models
Đánh giá các fusion models kết hợp ASR và Speaker Diarization:

Fusion Models:
1. Whisper + SpeechBrain (realtime_diarization_improved.py)
2. SenseVoice + SpeechBrain (senvoi_spebrai_fixed.py)
3. Whisper + PyAnnote (whisper_pyannote.py)
4. SenseVoice + PyAnnote (sensevoice_pyannote.py)
5. Whisper + NeMo (whisper_nemo.py)
6. SenseVoice + NeMo (sensevoice_nemo.py)

Metrics:
- Speaker Diarization: Sử dụng speaker embeddings đã được đánh giá trong eval_diarization.py
- ASR: WER (Word Error Rate), CER (Character Error Rate)

Dataset: JVS Corpus (Japanese audio with transcripts)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm

# Configuration
RESULTS_DIR = Path(__file__).parent / "eval_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Fusion models configuration
FUSION_MODELS = {
    "whisper_speechbrain": {
        "name": "Whisper + SpeechBrain",
        "asr": "faster-whisper",
        "speaker": "SpeechBrain ECAPA-TDNN",
        "script": "realtime_diarization_improved.py",
        "note": "Original baseline"
    },
    "sensevoice_speechbrain": {
        "name": "SenseVoice + SpeechBrain",
        "asr": "FunAudioLLM/SenseVoiceSmall",
        "speaker": "SpeechBrain ECAPA-TDNN",
        "script": "senvoi_spebrai_fixed.py",
        "note": "Already implemented"
    },
    "whisper_pyannote": {
        "name": "Whisper + PyAnnote",
        "asr": "faster-whisper",
        "speaker": "PyAnnote WeSpeaker-ResNet34",
        "script": "whisper_pyannote.py",
        "note": "New fusion model"
    },
    "sensevoice_pyannote": {
        "name": "SenseVoice + PyAnnote",
        "asr": "FunAudioLLM/SenseVoiceSmall",
        "speaker": "PyAnnote WeSpeaker-ResNet34",
        "script": "sensevoice_pyannote.py",
        "note": "New fusion model"
    },
    "whisper_nemo": {
        "name": "Whisper + NeMo",
        "asr": "faster-whisper",
        "speaker": "NeMo TitaNet Large",
        "script": "whisper_nemo.py",
        "note": "New fusion model"
    },
    "sensevoice_nemo": {
        "name": "SenseVoice + NeMo",
        "asr": "FunAudioLLM/SenseVoiceSmall",
        "speaker": "NeMo TitaNet Large",
        "script": "sensevoice_nemo.py",
        "note": "New fusion model"
    }
}


def print_summary():
    """Print summary of fusion models and their components"""
    print("="*80)
    print("FUSION SPEAKER DIARIZATION MODELS EVALUATION")
    print("="*80)
    print("\nAvailable Fusion Models:")
    print("-" * 80)
    
    for model_id, info in FUSION_MODELS.items():
        print(f"\n{info['name']}")
        print(f"  ASR Model:     {info['asr']}")
        print(f"  Speaker Model: {info['speaker']}")
        print(f"  Script:        {info['script']}")
        print(f"  Note:          {info['note']}")
    
    print("\n" + "="*80)
    print("\nEvaluation Metrics:")
    print("-" * 80)
    print("1. Speaker Diarization Performance:")
    print("   - Based on speaker embedding comparison from eval_diarization.py")
    print("   - SpeechBrain: EER = 15.57%, AUC = 0.9353")
    print("   - NeMo:        EER = 14.89%, AUC = 0.9403")
    print("   - PyAnnote:    (Run eval_diarization.py to get results)")
    print("\n2. ASR Performance:")
    print("   - WER (Word Error Rate)")
    print("   - CER (Character Error Rate)")
    print("   - Requires ground truth transcripts from dataset")
    print("\n" + "="*80)


def generate_summary_report(output_file="fusion_models_summary.txt"):
    """Generate comprehensive summary report"""
    
    output_path = RESULTS_DIR / output_file
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("FUSION SPEAKER DIARIZATION MODELS - COMPREHENSIVE SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overview
        f.write("OVERVIEW\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Fusion Models: {len(FUSION_MODELS)}\n\n")
        
        # Model details
        f.write("FUSION MODELS DETAILS\n")
        f.write("-" * 80 + "\n\n")
        
        for model_id, info in FUSION_MODELS.items():
            f.write(f"{model_id.upper()}: {info['name']}\n")
            f.write(f"  ASR Component:     {info['asr']}\n")
            f.write(f"  Speaker Component: {info['speaker']}\n")
            f.write(f"  Implementation:    {info['script']}\n")
            f.write(f"  Status:            {info['note']}\n")
            f.write("\n")
        
        # Speaker embedding performance (from eval_diarization.py results)
        f.write("\nSPEAKER EMBEDDING PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write("Based on JVS dataset evaluation (eval_diarization.py):\n\n")
        
        f.write("SpeechBrain ECAPA-TDNN:\n")
        f.write("  Used by: Whisper+SpeechBrain, SenseVoice+SpeechBrain\n")
        f.write("  EER:  15.57%\n")
        f.write("  AUC:  0.9353\n")
        f.write("  Best F1: 86.38%\n\n")
        
        f.write("NeMo TitaNet Large:\n")
        f.write("  Used by: Whisper+NeMo, SenseVoice+NeMo\n")
        f.write("  EER:  14.89%\n")
        f.write("  AUC:  0.9403\n")
        f.write("  Best F1: 87.02%\n\n")
        
        f.write("PyAnnote WeSpeaker-ResNet34:\n")
        f.write("  Used by: Whisper+PyAnnote, SenseVoice+PyAnnote\n")
        f.write("  Status: Run eval_diarization.py to get results\n\n")
        
        # ASR comparison
        f.write("\nASR MODEL COMPARISON\n")
        f.write("-" * 80 + "\n")
        f.write("Whisper (faster-whisper):\n")
        f.write("  - Multilingual support (99 languages)\n")
        f.write("  - Strong for English, European languages\n")
        f.write("  - Model sizes: tiny, base, small, medium, large-v3\n")
        f.write("  - Used by: Whisper+SpeechBrain, Whisper+PyAnnote, Whisper+NeMo\n\n")
        
        f.write("SenseVoice (FunAudioLLM):\n")
        f.write("  - Optimized for Chinese, English, Japanese, Korean, Cantonese\n")
        f.write("  - Includes emotion recognition\n")
        f.write("  - Includes event detection (Speech, Music, Applause)\n")
        f.write("  - Used by: SenseVoice+SpeechBrain, SenseVoice+PyAnnote, SenseVoice+NeMo\n\n")
        
        # Usage recommendations
        f.write("\nUSAGE RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n")
        f.write("1. Best for English/European languages:\n")
        f.write("   - Whisper + NeMo (best speaker performance)\n")
        f.write("   - Whisper + PyAnnote (alternative)\n\n")
        
        f.write("2. Best for Asian languages (Chinese, Japanese, Korean):\n")
        f.write("   - SenseVoice + NeMo (best speaker performance)\n")
        f.write("   - SenseVoice + PyAnnote (alternative)\n\n")
        
        f.write("3. If you need emotion detection:\n")
        f.write("   - Any SenseVoice-based model\n\n")
        
        f.write("4. Most stable (already tested):\n")
        f.write("   - Whisper + SpeechBrain (original baseline)\n")
        f.write("   - SenseVoice + SpeechBrain\n\n")
        
        # Implementation files
        f.write("\nIMPLEMENTATION FILES\n")
        f.write("-" * 80 + "\n")
        for model_id, info in FUSION_MODELS.items():
            script_path = Path(__file__).parent.parent / info['script']
            status = "✓ Exists" if script_path.exists() else "✗ Missing"
            f.write(f"{status}: {info['script']}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("\nTO RUN EVALUATION:\n")
        f.write("1. For speaker embedding evaluation:\n")
        f.write("   python eval_diarization.py --dataset <path_to_jvs>\n\n")
        f.write("2. For single audio file testing:\n")
        f.write("   python whisper_pyannote.py --audio_file <audio.wav>\n")
        f.write("   python sensevoice_nemo.py --audio_file <audio.wav>\n")
        f.write("   etc.\n\n")
        f.write("3. For ASR evaluation (requires ground truth):\n")
        f.write("   # Create custom evaluation script based on your dataset\n")
        f.write("="*80 + "\n")
    
    print(f"\n✅ Summary report saved to: {output_path}")
    return output_path


def check_implementations():
    """Check which fusion model implementations exist"""
    print("\n" + "="*80)
    print("CHECKING FUSION MODEL IMPLEMENTATIONS")
    print("="*80 + "\n")
    
    parent_dir = Path(__file__).parent.parent
    
    results = {
        "existing": [],
        "missing": []
    }
    
    for model_id, info in FUSION_MODELS.items():
        script_path = parent_dir / info['script']
        
        if script_path.exists():
            print(f"✓ {info['name']}")
            print(f"  File: {info['script']}")
            results["existing"].append(model_id)
        else:
            print(f"✗ {info['name']}")
            print(f"  File: {info['script']} (NOT FOUND)")
            results["missing"].append(model_id)
        print()
    
    print("-" * 80)
    print(f"Existing: {len(results['existing'])}/{len(FUSION_MODELS)}")
    print(f"Missing:  {len(results['missing'])}/{len(FUSION_MODELS)}")
    print("="*80)
    
    return results


def compare_speaker_embeddings():
    """Compare speaker embedding performance from eval_diarization.py results"""
    print("\n" + "="*80)
    print("SPEAKER EMBEDDING COMPARISON")
    print("="*80)
    print("\nBased on evaluation results from eval_diarization.py:")
    print("-" * 80)
    
    # Data from result.log
    embeddings_performance = {
        "SpeechBrain ECAPA-TDNN": {
            "eer": 15.57,
            "auc": 0.9353,
            "best_f1": 86.38,
            "used_by": ["Whisper+SpeechBrain", "SenseVoice+SpeechBrain"]
        },
        "NeMo TitaNet Large": {
            "eer": 14.89,
            "auc": 0.9403,
            "best_f1": 87.02,
            "used_by": ["Whisper+NeMo", "SenseVoice+NeMo"]
        },
        "PyAnnote WeSpeaker-ResNet34": {
            "eer": None,
            "auc": None,
            "best_f1": None,
            "used_by": ["Whisper+PyAnnote", "SenseVoice+PyAnnote"],
            "note": "Run eval_diarization.py to get results"
        }
    }
    
    for embedding_name, perf in embeddings_performance.items():
        print(f"\n{embedding_name}")
        print(f"  Used by: {', '.join(perf['used_by'])}")
        
        if perf['eer'] is not None:
            print(f"  EER:     {perf['eer']:.2f}%")
            print(f"  AUC:     {perf['auc']:.4f}")
            print(f"  Best F1: {perf['best_f1']:.2f}%")
        else:
            print(f"  Status:  {perf.get('note', 'Not evaluated')}")
    
    print("\n" + "-" * 80)
    print("CONCLUSION:")
    print("  - NeMo TitaNet Large has the best speaker verification performance")
    print("  - SpeechBrain ECAPA-TDNN is close behind and widely tested")
    print("  - PyAnnote results pending (run eval_diarization.py)")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Fusion Models Evaluation Summary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Print summary of all fusion models
  python eval_fusion_models.py --summary
  
  # Check which implementations exist
  python eval_fusion_models.py --check
  
  # Compare speaker embedding performance
  python eval_fusion_models.py --compare
  
  # Generate comprehensive report
  python eval_fusion_models.py --report
  
  # All above
  python eval_fusion_models.py --all
        """
    )
    
    parser.add_argument("--summary", action="store_true",
                       help="Print summary of fusion models")
    parser.add_argument("--check", action="store_true",
                       help="Check which implementations exist")
    parser.add_argument("--compare", action="store_true",
                       help="Compare speaker embedding performance")
    parser.add_argument("--report", action="store_true",
                       help="Generate comprehensive summary report")
    parser.add_argument("--all", action="store_true",
                       help="Run all checks and generate report")
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any([args.summary, args.check, args.compare, args.report, args.all]):
        parser.print_help()
        return
    
    # Run requested operations
    if args.all or args.summary:
        print_summary()
    
    if args.all or args.check:
        check_implementations()
    
    if args.all or args.compare:
        compare_speaker_embeddings()
    
    if args.all or args.report:
        report_path = generate_summary_report()
        print(f"\n📄 Read the full report at: {report_path}")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. To evaluate PyAnnote speaker embeddings:")
    print("   cd evaluation")
    print("   python eval_diarization.py --dataset <path_to_jvs>")
    print("\n2. To test fusion models on single audio files:")
    print("   python whisper_pyannote.py --audio_file <audio.wav>")
    print("   python sensevoice_nemo.py --audio_file <audio.wav>")
    print("\n3. To evaluate ASR performance:")
    print("   - Requires ground truth transcripts from dataset")
    print("   - Create custom evaluation script with WER/CER metrics")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
