"""
Evaluation Script for Neural Diarizer on JVS Dataset
====================================================
Đánh giá hiệu suất của NeMo Neural Diarizer trên JVS corpus.

Dataset Structure:
jvs_ver1/
├── jvs001/
│   ├── parallel100/  (100 files - parallel speech)
│   ├── nonpara30/    (30 files - non-parallel speech)
│   ├── whisper10/    (10 files - whispered speech)
│   └── falset10/     (10 files - falsetto speech)
├── jvs002/
└── ... (100 speakers total)

Usage:
    # Evaluate on single speaker
    python eval_neural_NeMo.py --jvs-root dataset/jvs_ver1/jvs_ver1 --speaker jvs001
    
    # Evaluate on multiple speakers
    python eval_neural_NeMo.py --jvs-root dataset/jvs_ver1/jvs_ver1 --speakers jvs001 jvs002 jvs003
    
    # Evaluate on all speakers
    python eval_neural_NeMo.py --jvs-root dataset/jvs_ver1/jvs_ver1 --all-speakers
    
    # Evaluate specific category only
    python eval_neural_NeMo.py --jvs-root dataset/jvs_ver1/jvs_ver1 --speaker jvs001 --category parallel100
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict
import soundfile as sf
from tqdm import tqdm

from neural_diarizer import NeuralDiarizer


# Categories in JVS dataset
JVS_CATEGORIES = ["parallel100", "nonpara30", "whisper10", "falset10"]


class DiarizationEvaluator:
    """Evaluator for speaker diarization on JVS dataset"""
    
    def __init__(self, neural_diarizer: NeuralDiarizer, verbose: bool = True):
        """
        Initialize evaluator
        
        Parameters
        ----------
        neural_diarizer : NeuralDiarizer
            Neural diarizer instance
        verbose : bool
            Enable verbose output
        """
        self.diarizer = neural_diarizer
        self.verbose = verbose
    
    def evaluate_single_file(
        self,
        audio_path: Path,
        expected_speakers: int = 1
    ) -> Dict:
        """
        Evaluate diarization on single audio file
        
        Parameters
        ----------
        audio_path : Path
            Path to audio file
        expected_speakers : int
            Expected number of speakers (for JVS single-speaker: 1)
        
        Returns
        -------
        Dict
            Evaluation metrics
        """
        # Run diarization
        segments = self.diarizer.diarize_audio(audio_path, num_speakers=None)
        
        # Count detected speakers
        detected_speakers = len(set(seg['speaker'] for seg in segments))
        
        # Get audio duration
        audio_data, sample_rate = sf.read(str(audio_path))
        audio_duration = len(audio_data) / sample_rate
        
        # Calculate total speech time
        total_speech = sum(seg['end'] - seg['start'] for seg in segments)
        
        # Calculate speech ratio
        speech_ratio = total_speech / audio_duration if audio_duration > 0 else 0
        
        # Check if number of speakers is correct
        correct_speaker_count = (detected_speakers == expected_speakers)
        
        return {
            "audio_path": str(audio_path),
            "audio_name": audio_path.name,
            "expected_speakers": expected_speakers,
            "detected_speakers": detected_speakers,
            "correct_speaker_count": correct_speaker_count,
            "num_segments": len(segments),
            "audio_duration": audio_duration,
            "total_speech_time": total_speech,
            "speech_ratio": speech_ratio,
            "segments": segments
        }
    
    def evaluate_speaker(
        self,
        jvs_root: Path,
        speaker_id: str,
        categories: List[str] = None,
        max_files_per_category: int = None
    ) -> Dict:
        """
        Evaluate diarization on all files from a speaker
        
        Parameters
        ----------
        jvs_root : Path
            Root directory of JVS dataset
        speaker_id : str
            Speaker ID (e.g., "jvs001")
        categories : List[str], optional
            Categories to evaluate (default: all)
        max_files_per_category : int, optional
            Maximum number of files per category to evaluate
        
        Returns
        -------
        Dict
            Evaluation results for the speaker
        """
        speaker_dir = jvs_root / speaker_id
        
        if not speaker_dir.exists():
            raise ValueError(f"Speaker directory not found: {speaker_dir}")
        
        if categories is None:
            categories = JVS_CATEGORIES
        
        results = {
            "speaker_id": speaker_id,
            "categories": {},
            "total_files": 0,
            "total_correct": 0,
            "total_accuracy": 0.0
        }
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating Speaker: {speaker_id}")
            print(f"{'='*60}")
        
        for category in categories:
            category_dir = speaker_dir / category / "wav24kHz16bit"
            
            if not category_dir.exists():
                if self.verbose:
                    print(f"⚠️  Category not found: {category}")
                continue
            
            # Get all WAV files
            wav_files = sorted(list(category_dir.glob("*.wav")))
            
            if max_files_per_category:
                wav_files = wav_files[:max_files_per_category]
            
            if self.verbose:
                print(f"\n📁 Category: {category} ({len(wav_files)} files)")
            
            category_results = {
                "files": [],
                "total_files": len(wav_files),
                "correct_count": 0,
                "accuracy": 0.0
            }
            
            # Evaluate each file
            iterator = tqdm(wav_files, desc=f"  {category}") if self.verbose else wav_files
            
            for wav_file in iterator:
                try:
                    file_result = self.evaluate_single_file(wav_file, expected_speakers=1)
                    category_results["files"].append(file_result)
                    
                    if file_result["correct_speaker_count"]:
                        category_results["correct_count"] += 1
                    
                except Exception as e:
                    if self.verbose:
                        print(f"❌ Error processing {wav_file.name}: {e}")
            
            # Calculate category accuracy
            if category_results["total_files"] > 0:
                category_results["accuracy"] = (
                    category_results["correct_count"] / category_results["total_files"]
                )
            
            results["categories"][category] = category_results
            results["total_files"] += category_results["total_files"]
            results["total_correct"] += category_results["correct_count"]
            
            if self.verbose:
                print(f"  ✅ Accuracy: {category_results['accuracy']*100:.2f}% "
                      f"({category_results['correct_count']}/{category_results['total_files']})")
        
        # Calculate overall accuracy
        if results["total_files"] > 0:
            results["total_accuracy"] = results["total_correct"] / results["total_files"]
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Overall Accuracy: {results['total_accuracy']*100:.2f}% "
                  f"({results['total_correct']}/{results['total_files']})")
            print(f"{'='*60}\n")
        
        return results
    
    def evaluate_multiple_speakers(
        self,
        jvs_root: Path,
        speaker_ids: List[str],
        categories: List[str] = None,
        max_files_per_category: int = None
    ) -> Dict:
        """
        Evaluate diarization on multiple speakers
        
        Parameters
        ----------
        jvs_root : Path
            Root directory of JVS dataset
        speaker_ids : List[str]
            List of speaker IDs
        categories : List[str], optional
            Categories to evaluate
        max_files_per_category : int, optional
            Maximum files per category
        
        Returns
        -------
        Dict
            Aggregated evaluation results
        """
        all_results = {
            "speakers": {},
            "summary": {
                "total_speakers": len(speaker_ids),
                "total_files": 0,
                "total_correct": 0,
                "overall_accuracy": 0.0,
                "category_accuracy": {}
            }
        }
        
        if self.verbose:
            print(f"\n🎯 Evaluating {len(speaker_ids)} speakers...")
        
        # Evaluate each speaker
        for speaker_id in speaker_ids:
            try:
                speaker_results = self.evaluate_speaker(
                    jvs_root, speaker_id, categories, max_files_per_category
                )
                all_results["speakers"][speaker_id] = speaker_results
                
                # Update summary
                all_results["summary"]["total_files"] += speaker_results["total_files"]
                all_results["summary"]["total_correct"] += speaker_results["total_correct"]
                
            except Exception as e:
                if self.verbose:
                    print(f"❌ Error evaluating {speaker_id}: {e}")
        
        # Calculate overall accuracy
        if all_results["summary"]["total_files"] > 0:
            all_results["summary"]["overall_accuracy"] = (
                all_results["summary"]["total_correct"] / 
                all_results["summary"]["total_files"]
            )
        
        # Calculate per-category accuracy across all speakers
        category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for speaker_results in all_results["speakers"].values():
            for category, cat_results in speaker_results["categories"].items():
                category_stats[category]["correct"] += cat_results["correct_count"]
                category_stats[category]["total"] += cat_results["total_files"]
        
        for category, stats in category_stats.items():
            if stats["total"] > 0:
                accuracy = stats["correct"] / stats["total"]
                all_results["summary"]["category_accuracy"][category] = {
                    "accuracy": accuracy,
                    "correct": stats["correct"],
                    "total": stats["total"]
                }
        
        return all_results
    
    def save_results(self, results: Dict, output_path: Path):
        """Save evaluation results to JSON"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"💾 Results saved to: {output_path}")
    
    def print_summary(self, results: Dict):
        """Print evaluation summary"""
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        
        summary = results["summary"]
        
        print(f"\n📊 Overall Statistics:")
        print(f"   Total Speakers: {summary['total_speakers']}")
        print(f"   Total Files: {summary['total_files']}")
        print(f"   Correct Predictions: {summary['total_correct']}")
        print(f"   Overall Accuracy: {summary['overall_accuracy']*100:.2f}%")
        
        if summary["category_accuracy"]:
            print(f"\n📁 Per-Category Accuracy:")
            for category, stats in summary["category_accuracy"].items():
                print(f"   {category:15s}: {stats['accuracy']*100:.2f}% "
                      f"({stats['correct']}/{stats['total']})")
        
        print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Neural Diarizer on JVS Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--jvs-root", type=str, required=True,
                        help="Root directory of JVS dataset")
    parser.add_argument("--speaker", type=str,
                        help="Single speaker ID to evaluate (e.g., jvs001)")
    parser.add_argument("--speakers", type=str, nargs='+',
                        help="Multiple speaker IDs to evaluate")
    parser.add_argument("--all-speakers", action='store_true',
                        help="Evaluate all speakers in dataset")
    parser.add_argument("--category", type=str, choices=JVS_CATEGORIES,
                        help="Specific category to evaluate")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Maximum files per category")
    parser.add_argument("--config", type=str, default="diar_infer_config.yaml",
                        help="Path to diarization config")
    parser.add_argument("--output-dir", type=str, default="eval_output",
                        help="Output directory for diarization")
    parser.add_argument("--results-dir", type=str, default="eval_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--max-speakers", type=int, default=8,
                        help="Maximum number of speakers")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    parser.add_argument("--no-verbose", action='store_true',
                        help="Disable verbose output")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.speaker, args.speakers, args.all_speakers]):
        parser.error("Must specify --speaker, --speakers, or --all-speakers")
    
    jvs_root = Path(args.jvs_root)
    if not jvs_root.exists():
        parser.error(f"JVS root directory not found: {jvs_root}")
    
    # Determine device
    import torch
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    verbose = not args.no_verbose
    
    # Initialize diarizer
    print("🚀 Initializing Neural Diarizer...")
    config_path = args.config if os.path.exists(args.config) else None
    diarizer = NeuralDiarizer(
        config_path=config_path,
        output_dir=args.output_dir,
        device=device,
        max_num_speakers=args.max_speakers,
        verbose=verbose
    )
    
    # Initialize evaluator
    evaluator = DiarizationEvaluator(diarizer, verbose=verbose)
    
    # Determine speaker IDs to evaluate
    if args.all_speakers:
        # Get all speaker directories
        speaker_dirs = sorted([d for d in jvs_root.iterdir() 
                              if d.is_dir() and d.name.startswith('jvs')])
        speaker_ids = [d.name for d in speaker_dirs]
        print(f"📂 Found {len(speaker_ids)} speakers in dataset")
    elif args.speakers:
        speaker_ids = args.speakers
    else:
        speaker_ids = [args.speaker]
    
    # Determine categories
    categories = [args.category] if args.category else None
    
    # Run evaluation
    if len(speaker_ids) == 1:
        results = evaluator.evaluate_speaker(
            jvs_root,
            speaker_ids[0],
            categories=categories,
            max_files_per_category=args.max_files
        )
        # Wrap single speaker results
        results = {
            "speakers": {speaker_ids[0]: results},
            "summary": {
                "total_speakers": 1,
                "total_files": results["total_files"],
                "total_correct": results["total_correct"],
                "overall_accuracy": results["total_accuracy"],
                "category_accuracy": {}
            }
        }
    else:
        results = evaluator.evaluate_multiple_speakers(
            jvs_root,
            speaker_ids,
            categories=categories,
            max_files_per_category=args.max_files
        )
    
    # Print summary
    evaluator.print_summary(results)
    
    # Save results
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if len(speaker_ids) == 1:
        output_file = results_dir / f"eval_{speaker_ids[0]}.json"
    else:
        output_file = results_dir / f"eval_{len(speaker_ids)}_speakers.json"
    
    evaluator.save_results(results, output_file)
    
    print(f"✅ Evaluation complete!")
    print(f"   Results: {output_file}")
    print(f"   Diarization outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
