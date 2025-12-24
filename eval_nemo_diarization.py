"""
Comprehensive Evaluation of NeMo Diarization Models

This script evaluates and compares finetuned vs pretrained NeMo diarization models.

Models evaluated:
1. Pretrained NeMo (baseline: VAD + Speaker pretrained)
2. Finetuned VAD (finetuned VAD + pretrained Speaker)

Metrics computed:
- DER (Diarization Error Rate)
- JER (Jaccard Error Rate)
- FA (False Alarm Rate)
- Miss (Missed Speech Rate)
- Precision, Recall, F1

Test datasets:
- voxconverse_test
- callhome_jpn
- callhome_eng (30%)

Usage:
    modal run eval_nemo_diarization.py
"""

import modal
import os
from pathlib import Path
from datetime import datetime

app = modal.App("eval-nemo-diarization")

# Modal image with NeMo
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
    .pip_install(
        "pip==23.3.2",
        "setuptools==69.0.3",
        "wheel==0.42.0",
        "Cython==3.0.8",
    )
    .pip_install("numpy==1.24.3")
    .pip_install(
        "pyarrow==14.0.1",
        "datasets==2.16.1",
    )
    .pip_install(
        "torch==2.1.0",
        "torchaudio==2.1.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "huggingface-hub==0.20.3",
        "transformers==4.36.2",
        "tokenizers==0.15.0",
    )
    .pip_install(
        "pyannote.audio==2.1.1",
        "pyannote.metrics==3.2.1",
        "pyannote.core==4.5",
    )
    .pip_install(
        "pytorch-lightning==2.1.0",
        "torchmetrics==1.2.1",
    )
    .pip_install(
        "soundfile==0.12.1",
        "librosa==0.10.1",
        "scikit-learn==1.3.2",
    )
    .pip_install(
        "matplotlib==3.8.2",
        "seaborn==0.13.0",
        "pandas==2.1.4",
    )
    .pip_install(
        "tqdm==4.66.1",
        "scipy==1.11.4",
    )
    .pip_install(
        "nemo_toolkit[asr]==1.23.0",
    )
    .pip_install(
        "webdataset==0.2.86",
        "braceexpand==0.1.7",
    )
)

# Volumes
dataset_volume = modal.Volume.from_name("nemo-dataset", create_if_missing=True)
results_volume = modal.Volume.from_name("nemo-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,  # 2 hours
    volumes={
        "/dataset": dataset_volume,
        "/results": results_volume,
    },
    memory=32768,
    cpu=8.0,
)
def evaluate_all_models():
    """Evaluate finetuned VAD model vs pretrained baseline on all test datasets"""
    import json
    import numpy as np
    import pandas as pd
    import torch
    import torchaudio
    from tqdm import tqdm
    from pathlib import Path
    
    # NeMo imports
    from nemo.collections.asr.models import EncDecClassificationModel, EncDecSpeakerLabelModel
    
    # Pyannote imports
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
    
    from sklearn.metrics import precision_recall_fscore_support
    from scipy.spatial.distance import cdist
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("=" * 80)
    print("🔍 EVALUATING NeMo DIARIZATION MODELS")
    print("=" * 80)
    print()
    
    datasets_to_eval = ['voxconverse_test', 'callhome_jpn', 'callhome_eng_test']
    print(f"Test datasets: {', '.join(datasets_to_eval)}")
    print()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    print()
    
    # === LOAD MODELS ===
    print("📦 Loading models...")
    models = {}
    
    # 1. Pretrained baseline
    print("   [1/2] Loading pretrained NeMo pipeline...")
    try:
        models['pretrained'] = {
            'name': 'Pretrained NeMo (titanet_large)',
            'vad': EncDecClassificationModel.from_pretrained("vad_multilingual_marblenet"),
            'speaker': EncDecSpeakerLabelModel.from_pretrained("titanet_large"),
        }
        models['pretrained']['vad'].to(device)
        models['pretrained']['speaker'].to(device)
        models['pretrained']['vad'].eval()
        models['pretrained']['speaker'].eval()
        print("      ✓ Loaded pretrained models")
    except Exception as e:
        print(f"      ❌ Error: {e}")
        raise
    
    # 2. Finetuned VAD
    print("   [2/2] Loading finetuned VAD model...")
    try:
        vad_checkpoint = Path("/results/checkpoints/vad/best_vad_model.nemo")
        if vad_checkpoint.exists():
            models['finetuned_vad'] = {
                'name': 'Finetuned VAD + titanet_large',
                'vad': EncDecClassificationModel.restore_from(str(vad_checkpoint)),
                'speaker': EncDecSpeakerLabelModel.from_pretrained("titanet_large"),
            }
            models['finetuned_vad']['vad'].to(device)
            models['finetuned_vad']['speaker'].to(device)
            models['finetuned_vad']['vad'].eval()
            models['finetuned_vad']['speaker'].eval()
            print(f"      ✓ Loaded finetuned VAD from {vad_checkpoint.name}")
        else:
            print(f"      ⚠️  Checkpoint not found: {vad_checkpoint}")
            print("      Will only evaluate pretrained model")
    except Exception as e:
        print(f"      ⚠️  Error loading finetuned VAD: {e}")
    
    print()
    print(f"✅ Loaded {len(models)} model(s)")
    print()
    
    # === PREPARE TEST DATASETS ===
    print("📊 Preparing test datasets...")
    test_datasets = {}
    
    for dataset_name in datasets_to_eval:
        if dataset_name == 'voxconverse_test':
            dataset_path = Path("/dataset/test/voxconverse_test")
        elif dataset_name == 'callhome_jpn':
            dataset_path = Path("/dataset/test/callhome_jpn")
        elif dataset_name == 'callhome_eng_test':
            dataset_path = Path("/dataset/test/callhome_eng")
        else:
            continue
        
        if dataset_path.exists():
            audio_files = list((dataset_path / "audio").glob("*.wav"))
            test_datasets[dataset_name] = {
                'path': dataset_path,
                'audio_files': audio_files,
                'num_files': len(audio_files),
            }
            print(f"   ✓ {dataset_name}: {len(audio_files)} files")
        else:
            print(f"   ⚠️  {dataset_name}: not found")
    
    print()
    
    # === EVALUATION LOOP ===
    print("🚀 Starting evaluation...")
    print("=" * 80)
    print()
    
    all_results = {}
    
    for model_key, model_dict in models.items():
        print(f"\n{'=' * 80}")
        print(f"EVALUATING: {model_dict['name'].upper()}")
        print(f"{'=' * 80}\n")
        
        model_results = {}
        
        for dataset_name, dataset_info in test_datasets.items():
            print(f"\n  📊 Dataset: {dataset_name}")
            print(f"  {'─' * 76}")
            
            dataset_results = evaluate_on_dataset(
                vad_model=model_dict['vad'],
                speaker_model=model_dict['speaker'],
                dataset_info=dataset_info,
                device=device,
            )
            
            model_results[dataset_name] = dataset_results
            
            # Print summary
            print(f"\n  Results:")
            print(f"    DER:        {dataset_results['metrics']['DER']:.2%}")
            print(f"    JER:        {dataset_results['metrics']['JER']:.2%}")
            print(f"    FA Rate:    {dataset_results['metrics']['FA_rate']:.2%}")
            print(f"    Miss Rate:  {dataset_results['metrics']['Miss_rate']:.2%}")
            print(f"    Precision:  {dataset_results['metrics']['Precision']:.2%}")
            print(f"    Recall:     {dataset_results['metrics']['Recall']:.2%}")
            print(f"    F1:         {dataset_results['metrics']['F1']:.2%}")
            print()
        
        all_results[model_key] = {
            'name': model_dict['name'],
            'results': model_results,
        }
    
    print("=" * 80)
    print("✅ EVALUATION COMPLETED")
    print("=" * 80)
    print()
    
    # === GENERATE COMPARISON TABLE ===
    print("📊 Generating comparison table...")
    comparison_table = generate_comparison_table(all_results)
    print()
    print(comparison_table)
    print()
    
    # === SAVE RESULTS ===
    print("💾 Saving results...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"/results/evaluation_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    results_file = results_dir / "detailed_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"   ✓ {results_file.name}")
    
    # Save comparison table
    table_file = results_dir / "comparison_table.csv"
    comparison_table.to_csv(table_file, index=False)
    print(f"   ✓ {table_file.name}")
    
    # Generate plots
    print("\n📈 Generating plots...")
    generate_comparison_plots(all_results, results_dir)
    
    # Commit volume
    results_volume.commit()
    print("\n✅ Results committed to volume")
    print()
    
    return {
        'results': all_results,
        'comparison_table': comparison_table.to_dict(),
        'results_dir': str(results_dir),
    }


def evaluate_on_dataset(vad_model, speaker_model, dataset_info, device):
    """Evaluate a single model on a single dataset"""
    import torch
    import torchaudio
    from tqdm import tqdm
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
    import numpy as np
    
    # Initialize metrics
    der_metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)
    jer_metric = JaccardErrorRate(collar=0.25)
    
    file_results = []
    
    total_fa = 0.0
    total_miss = 0.0
    total_speech_time = 0.0
    total_correct = 0.0
    total_hyp_speech_time = 0.0
    total_duration = 0.0
    
    for audio_file in tqdm(dataset_info['audio_files'], desc="    Processing", leave=False):
        try:
            # Get reference RTTM
            reference = load_rttm(audio_file, dataset_info['path'])
            if reference is None:
                continue
            
            # Get audio duration
            import torchaudio
            waveform, sample_rate = torchaudio.load(str(audio_file))
            audio_duration = waveform.shape[1] / sample_rate
            
            # Run NeMo diarization
            hypothesis = run_nemo_diarization(
                audio_file,
                vad_model,
                speaker_model,
                device
            )
            
            if hypothesis is None:
                continue
            
            # Compute DER and JER with UEM
            uem = Timeline([Segment(0, audio_duration)])
            der_score = float(der_metric(reference, hypothesis, uem=uem))
            jer_score = float(jer_metric(reference, hypothesis, uem=uem))
            
            # Compute FA and Miss
            ref_timeline = reference.get_timeline()
            hyp_timeline = hypothesis.get_timeline()
            
            # Calculate time components
            speech_time = ref_timeline.duration()  # Total reference speech
            hyp_speech_time = hyp_timeline.duration()  # Total hypothesis speech
            
            # Correct detection: intersection of reference and hypothesis
            correct_time = hyp_timeline.crop(ref_timeline).duration()
            
            # False Alarm: hypothesis speech outside reference speech
            fa_time = hyp_timeline.extrude(ref_timeline).duration()
            
            # Miss: reference speech not detected by hypothesis
            miss_time = ref_timeline.extrude(hyp_timeline).duration()

            total_speech_time += speech_time
            total_hyp_speech_time += hyp_speech_time
            total_fa += fa_time
            total_miss += miss_time
            total_correct += correct_time
            total_duration += audio_duration
            
            file_results.append({
                'file': audio_file.name,
                'DER': der_score,
                'JER': jer_score,
                'FA': fa_time,
                'Miss': miss_time,
                'Correct': correct_time,
                'speech_duration': speech_time,
            })
            
        except Exception as e:
            print(f"\n      ⚠️  Error processing {audio_file.name}: {e}")
            continue
    
    if len(file_results) == 0:
        return {
            'metrics': {
                'DER': 1.0, 'JER': 1.0, 'FA_rate': 1.0, 'Miss_rate': 1.0,
                'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0
            },
            'per_file': [],
            'num_files': 0,
        }
    
    # Aggregate metrics
    avg_der = np.mean([r['DER'] for r in file_results])
    avg_jer = np.mean([r['JER'] for r in file_results])
    
    # FA Rate: False Alarm relative to total non-speech time
    # Or more commonly: FA / total_duration
    total_non_speech = total_duration - total_speech_time
    fa_rate = total_fa / total_non_speech if total_non_speech > 0 else 0.0
    
    # Miss Rate: Missed Speech relative to total reference speech
    miss_rate = total_miss / total_speech_time if total_speech_time > 0 else 0.0
    
    # Compute Precision, Recall, F1
    # Precision: How much of detected speech is correct
    precision = total_correct / total_hyp_speech_time if total_hyp_speech_time > 0 else 0.0
    
    # Recall: How much of reference speech is detected
    recall = total_correct / total_speech_time if total_speech_time > 0 else 0.0
    
    # F1 Score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'metrics': {
            'DER': float(avg_der),
            'JER': float(avg_jer),
            'FA_rate': float(fa_rate),
            'Miss_rate': float(miss_rate),
            'Precision': float(precision),
            'Recall': float(recall),
            'F1': float(f1),
        },
        'per_file': file_results,
        'num_files': len(file_results),
    }


def load_rttm(audio_file, dataset_path):
    """Load RTTM reference annotation"""
    from pyannote.core import Annotation, Segment
    from pathlib import Path
    
    # Find RTTM file
    rttm_dir = dataset_path / "rttm"
    if not rttm_dir.exists():
        rttm_dir = dataset_path / "labels"
    
    rttm_file = rttm_dir / f"{audio_file.stem}.rttm"
    if not rttm_file.exists():
        file_id = audio_file.stem.replace("audio_", "")
        rttm_file = rttm_dir / f"labels_{file_id}.rttm"
    
    if not rttm_file.exists():
        return None
    
    # Parse RTTM
    annotation = Annotation()
    with open(rttm_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            
            annotation[Segment(start, start + duration)] = speaker
    
    return annotation


def run_nemo_diarization(audio_file, vad_model, speaker_model, device):
    """Run NeMo diarization pipeline"""
    import torch
    import torchaudio
    from pyannote.core import Annotation, Segment
    from sklearn.cluster import SpectralClustering
    from scipy.spatial.distance import cdist
    import numpy as np
    
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(str(audio_file))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        duration = waveform.shape[1] / sample_rate
        
        # Extract embeddings with sliding window
        window_size = 1.5
        hop_size = 0.75
        
        embeddings = []
        timestamps = []
        
        num_samples_window = int(window_size * sample_rate)
        num_samples_hop = int(hop_size * sample_rate)
        
        with torch.no_grad():
            for start_sample in range(0, waveform.shape[1] - num_samples_window, num_samples_hop):
                window = waveform[:, start_sample:start_sample + num_samples_window]
                window = window.to(device)
                
                # Get speaker embedding
                _, emb = speaker_model(
                    input_signal=window,
                    input_signal_length=torch.tensor([num_samples_window]).to(device)
                )
                
                embeddings.append(emb.cpu().numpy().flatten())
                timestamps.append(start_sample / sample_rate + window_size / 2)
        
        if len(embeddings) == 0:
            return None
        
        embeddings = np.array(embeddings)
        timestamps = np.array(timestamps)
        
        # Clustering
        num_speakers = min(8, len(embeddings) // 10)
        num_speakers = max(2, num_speakers)
        
        affinity = 1 - cdist(embeddings, embeddings, metric='cosine')
        affinity = np.maximum(affinity, 0)
        
        clustering = SpectralClustering(
            n_clusters=num_speakers,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=42
        )
        labels = clustering.fit_predict(affinity)
        
        # Build hypothesis
        hypothesis = Annotation()
        current_speaker = None
        segment_start = None
        
        for i, (timestamp, label) in enumerate(zip(timestamps, labels)):
            if current_speaker is None:
                current_speaker = label
                segment_start = timestamp - window_size / 2
            elif label != current_speaker:
                segment_end = timestamp - window_size / 2
                hypothesis[Segment(segment_start, segment_end)] = f"speaker_{int(current_speaker)}"
                current_speaker = label
                segment_start = segment_end
        
        # Add last segment
        if current_speaker is not None:
            hypothesis[Segment(segment_start, duration)] = f"speaker_{int(current_speaker)}"
        
        return hypothesis
        
    except Exception as e:
        print(f"      ⚠️  NeMo diarization error: {e}")
        return None


def generate_comparison_table(all_results):
    """Generate comparison table across all models and datasets"""
    import pandas as pd
    
    rows = []
    
    for model_key, model_data in all_results.items():
        model_name = model_data['name']
        for dataset_name, dataset_results in model_data['results'].items():
            metrics = dataset_results['metrics']
            
            row = {
                'Model': model_name,
                'Dataset': dataset_name,
                'DER (%)': f"{metrics['DER'] * 100:.2f}",
                'JER (%)': f"{metrics['JER'] * 100:.2f}",
                'FA (%)': f"{metrics['FA_rate'] * 100:.2f}",
                'Miss (%)': f"{metrics['Miss_rate'] * 100:.2f}",
                'Precision (%)': f"{metrics['Precision'] * 100:.2f}",
                'Recall (%)': f"{metrics['Recall'] * 100:.2f}",
                'F1 (%)': f"{metrics['F1'] * 100:.2f}",
                'Files': dataset_results['num_files'],
            }
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def generate_comparison_plots(all_results, output_dir):
    """Generate comparison plots"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    
    # Prepare data for plotting
    plot_data = []
    
    for model_key, model_data in all_results.items():
        model_name = model_data['name']
        for dataset_name, dataset_results in model_data['results'].items():
            metrics = dataset_results['metrics']
            
            plot_data.append({
                'Model': model_name,
                'Dataset': dataset_name,
                'DER': metrics['DER'] * 100,
                'JER': metrics['JER'] * 100,
                'FA': metrics['FA_rate'] * 100,
                'Miss': metrics['Miss_rate'] * 100,
                'Precision': metrics['Precision'] * 100,
                'Recall': metrics['Recall'] * 100,
                'F1': metrics['F1'] * 100,
            })
    
    df = pd.DataFrame(plot_data)
    
    # Plot 1: DER comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Dataset', y='DER', hue='Model')
    plt.title('Diarization Error Rate (DER) Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('DER (%)', fontsize=12)
    plt.xlabel('Dataset', fontsize=12)
    plt.legend(title='Model', fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "der_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: All main metrics
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Comprehensive Metrics Comparison', fontsize=16, fontweight='bold')
    
    metrics_to_plot = ['DER', 'JER', 'FA', 'Miss', 'Precision', 'F1']
    titles = ['DER (%)', 'JER (%)', 'False Alarm (%)', 'Miss Rate (%)', 'Precision (%)', 'F1 Score (%)']
    
    for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        ax = axes[idx // 3, idx % 3]
        sns.barplot(data=df, x='Dataset', y=metric, hue='Model', ax=ax)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel('%', fontsize=10)
        ax.set_xlabel('')
        ax.grid(axis='y', alpha=0.3)
        
        if idx != 1:
            ax.get_legend().remove()
        else:
            ax.legend(title='Model', fontsize=9, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / "all_metrics_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Improvement heatmap (if finetuned exists)
    if len(df['Model'].unique()) > 1:
        pivot_data = {}
        for dataset in df['Dataset'].unique():
            pretrained_der = df[(df['Model'] == 'Pretrained NeMo') & (df['Dataset'] == dataset)]['DER'].values
            finetuned_der = df[(df['Model'] == 'Finetuned VAD') & (df['Dataset'] == dataset)]['DER'].values
            
            if len(pretrained_der) > 0 and len(finetuned_der) > 0:
                improvement = ((pretrained_der[0] - finetuned_der[0]) / pretrained_der[0]) * 100
                pivot_data[dataset] = improvement
        
        if pivot_data:
            plt.figure(figsize=(10, 4))
            datasets = list(pivot_data.keys())
            improvements = list(pivot_data.values())
            colors = ['green' if x > 0 else 'red' for x in improvements]
            
            plt.barh(datasets, improvements, color=colors, alpha=0.7)
            plt.xlabel('DER Improvement (%)', fontsize=12)
            plt.title('VAD Finetuning Impact: DER Improvement vs Pretrained', fontsize=14, fontweight='bold')
            plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
            plt.grid(axis='x', alpha=0.3)
            
            for i, v in enumerate(improvements):
                plt.text(v, i, f' {v:+.1f}%', va='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(output_dir / "improvement_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("   ✓ improvement_comparison.png")
    
    print("   ✓ der_comparison.png")
    print("   ✓ all_metrics_comparison.png")


@app.local_entrypoint()
def main():
    """Main entry point - Evaluate finetuned VAD vs pretrained baseline"""
    
    print("\n" + "=" * 80)
    print("🚀 STARTING NeMo DIARIZATION EVALUATION")
    print("=" * 80)
    print()
    print("Comparing:")
    print("  • Pretrained NeMo (baseline)")
    print("  • Finetuned VAD (improved)")
    print()
    print("Test datasets:")
    print("  • voxconverse_test (English)")
    print("  • callhome_jpn (Japanese)")
    print("  • callhome_eng (English, 30% test split)")
    print()
    
    result = evaluate_all_models.remote()
    
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETED")
    print("=" * 80)
    print(f"\n📂 Results saved to: {result['results_dir']}")
    print()
    
    # Print summary table
    print("=" * 80)
    print("📊 DETAILED RESULTS")
    print("=" * 80)
    print()
    import pandas as pd
    df = pd.DataFrame(result['comparison_table'])
    print(df.to_string(index=False))
    print()
    print("=" * 80)
    print()
    
    # Calculate overall improvement
    try:
        pretrained_avg = df[df['Model'] == 'Pretrained NeMo']['DER (%)'].apply(lambda x: float(x.replace('%', ''))).mean()
        finetuned_avg = df[df['Model'] == 'Finetuned VAD']['DER (%)'].apply(lambda x: float(x.replace('%', ''))).mean()
        improvement = ((pretrained_avg - finetuned_avg) / pretrained_avg) * 100
        
        print(f"📈 Overall DER Improvement: {improvement:+.2f}%")
        print(f"   Pretrained avg: {pretrained_avg:.2f}%")
        print(f"   Finetuned avg:  {finetuned_avg:.2f}%")
        print()
    except:
        pass
    
    print("✅ Check Modal Storage for detailed results and plots!")
    print()
