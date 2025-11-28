"""
Comparison script for ASR evaluation results
So sánh kết quả đánh giá ASR của 3 model:
- Whisper Small
- SenseVoice  
- SenseVoice + SpeechBrain

Metrics compared: WER, CER, RTF (Real-time Factor)
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

# Set font for Vietnamese support
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Color palette for models
COLORS = {
    'whisper-small': '#FF6B6B',
    'sensevoice': '#4ECDC4', 
    'sensevoice-speechbrain': '#45B7D1'
}

MODEL_NAMES = {
    'whisper-small': 'Whisper Small',
    'sensevoice': 'SenseVoice',
    'sensevoice-speechbrain': 'SenseVoice + SpeechBrain'
}

def load_summary_data(results_dir):
    """Load summary data from all models"""
    results_dir = Path(results_dir)
    summary_data = {}
    
    # Find all summary JSON files
    for json_file in results_dir.glob("*_summary.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                model_name = data['model']
                summary_data[model_name] = data
                print(f"✓ Loaded {model_name}: {data['num_samples']} samples")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return summary_data

def load_detailed_data(results_dir):
    """Load detailed data from CSV files"""
    results_dir = Path(results_dir)
    detailed_data = {}
    
    # Find all checkpoint CSV files
    for csv_file in results_dir.glob("*_checkpoint.csv"):
        try:
            model_name = csv_file.stem.replace('eval_', '').replace('_checkpoint', '')
            df = pd.read_csv(csv_file)
            detailed_data[model_name] = df
            print(f"✓ Loaded detailed data for {model_name}: {len(df)} samples")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    return detailed_data

def create_summary_table(summary_data):
    """Create summary comparison table"""
    print("\n" + "="*80)
    print("ASR MODEL COMPARISON SUMMARY")
    print("="*80)
    
    # Create DataFrame
    rows = []
    for model, data in summary_data.items():
        rows.append({
            'Model': MODEL_NAMES.get(model, model),
            'Samples': data['num_samples'],
            'WER (%)': f"{data['avg_wer']*100:.2f}",
            'CER (%)': f"{data['avg_cer']*100:.2f}",
            'Avg RTF': f"{data['avg_rtf']:.3f}",
            'Median RTF': f"{data['median_rtf']:.3f}",
            'Min RTF': f"{data['min_rtf']:.3f}",
            'Max RTF': f"{data['max_rtf']:.3f}"
        })
    
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    
    # Performance ranking
    print("\n" + "-"*80)
    print("PERFORMANCE RANKING")
    print("-"*80)
    
    # Sort by WER (lower is better)
    wer_ranking = sorted(summary_data.items(), key=lambda x: x[1]['avg_wer'])
    print("WER (Word Error Rate) - Lower is better:")
    for i, (model, data) in enumerate(wer_ranking, 1):
        print(f"  {i}. {MODEL_NAMES.get(model, model)}: {data['avg_wer']*100:.2f}%")
    
    # Sort by CER (lower is better)  
    cer_ranking = sorted(summary_data.items(), key=lambda x: x[1]['avg_cer'])
    print("\nCER (Character Error Rate) - Lower is better:")
    for i, (model, data) in enumerate(cer_ranking, 1):
        print(f"  {i}. {MODEL_NAMES.get(model, model)}: {data['avg_cer']*100:.2f}%")
    
    # Sort by RTF (lower is better for real-time)
    rtf_ranking = sorted(summary_data.items(), key=lambda x: x[1]['avg_rtf'])
    print("\nRTF (Real-time Factor) - Lower is better for real-time:")
    for i, (model, data) in enumerate(rtf_ranking, 1):
        real_time = "✓ Real-time" if data['avg_rtf'] < 1.0 else "✗ Not real-time"
        print(f"  {i}. {MODEL_NAMES.get(model, model)}: {data['avg_rtf']:.3f} ({real_time})")
    
    print("="*80 + "\n")
    return df

def plot_summary_metrics(summary_data, output_dir):
    """Plot summary metrics comparison"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    models = list(summary_data.keys())
    model_labels = [MODEL_NAMES.get(m, m) for m in models]
    
    # Extract metrics
    wer_values = [summary_data[m]['avg_wer'] * 100 for m in models]
    cer_values = [summary_data[m]['avg_cer'] * 100 for m in models]  
    rtf_values = [summary_data[m]['avg_rtf'] for m in models]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ASR Model Comparison', fontsize=16, fontweight='bold')
    
    # WER comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(model_labels, wer_values, color=[COLORS.get(m, '#666666') for m in models])
    ax1.set_title('Word Error Rate (WER)', fontweight='bold')
    ax1.set_ylabel('WER (%)')
    ax1.set_ylim(0, max(wer_values) * 1.1)
    
    # Add value labels on bars
    for bar, val in zip(bars1, wer_values):
        height = bar.get_height()
        ax1.annotate(f'{val:.2f}%', 
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    # CER comparison  
    ax2 = axes[0, 1]
    bars2 = ax2.bar(model_labels, cer_values, color=[COLORS.get(m, '#666666') for m in models])
    ax2.set_title('Character Error Rate (CER)', fontweight='bold')
    ax2.set_ylabel('CER (%)')
    ax2.set_ylim(0, max(cer_values) * 1.1)
    
    # Add value labels on bars
    for bar, val in zip(bars2, cer_values):
        height = bar.get_height()
        ax2.annotate(f'{val:.2f}%', 
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    # RTF comparison
    ax3 = axes[1, 0]
    bars3 = ax3.bar(model_labels, rtf_values, color=[COLORS.get(m, '#666666') for m in models])
    ax3.set_title('Real-time Factor (RTF)', fontweight='bold')
    ax3.set_ylabel('RTF')
    ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Real-time threshold')
    ax3.legend()
    ax3.set_ylim(0, max(rtf_values) * 1.1)
    
    # Add value labels on bars
    for bar, val in zip(bars3, rtf_values):
        height = bar.get_height()
        ax3.annotate(f'{val:.3f}', 
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    # Overall score (weighted combination)
    # Lower WER + Lower CER + Lower RTF = Better
    ax4 = axes[1, 1]
    
    # Normalize metrics (0-1 scale, inverted for "lower is better")
    max_wer = max(wer_values)
    max_cer = max(cer_values) 
    max_rtf = max(rtf_values)
    
    normalized_scores = []
    for i, model in enumerate(models):
        wer_score = 1 - (wer_values[i] / max_wer)  # Inverted
        cer_score = 1 - (cer_values[i] / max_cer)  # Inverted
        rtf_score = 1 - (rtf_values[i] / max_rtf)  # Inverted
        
        # Weighted average (WER=40%, CER=30%, RTF=30%)
        overall_score = (wer_score * 0.4 + cer_score * 0.3 + rtf_score * 0.3) * 100
        normalized_scores.append(overall_score)
    
    bars4 = ax4.bar(model_labels, normalized_scores, color=[COLORS.get(m, '#666666') for m in models])
    ax4.set_title('Overall Performance Score', fontweight='bold')
    ax4.set_ylabel('Score (Higher is better)')
    ax4.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, val in zip(bars4, normalized_scores):
        height = bar.get_height()
        ax4.annotate(f'{val:.1f}', 
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    # Rotate x-axis labels for better readability
    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / 'asr_comparison_summary.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved summary plot: {plot_file}")
    
    plt.show()

def plot_detailed_distributions(detailed_data, output_dir):
    """Plot detailed metric distributions"""
    if not detailed_data:
        print("No detailed data available for distribution plots")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Distribution of ASR Metrics Across Test Samples', fontsize=16, fontweight='bold')
    
    # Prepare data for plotting
    all_models = list(detailed_data.keys())
    
    # WER distribution
    ax1 = axes[0, 0]
    wer_data = []
    labels = []
    for model in all_models:
        df = detailed_data[model]
        wer_data.append(df['wer'] * 100)  # Convert to percentage
        labels.append(MODEL_NAMES.get(model, model))
    
    ax1.boxplot(wer_data, labels=labels, patch_artist=True)
    ax1.set_title('WER Distribution (%)', fontweight='bold')
    ax1.set_ylabel('WER (%)')
    ax1.grid(axis='y', alpha=0.3)
    
    # CER distribution
    ax2 = axes[0, 1]  
    cer_data = []
    for model in all_models:
        df = detailed_data[model]
        cer_data.append(df['cer'] * 100)  # Convert to percentage
    
    ax2.boxplot(cer_data, labels=labels, patch_artist=True)
    ax2.set_title('CER Distribution (%)', fontweight='bold') 
    ax2.set_ylabel('CER (%)')
    ax2.grid(axis='y', alpha=0.3)
    
    # RTF distribution
    ax3 = axes[1, 0]
    rtf_data = []
    for model in all_models:
        df = detailed_data[model]
        rtf_data.append(df['rtf'])
    
    ax3.boxplot(rtf_data, labels=labels, patch_artist=True)
    ax3.set_title('RTF Distribution', fontweight='bold')
    ax3.set_ylabel('RTF')
    ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Real-time threshold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Processing time vs Audio duration
    ax4 = axes[1, 1]
    for i, model in enumerate(all_models):
        df = detailed_data[model]
        ax4.scatter(df['audio_duration'], df['processing_time'], 
                   alpha=0.6, label=MODEL_NAMES.get(model, model), 
                   color=list(COLORS.values())[i % len(COLORS)])
    
    ax4.set_title('Processing Time vs Audio Duration', fontweight='bold')
    ax4.set_xlabel('Audio Duration (s)')
    ax4.set_ylabel('Processing Time (s)')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # Add real-time line (y=x)
    max_duration = max([detailed_data[m]['audio_duration'].max() for m in all_models])
    ax4.plot([0, max_duration], [0, max_duration], 'r--', alpha=0.7, label='Real-time line')
    
    # Color boxes in boxplots
    colors = [list(COLORS.values())[i % len(COLORS)] for i in range(len(all_models))]
    
    for ax, data in [(ax1, wer_data), (ax2, cer_data), (ax3, rtf_data)]:
        boxes = ax.findobj(plt.matplotlib.patches.PathPatch)
        for box, color in zip(boxes, colors):
            box.set_facecolor(color)
            box.set_alpha(0.7)
    
    # Rotate x-axis labels
    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / 'asr_detailed_distributions.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved distribution plot: {plot_file}")
    
    plt.show()

def create_detailed_statistics(detailed_data, output_dir):
    """Create detailed statistics table"""
    if not detailed_data:
        print("No detailed data available for statistics")
        return
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*80)
    print("DETAILED STATISTICS")
    print("="*80)
    
    stats_data = []
    
    for model, df in detailed_data.items():
        model_name = MODEL_NAMES.get(model, model)
        
        # Calculate statistics
        stats = {
            'Model': model_name,
            'Samples': len(df),
            'WER Mean (%)': f"{df['wer'].mean()*100:.2f}",
            'WER Std (%)': f"{df['wer'].std()*100:.2f}",
            'WER Median (%)': f"{df['wer'].median()*100:.2f}",
            'CER Mean (%)': f"{df['cer'].mean()*100:.2f}",
            'CER Std (%)': f"{df['cer'].std()*100:.2f}", 
            'CER Median (%)': f"{df['cer'].median()*100:.2f}",
            'RTF Mean': f"{df['rtf'].mean():.3f}",
            'RTF Std': f"{df['rtf'].std():.3f}",
            'RTF Median': f"{df['rtf'].median():.3f}",
            'Realtime %': f"{(df['rtf'] < 1.0).mean()*100:.1f}%"
        }
        stats_data.append(stats)
    
    stats_df = pd.DataFrame(stats_data)
    print(stats_df.to_string(index=False))
    
    # Save to CSV
    output_file = output_dir / 'detailed_statistics.csv'
    stats_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ Saved detailed statistics: {output_file}")

def save_comparison_results(summary_data, detailed_data, output_dir):
    """Save comprehensive comparison results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create comprehensive report
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'models_compared': len(summary_data),
        'summary': summary_data,
        'comparison': {
            'best_wer': min(summary_data.items(), key=lambda x: x[1]['avg_wer'])[0],
            'best_cer': min(summary_data.items(), key=lambda x: x[1]['avg_cer'])[0],
            'best_rtf': min(summary_data.items(), key=lambda x: x[1]['avg_rtf'])[0],
            'realtime_capable': [model for model, data in summary_data.items() 
                                if data['avg_rtf'] < 1.0]
        }
    }
    
    # Save JSON report
    report_file = output_dir / 'comparison_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved comparison report: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Compare ASR evaluation results')
    parser.add_argument('--results_dir', type=str, 
                       default='eval_results',
                       help='Directory containing evaluation results')
    parser.add_argument('--output_dir', type=str,
                       default='comparison_output', 
                       help='Directory to save comparison results')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist")
        return
    
    print(f"Loading evaluation results from: {results_dir}")
    print(f"Output will be saved to: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    summary_data = load_summary_data(results_dir)
    if not summary_data:
        print("No summary data found!")
        return
    
    detailed_data = load_detailed_data(results_dir)
    
    # Create comparison table
    summary_df = create_summary_table(summary_data)
    
    # Create detailed statistics
    if detailed_data:
        create_detailed_statistics(detailed_data, output_dir)
    
    # Generate plots
    if not args.no_plots:
        plot_summary_metrics(summary_data, output_dir)
        if detailed_data:
            plot_detailed_distributions(detailed_data, output_dir)
    
    # Save results
    save_comparison_results(summary_data, detailed_data, output_dir)
    
    print("\n✓ Comparison completed successfully!")

if __name__ == "__main__":
    main()