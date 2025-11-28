"""
Comparison script for Speaker Diarization evaluation results
So sánh kết quả đánh giá diarization của 3 model:
- Whisper + SpeechBrain
- SenseVoice  
- SenseVoice + SpeechBrain

Metrics compared: DER, JER, Coverage, Purity, F1-score, RTF
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

# Set font for Vietnamese support
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Color palette for models
COLORS = {
    'whisper-speechbrain': '#FF6B6B',
    'sensevoice': '#4ECDC4', 
    'sensevoice-speechbrain': '#45B7D1'
}

MODEL_NAMES = {
    'whisper-speechbrain': 'Whisper + SpeechBrain',
    'sensevoice': 'SenseVoice',
    'sensevoice-speechbrain': 'SenseVoice + SpeechBrain'
}

def find_latest_results(results_dir):
    """Find the latest diarization evaluation results"""
    results_dir = Path(results_dir)
    
    # Look for individual summary files from eval_diarization.py
    result_files = list(results_dir.glob("eval_*_summary.json"))
    
    if not result_files:
        # Look for other possible formats
        result_files = list(results_dir.glob("diarization_eval_results_*.json"))
        if not result_files:
            result_files = list(results_dir.glob("diarization_eval_summary_*.csv"))
    
    if not result_files:
        return None
    
    # Get the latest file
    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    return latest_file

def load_diarization_results(results_dir):
    """Load diarization evaluation results from individual summary files"""
    results_dir = Path(results_dir)
    
    # Look for individual model summary files
    summary_files = list(results_dir.glob("eval_*_summary.json"))
    
    if not summary_files:
        print("No individual summary files found")
        return {}
    
    results_data = {}
    
    for summary_file in summary_files:
        try:
            # Extract model name from filename
            model_name = summary_file.stem.replace('eval_', '').replace('_summary', '')
            
            with open(summary_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to expected format
            results_data[model_name] = data
            print(f"✓ Loaded {model_name}: {data.get('num_files', 0)} files")
            
        except Exception as e:
            print(f"Error loading {summary_file}: {e}")
            continue
    
    return results_data

def create_summary_table(results_data):
    """Create summary comparison table"""
    print("\n" + "="*80)
    print("DIARIZATION MODEL COMPARISON SUMMARY")
    print("="*80)
    
    # Create DataFrame
    rows = []
    for model_key, data in results_data.items():
        # Handle direct data format from eval_diarization.py
        model_name = MODEL_NAMES.get(model_key, data.get('model', model_key))
        
        rows.append({
            'Model': model_name,
            'Files': data.get('num_files', 0),
            'DER (%)': f"{data.get('avg_der', 0)*100:.2f}",
            'JER (%)': f"{(1 - data.get('avg_f1', 0))*100:.2f}",  # JER = 1 - F1
            'Coverage (%)': f"{data.get('avg_coverage', 0)*100:.2f}",
            'Purity (%)': f"{data.get('avg_purity', 0)*100:.2f}",
            'F1 (%)': f"{data.get('avg_f1', 0)*100:.2f}",
            'Avg RTF': f"{data.get('avg_rtf', 0):.3f}",
            'Real-time': '✓' if data.get('avg_rtf', 0) < 1.0 else '✗'
        })
    
    if not rows:
        print("No summary data available")
        return None
    
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    
    # Performance ranking
    print("\n" + "-"*80)
    print("PERFORMANCE RANKING")
    print("-"*80)
    
    # Sort by DER (lower is better)
    der_ranking = sorted(results_data.items(), 
                        key=lambda x: x[1].get('avg_der', float('inf')))
    print("DER (Diarization Error Rate) - Lower is better:")
    for i, (model_key, data) in enumerate(der_ranking, 1):
        der = data.get('avg_der', 0)
        model_name = MODEL_NAMES.get(model_key, data.get('model', model_key))
        print(f"  {i}. {model_name}: {der*100:.2f}%")
    
    # Sort by F1 (higher is better)  
    f1_ranking = sorted(results_data.items(), 
                       key=lambda x: x[1].get('avg_f1', 0), 
                       reverse=True)
    print("\nF1-Score - Higher is better:")
    for i, (model_key, data) in enumerate(f1_ranking, 1):
        f1 = data.get('avg_f1', 0)
        model_name = MODEL_NAMES.get(model_key, data.get('model', model_key))
        print(f"  {i}. {model_name}: {f1*100:.2f}%")
    
    # Sort by RTF (lower is better for real-time)
    rtf_ranking = sorted(results_data.items(), 
                        key=lambda x: x[1].get('avg_rtf', float('inf')))
    print("\nRTF (Real-time Factor) - Lower is better for real-time:")
    for i, (model_key, data) in enumerate(rtf_ranking, 1):
        rtf = data.get('avg_rtf', 0)
        real_time = "✓ Real-time" if rtf < 1.0 else "✗ Not real-time"
        model_name = MODEL_NAMES.get(model_key, data.get('model', model_key))
        print(f"  {i}. {model_name}: {rtf:.3f} ({real_time})")
    
    print("="*80 + "\n")
    return df

def plot_comparison_metrics(results_data, output_dir):
    """Plot comparison metrics"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    models = list(results_data.keys())
    model_labels = [MODEL_NAMES.get(m, results_data[m].get('model', m)) for m in models]
    
    # Extract metrics from direct data format
    der_values = [results_data[m].get('avg_der', 0) * 100 for m in models]
    jer_values = [(1 - results_data[m].get('avg_f1', 0)) * 100 for m in models]  # JER = 1 - F1
    coverage_values = [results_data[m].get('avg_coverage', 0) * 100 for m in models]
    purity_values = [results_data[m].get('avg_purity', 0) * 100 for m in models]
    f1_values = [results_data[m].get('avg_f1', 0) * 100 for m in models]
    rtf_values = [results_data[m].get('avg_rtf', 0) for m in models]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Speaker Diarization Model Comparison', fontsize=16, fontweight='bold')
    
    # DER comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(model_labels, der_values, color=[COLORS.get(m, '#666666') for m in models])
    ax1.set_title('Diarization Error Rate (DER)', fontweight='bold')
    ax1.set_ylabel('DER (%)')
    ax1.set_ylim(0, max(der_values) * 1.1 if der_values else 1)
    
    # Add value labels
    for bar, val in zip(bars1, der_values):
        height = bar.get_height()
        ax1.annotate(f'{val:.2f}%', 
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    # JER comparison  
    ax2 = axes[0, 1]
    bars2 = ax2.bar(model_labels, jer_values, color=[COLORS.get(m, '#666666') for m in models])
    ax2.set_title('Jaccard Error Rate (JER)', fontweight='bold')
    ax2.set_ylabel('JER (%)')
    ax2.set_ylim(0, max(jer_values) * 1.1 if jer_values else 1)
    
    # Add value labels
    for bar, val in zip(bars2, jer_values):
        height = bar.get_height()
        ax2.annotate(f'{val:.2f}%', 
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    # F1-Score comparison
    ax3 = axes[0, 2]
    bars3 = ax3.bar(model_labels, f1_values, color=[COLORS.get(m, '#666666') for m in models])
    ax3.set_title('F1-Score', fontweight='bold')
    ax3.set_ylabel('F1-Score (%)')
    ax3.set_ylim(0, max(f1_values) * 1.1 if f1_values else 1)
    
    # Add value labels
    for bar, val in zip(bars3, f1_values):
        height = bar.get_height()
        ax3.annotate(f'{val:.2f}%', 
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    # Coverage vs Purity scatter plot
    ax4 = axes[1, 0]
    for i, model in enumerate(models):
        ax4.scatter(coverage_values[i], purity_values[i], 
                   s=200, alpha=0.7,
                   color=COLORS.get(model, '#666666'),
                   label=model_labels[i])
        ax4.annotate(model_labels[i], 
                    (coverage_values[i], purity_values[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')
    
    ax4.set_title('Coverage vs Purity', fontweight='bold')
    ax4.set_xlabel('Coverage (%)')
    ax4.set_ylabel('Purity (%)')
    ax4.grid(alpha=0.3)
    ax4.set_xlim(0, 100)
    ax4.set_ylim(0, 100)
    
    # RTF comparison
    ax5 = axes[1, 1]
    bars5 = ax5.bar(model_labels, rtf_values, color=[COLORS.get(m, '#666666') for m in models])
    ax5.set_title('Real-time Factor (RTF)', fontweight='bold')
    ax5.set_ylabel('RTF')
    ax5.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Real-time threshold')
    ax5.legend()
    ax5.set_ylim(0, max(rtf_values) * 1.1 if rtf_values else 1)
    
    # Add value labels
    for bar, val in zip(bars5, rtf_values):
        height = bar.get_height()
        ax5.annotate(f'{val:.3f}', 
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    # Overall performance radar chart
    ax6 = axes[1, 2]
    
    if models:
        # Normalize metrics for radar chart (higher is better)
        metrics_names = ['DER', 'JER', 'Coverage', 'Purity', 'F1', 'RTF']
        
        # For radar chart, invert error rates (DER, JER) and RTF so higher is better
        normalized_data = []
        for i, model in enumerate(models):
            normalized = [
                100 - der_values[i],  # Invert DER
                100 - jer_values[i],  # Invert JER  
                coverage_values[i],   # Coverage as-is
                purity_values[i],     # Purity as-is
                f1_values[i],         # F1 as-is
                100 - min(rtf_values[i] * 50, 100)  # Invert and scale RTF
            ]
            normalized_data.append(normalized)
        
        # Simple bar chart instead of radar for simplicity
        x_pos = np.arange(len(metrics_names))
        bar_width = 0.25
        
        for i, (model, data) in enumerate(zip(models, normalized_data)):
            ax6.bar(x_pos + i * bar_width, data, bar_width, 
                   alpha=0.7, color=COLORS.get(model, '#666666'),
                   label=MODEL_NAMES.get(model, model))
        
        ax6.set_title('Overall Performance Comparison', fontweight='bold')
        ax6.set_xlabel('Metrics (Normalized to 0-100)')
        ax6.set_ylabel('Score (Higher = Better)')
        ax6.set_xticks(x_pos + bar_width * (len(models) - 1) / 2)
        ax6.set_xticklabels(metrics_names, rotation=45)
        ax6.legend()
        ax6.grid(axis='y', alpha=0.3)
        ax6.set_ylim(0, 100)
    
    # Rotate x-axis labels for better readability
    for ax in axes.flat:
        if ax != ax4 and ax != ax6:  # Skip scatter plot and performance chart
            ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / 'diarization_comparison.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {plot_file}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Compare Diarization evaluation results')
    parser.add_argument('--results_dir', type=str, 
                       default='eval_results',
                       help='Directory containing evaluation results')
    parser.add_argument('--results_file', type=str,
                       help='Specific results file to analyze')
    parser.add_argument('--output_dir', type=str,
                       default='comparison_output', 
                       help='Directory to save comparison results')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data directly from results directory
    print(f"Loading results from: {results_dir}")
    print(f"Output will be saved to: {output_dir}")
    
    # Load data
    try:
        results_data = load_diarization_results(results_dir)
        if not results_data:
            print("No data found in results directory!")
            print("Please run diarization evaluation first:")
            print("  python eval_diarization.py")
            return
        
        print(f"Loaded results for {len(results_data)} models")
        
        # Create comparison table
        summary_df = create_summary_table(results_data)
        
        # Generate plots
        if not args.no_plots and summary_df is not None:
            plot_comparison_metrics(results_data, output_dir)
        
        # Save summary
        if summary_df is not None:
            summary_file = output_dir / 'diarization_comparison_summary.csv'
            summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
            print(f"✓ Saved comparison summary: {summary_file}")
        
        print("\n✓ Comparison completed successfully!")
        
    except Exception as e:
        print(f"Error processing results: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()