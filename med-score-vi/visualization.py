"""
Visualization module for MedScore Small LLM results comparison.
"""

import json
import jsonlines
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import os


class MedScoreVisualizer:
    """
    Comprehensive visualization class for MedScore results.
    
    Creates various plots to compare small LLM performance against baseline:
    1. Score distribution comparisons
    2. Correlation plots
    3. Performance metrics charts
    4. Error analysis visualizations
    """
    
    def __init__(self, baseline_file: str, small_llm_file: str, output_dir: str = "visualizations"):
        """
        Initialize visualizer with result files.
        
        Args:
            baseline_file: Path to baseline results
            small_llm_file: Path to small LLM results
            output_dir: Directory to save visualizations
        """
        self.baseline_file = baseline_file
        self.small_llm_file = small_llm_file
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        self.baseline_data = self._load_data(baseline_file)
        self.small_llm_data = self._load_data(small_llm_file)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def _load_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from JSONL file"""
        data = []
        with jsonlines.open(file_path) as reader:
            for item in reader.iter():
                data.append(item)
        return data
    
    def plot_score_distribution(self, save: bool = True) -> None:
        """Plot score distribution comparison"""
        baseline_scores = [r['score'] for r in self.baseline_data if r['score'] is not None]
        small_llm_scores = [r['score'] for r in self.small_llm_data if r['score'] is not None]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram comparison
        ax1.hist(baseline_scores, bins=20, alpha=0.7, label='Baseline', color='skyblue', edgecolor='black')
        ax1.hist(small_llm_scores, bins=20, alpha=0.7, label='Small LLM', color='lightcoral', edgecolor='black')
        ax1.set_xlabel('MedScore')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Score Distribution Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot comparison
        data_for_box = [baseline_scores, small_llm_scores]
        labels = ['Baseline', 'Small LLM']
        box_plot = ax2.boxplot(data_for_box, labels=labels, patch_artist=True)
        box_plot['boxes'][0].set_facecolor('skyblue')
        box_plot['boxes'][1].set_facecolor('lightcoral')
        ax2.set_ylabel('MedScore')
        ax2.set_title('Score Distribution Box Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'score_distribution.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_analysis(self, save: bool = True) -> None:
        """Plot correlation between baseline and small LLM scores"""
        # Create mapping by response ID
        baseline_by_id = {r['id']: r['score'] for r in self.baseline_data if r['score'] is not None}
        small_llm_by_id = {r['id']: r['score'] for r in self.small_llm_data if r['score'] is not None}
        
        common_ids = set(baseline_by_id.keys()) & set(small_llm_by_id.keys())
        
        if not common_ids:
            print("No common response IDs found for correlation analysis")
            return
        
        baseline_scores = [baseline_by_id[id] for id in common_ids]
        small_llm_scores = [small_llm_by_id[id] for id in common_ids]
        
        # Calculate correlation
        correlation = np.corrcoef(baseline_scores, small_llm_scores)[0, 1]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot
        ax.scatter(baseline_scores, small_llm_scores, alpha=0.6, s=50)
        
        # Add diagonal line
        min_score = min(min(baseline_scores), min(small_llm_scores))
        max_score = max(max(baseline_scores), max(small_llm_scores))
        ax.plot([min_score, max_score], [min_score, max_score], 'r--', alpha=0.8, label='Perfect Agreement')
        
        # Add trend line
        z = np.polyfit(baseline_scores, small_llm_scores, 1)
        p = np.poly1d(z)
        ax.plot(baseline_scores, p(baseline_scores), "g--", alpha=0.8, label=f'Trend Line (r={correlation:.3f})')
        
        ax.set_xlabel('Baseline MedScore')
        ax.set_ylabel('Small LLM MedScore')
        ax.set_title(f'Correlation Analysis (r = {correlation:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add text box with statistics
        textstr = f'Correlation: {correlation:.3f}\nResponses: {len(common_ids)}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'correlation_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_metrics(self, save: bool = True) -> None:
        """Plot comprehensive performance metrics"""
        # Calculate metrics
        baseline_scores = [r['score'] for r in self.baseline_data if r['score'] is not None]
        small_llm_scores = [r['score'] for r in self.small_llm_data if r['score'] is not None]
        
        metrics_data = {
            'Model': ['Baseline', 'Small LLM'],
            'Mean Score': [np.mean(baseline_scores), np.mean(small_llm_scores)],
            'Median Score': [np.median(baseline_scores), np.median(small_llm_scores)],
            'Std Dev': [np.std(baseline_scores), np.std(small_llm_scores)],
            'Min Score': [np.min(baseline_scores), np.min(small_llm_scores)],
            'Max Score': [np.max(baseline_scores), np.max(small_llm_scores)]
        }
        
        df = pd.DataFrame(metrics_data)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        metrics = ['Mean Score', 'Median Score', 'Std Dev', 'Min Score', 'Max Score']
        colors = ['skyblue', 'lightcoral']
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            bars = ax.bar(df['Model'], df[metric], color=colors)
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            
            # Add value labels on bars
            for bar, value in zip(bars, df[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        # Add improvement percentage
        improvement = ((df.loc[1, 'Mean Score'] - df.loc[0, 'Mean Score']) / df.loc[0, 'Mean Score']) * 100
        axes[5].text(0.5, 0.5, f'Relative Improvement:\n{improvement:.1f}%', 
                    ha='center', va='center', fontsize=16, 
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        axes[5].set_title('Performance Improvement')
        axes[5].axis('off')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_claim_analysis(self, save: bool = True) -> None:
        """Plot claim-level analysis"""
        # Extract claims
        baseline_claims = []
        small_llm_claims = []
        
        for result in self.baseline_data:
            for claim in result.get('claims', []):
                baseline_claims.append(claim['score'])
        
        for result in self.small_llm_data:
            for claim in result.get('claims', []):
                small_llm_claims.append(claim['score'])
        
        # Calculate claim statistics
        baseline_true = sum(1 for score in baseline_claims if score == 1.0)
        baseline_false = sum(1 for score in baseline_claims if score == 0.0)
        small_llm_true = sum(1 for score in small_llm_claims if score == 1.0)
        small_llm_false = sum(1 for score in small_llm_claims if score == 0.0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Claim accuracy comparison
        models = ['Baseline', 'Small LLM']
        accuracies = [
            baseline_true / len(baseline_claims) if baseline_claims else 0,
            small_llm_true / len(small_llm_claims) if small_llm_claims else 0
        ]
        
        bars1 = ax1.bar(models, accuracies, color=['skyblue', 'lightcoral'])
        ax1.set_ylabel('Claim Accuracy')
        ax1.set_title('Claim-Level Accuracy Comparison')
        ax1.set_ylim(0, 1)
        
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Claim distribution
        claim_data = {
            'Model': ['Baseline', 'Baseline', 'Small LLM', 'Small LLM'],
            'Claim Type': ['True', 'False', 'True', 'False'],
            'Count': [baseline_true, baseline_false, small_llm_true, small_llm_false]
        }
        
        df_claims = pd.DataFrame(claim_data)
        pivot_df = df_claims.pivot(index='Model', columns='Claim Type', values='Count')
        
        pivot_df.plot(kind='bar', ax=ax2, color=['lightgreen', 'lightcoral'])
        ax2.set_ylabel('Number of Claims')
        ax2.set_title('Claim Distribution by Type')
        ax2.legend(title='Claim Type')
        ax2.tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'claim_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_error_analysis(self, save: bool = True) -> None:
        """Plot error analysis and difference distribution"""
        # Create mapping by response ID
        baseline_by_id = {r['id']: r['score'] for r in self.baseline_data if r['score'] is not None}
        small_llm_by_id = {r['id']: r['score'] for r in self.small_llm_data if r['score'] is not None}
        
        common_ids = set(baseline_by_id.keys()) & set(small_llm_by_id.keys())
        
        if not common_ids:
            print("No common response IDs found for error analysis")
            return
        
        differences = []
        for id in common_ids:
            diff = small_llm_by_id[id] - baseline_by_id[id]
            differences.append(diff)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Difference distribution
        ax1.hist(differences, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='No Difference')
        ax1.axvline(x=np.mean(differences), color='green', linestyle='-', alpha=0.8, 
                   label=f'Mean Difference: {np.mean(differences):.3f}')
        ax1.set_xlabel('Score Difference (Small LLM - Baseline)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Score Difference Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Absolute difference
        abs_differences = [abs(d) for d in differences]
        ax2.hist(abs_differences, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.axvline(x=np.mean(abs_differences), color='red', linestyle='-', alpha=0.8,
                   label=f'Mean Absolute Difference: {np.mean(abs_differences):.3f}')
        ax2.set_xlabel('Absolute Score Difference')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Absolute Score Difference Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'error_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_dashboard(self, save: bool = True) -> None:
        """Create comprehensive dashboard with all visualizations"""
        print("Creating comprehensive MedScore visualization dashboard...")
        
        self.plot_score_distribution(save=save)
        self.plot_correlation_analysis(save=save)
        self.plot_performance_metrics(save=save)
        self.plot_claim_analysis(save=save)
        self.plot_error_analysis(save=save)
        
        print(f"All visualizations saved to {self.output_dir}/")
    
    def generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics for the results"""
        baseline_scores = [r['score'] for r in self.baseline_data if r['score'] is not None]
        small_llm_scores = [r['score'] for r in self.small_llm_data if r['score'] is not None]
        
        # Create mapping for correlation
        baseline_by_id = {r['id']: r['score'] for r in self.baseline_data if r['score'] is not None}
        small_llm_by_id = {r['id']: r['score'] for r in self.small_llm_data if r['score'] is not None}
        common_ids = set(baseline_by_id.keys()) & set(small_llm_by_id.keys())
        
        correlation = 0.0
        if common_ids:
            baseline_common = [baseline_by_id[id] for id in common_ids]
            small_llm_common = [small_llm_by_id[id] for id in common_ids]
            correlation = np.corrcoef(baseline_common, small_llm_common)[0, 1]
        
        return {
            'baseline': {
                'count': len(baseline_scores),
                'mean': np.mean(baseline_scores),
                'std': np.std(baseline_scores),
                'min': np.min(baseline_scores),
                'max': np.max(baseline_scores)
            },
            'small_llm': {
                'count': len(small_llm_scores),
                'mean': np.mean(small_llm_scores),
                'std': np.std(small_llm_scores),
                'min': np.min(small_llm_scores),
                'max': np.max(small_llm_scores)
            },
            'comparison': {
                'correlation': correlation,
                'mean_difference': np.mean(small_llm_scores) - np.mean(baseline_scores),
                'relative_improvement': ((np.mean(small_llm_scores) - np.mean(baseline_scores)) / np.mean(baseline_scores)) * 100 if np.mean(baseline_scores) > 0 else 0
            }
        }


def create_medscore_visualizations(baseline_file: str, small_llm_file: str, output_dir: str = "visualizations"):
    """
    Create comprehensive MedScore visualizations.
    
    Args:
        baseline_file: Path to baseline results
        small_llm_file: Path to small LLM results
        output_dir: Directory to save visualizations
    """
    visualizer = MedScoreVisualizer(baseline_file, small_llm_file, output_dir)
    visualizer.create_comprehensive_dashboard()
    
    # Generate and save summary statistics
    stats = visualizer.generate_summary_statistics()
    with open(os.path.join(output_dir, 'summary_statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nSummary Statistics:")
    print(f"Baseline Mean Score: {stats['baseline']['mean']:.4f}")
    print(f"Small LLM Mean Score: {stats['small_llm']['mean']:.4f}")
    print(f"Correlation: {stats['comparison']['correlation']:.4f}")
    print(f"Relative Improvement: {stats['comparison']['relative_improvement']:.1f}%")
    
    return visualizer


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python visualization.py <baseline_file> <small_llm_file> [output_dir]")
        sys.exit(1)
    
    baseline_file = sys.argv[1]
    small_llm_file = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "visualizations"
    
    create_medscore_visualizations(baseline_file, small_llm_file, output_dir)
