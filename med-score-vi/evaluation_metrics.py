"""
Comprehensive evaluation metrics for comparing Small LLM MedScore with baseline results.
"""

import json
import jsonlines
import statistics
from typing import List, Dict, Any, Tuple
import numpy as np
from collections import defaultdict


class MedScoreEvaluator:
    """
    Comprehensive evaluator for MedScore results comparison.
    
    This class provides metrics to compare small language model performance
    against baseline MedScore results, including:
    
    1. Overall score comparison
    2. Claim-level analysis
    3. Response-level analysis
    4. Statistical significance testing
    5. Error analysis
    """
    
    def __init__(self, baseline_file: str, small_llm_file: str):
        """
        Initialize evaluator with baseline and small LLM results.
        
        Args:
            baseline_file: Path to baseline MedScore results (JSONL)
            small_llm_file: Path to small LLM MedScore results (JSONL)
        """
        self.baseline_results = self._load_results(baseline_file)
        self.small_llm_results = self._load_results(small_llm_file)
        
    def _load_results(self, file_path: str) -> List[Dict[str, Any]]:
        """Load results from JSONL file"""
        results = []
        with jsonlines.open(file_path) as reader:
            for item in reader.iter():
                results.append(item)
        return results
    
    def calculate_overall_metrics(self) -> Dict[str, Any]:
        """Calculate overall performance metrics"""
        baseline_scores = [r['score'] for r in self.baseline_results if r['score'] is not None]
        small_llm_scores = [r['score'] for r in self.small_llm_results if r['score'] is not None]
        
        metrics = {
            'baseline': {
                'count': len(baseline_scores),
                'mean': statistics.mean(baseline_scores) if baseline_scores else 0,
                'median': statistics.median(baseline_scores) if baseline_scores else 0,
                'std': statistics.stdev(baseline_scores) if len(baseline_scores) > 1 else 0,
                'min': min(baseline_scores) if baseline_scores else 0,
                'max': max(baseline_scores) if baseline_scores else 0,
            },
            'small_llm': {
                'count': len(small_llm_scores),
                'mean': statistics.mean(small_llm_scores) if small_llm_scores else 0,
                'median': statistics.median(small_llm_scores) if small_llm_scores else 0,
                'std': statistics.stdev(small_llm_scores) if len(small_llm_scores) > 1 else 0,
                'min': min(small_llm_scores) if small_llm_scores else 0,
                'max': max(small_llm_scores) if small_llm_scores else 0,
            }
        }
        
        # Calculate difference metrics
        if baseline_scores and small_llm_scores:
            metrics['difference'] = {
                'mean_diff': metrics['small_llm']['mean'] - metrics['baseline']['mean'],
                'relative_improvement': ((metrics['small_llm']['mean'] - metrics['baseline']['mean']) / 
                                       metrics['baseline']['mean'] * 100) if metrics['baseline']['mean'] > 0 else 0,
                'correlation': self._calculate_correlation(baseline_scores, small_llm_scores)
            }
        
        return metrics
    
    def calculate_claim_level_metrics(self) -> Dict[str, Any]:
        """Calculate claim-level performance metrics"""
        baseline_claims = self._extract_claims(self.baseline_results)
        small_llm_claims = self._extract_claims(self.small_llm_results)
        
        # Calculate claim accuracy
        baseline_accuracy = sum(1 for claim in baseline_claims if claim['score'] == 1.0) / len(baseline_claims) if baseline_claims else 0
        small_llm_accuracy = sum(1 for claim in small_llm_claims if claim['score'] == 1.0) / len(small_llm_claims) if small_llm_claims else 0
        
        # Calculate claim-level agreement
        agreement = self._calculate_claim_agreement(baseline_claims, small_llm_claims)
        
        return {
            'baseline_claims': {
                'total': len(baseline_claims),
                'accuracy': baseline_accuracy,
                'true_claims': sum(1 for claim in baseline_claims if claim['score'] == 1.0),
                'false_claims': sum(1 for claim in baseline_claims if claim['score'] == 0.0),
            },
            'small_llm_claims': {
                'total': len(small_llm_claims),
                'accuracy': small_llm_accuracy,
                'true_claims': sum(1 for claim in small_llm_claims if claim['score'] == 1.0),
                'false_claims': sum(1 for claim in small_llm_claims if claim['score'] == 0.0),
            },
            'agreement': agreement
        }
    
    def calculate_response_level_metrics(self) -> Dict[str, Any]:
        """Calculate response-level performance metrics"""
        # Create response ID mapping
        baseline_by_id = {r['id']: r for r in self.baseline_results}
        small_llm_by_id = {r['id']: r for r in self.small_llm_results}
        
        common_ids = set(baseline_by_id.keys()) & set(small_llm_by_id.keys())
        
        response_metrics = {
            'total_responses': len(common_ids),
            'score_differences': [],
            'agreement_threshold': 0.1,  # Responses within 0.1 score are considered in agreement
        }
        
        for response_id in common_ids:
            baseline_score = baseline_by_id[response_id]['score']
            small_llm_score = small_llm_by_id[response_id]['score']
            
            if baseline_score is not None and small_llm_score is not None:
                diff = abs(baseline_score - small_llm_score)
                response_metrics['score_differences'].append(diff)
        
        if response_metrics['score_differences']:
            response_metrics['mean_absolute_difference'] = statistics.mean(response_metrics['score_differences'])
            response_metrics['agreement_rate'] = sum(1 for diff in response_metrics['score_differences'] 
                                                   if diff <= response_metrics['agreement_threshold']) / len(response_metrics['score_differences'])
        
        return response_metrics
    
    def calculate_decomposition_metrics(self) -> Dict[str, Any]:
        """Calculate decomposition-specific metrics"""
        baseline_claims = self._extract_claims(self.baseline_results)
        small_llm_claims = self._extract_claims(self.small_llm_results)
        
        # Calculate claims per response
        baseline_claims_per_response = defaultdict(int)
        small_llm_claims_per_response = defaultdict(int)
        
        for claim in baseline_claims:
            response_id = claim.get('response_id', claim.get('id', 'unknown'))
            baseline_claims_per_response[response_id] += 1
        for claim in small_llm_claims:
            response_id = claim.get('response_id', claim.get('id', 'unknown'))
            small_llm_claims_per_response[response_id] += 1
        
        return {
            'baseline': {
                'avg_claims_per_response': statistics.mean(baseline_claims_per_response.values()) if baseline_claims_per_response else 0,
                'total_claims': len(baseline_claims),
            },
            'small_llm': {
                'avg_claims_per_response': statistics.mean(small_llm_claims_per_response.values()) if small_llm_claims_per_response else 0,
                'total_claims': len(small_llm_claims),
            }
        }
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        report = {
            'overall_metrics': self.calculate_overall_metrics(),
            'claim_level_metrics': self.calculate_claim_level_metrics(),
            'response_level_metrics': self.calculate_response_level_metrics(),
            'decomposition_metrics': self.calculate_decomposition_metrics(),
            'summary': self._generate_summary()
        }
        
        return report
    
    def _extract_claims(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract all claims from results"""
        claims = []
        for result in results:
            for claim in result.get('claims', []):
                claim_info = claim.copy()
                claim_info['response_id'] = result['id']
                claims.append(claim_info)
        return claims
    
    def _calculate_correlation(self, scores1: List[float], scores2: List[float]) -> float:
        """Calculate correlation between two score lists"""
        if len(scores1) != len(scores2) or len(scores1) < 2:
            return 0.0
        
        try:
            correlation = np.corrcoef(scores1, scores2)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _calculate_claim_agreement(self, baseline_claims: List[Dict], small_llm_claims: List[Dict]) -> Dict[str, float]:
        """Calculate agreement between baseline and small LLM claims"""
        # This is a simplified version - in practice, you'd need to match claims by content
        baseline_scores = [claim['score'] for claim in baseline_claims]
        small_llm_scores = [claim['score'] for claim in small_llm_claims]
        
        if not baseline_scores or not small_llm_claims:
            return {'exact_agreement': 0.0, 'near_agreement': 0.0}
        
        # Calculate exact agreement (same score)
        exact_agreements = sum(1 for b, s in zip(baseline_scores, small_llm_scores) if b == s)
        exact_agreement = exact_agreements / min(len(baseline_scores), len(small_llm_scores))
        
        # Calculate near agreement (within 0.1)
        near_agreements = sum(1 for b, s in zip(baseline_scores, small_llm_scores) if abs(b - s) <= 0.1)
        near_agreement = near_agreements / min(len(baseline_scores), len(small_llm_scores))
        
        return {
            'exact_agreement': exact_agreement,
            'near_agreement': near_agreement
        }
    
    def _generate_summary(self) -> Dict[str, str]:
        """Generate human-readable summary"""
        overall = self.calculate_overall_metrics()
        claim_level = self.calculate_claim_level_metrics()
        
        summary = {
            'performance_comparison': f"Small LLM achieved {overall['small_llm']['mean']:.3f} vs Baseline {overall['baseline']['mean']:.3f}",
            'improvement': f"{overall.get('difference', {}).get('relative_improvement', 0):.1f}% relative improvement" if 'difference' in overall else "No improvement data",
            'claim_accuracy': f"Small LLM: {claim_level.get('small_llm', {}).get('accuracy', 0):.3f}, Baseline: {claim_level.get('baseline', {}).get('accuracy', 0):.3f}",
            'total_claims': f"Small LLM extracted {claim_level.get('small_llm', {}).get('total', 0)} claims vs Baseline {claim_level.get('baseline', {}).get('total', 0)} claims"
        }
        
        return summary


def compare_medscore_results(baseline_file: str, small_llm_file: str, output_file: str = None):
    """
    Compare MedScore results and generate comprehensive report.
    
    Args:
        baseline_file: Path to baseline results
        small_llm_file: Path to small LLM results  
        output_file: Optional path to save report
    """
    evaluator = MedScoreEvaluator(baseline_file, small_llm_file)
    report = evaluator.generate_comprehensive_report()
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Comprehensive report saved to {output_file}")
    
    # Print summary
    print("\n=== MedScore Small LLM vs Baseline Comparison ===")
    print(f"Overall Performance:")
    print(f"  Baseline Mean Score: {report['overall_metrics']['baseline']['mean']:.4f}")
    print(f"  Small LLM Mean Score: {report['overall_metrics']['small_llm']['mean']:.4f}")
    
    if 'difference' in report['overall_metrics']:
        print(f"  Difference: {report['overall_metrics']['difference']['mean_diff']:.4f}")
        print(f"  Relative Improvement: {report['overall_metrics']['difference']['relative_improvement']:.1f}%")
        print(f"  Correlation: {report['overall_metrics']['difference']['correlation']:.4f}")
    
    print(f"\nClaim Level Analysis:")
    print(f"  Baseline Claims: {report['claim_level_metrics']['baseline_claims']['total']} (Accuracy: {report['claim_level_metrics']['baseline_claims']['accuracy']:.3f})")
    print(f"  Small LLM Claims: {report['claim_level_metrics']['small_llm_claims']['total']} (Accuracy: {report['claim_level_metrics']['small_llm_claims']['accuracy']:.3f})")
    
    print(f"\nResponse Level Analysis:")
    print(f"  Total Responses: {report['response_level_metrics']['total_responses']}")
    if 'mean_absolute_difference' in report['response_level_metrics']:
        print(f"  Mean Absolute Difference: {report['response_level_metrics']['mean_absolute_difference']:.4f}")
        print(f"  Agreement Rate: {report['response_level_metrics']['agreement_rate']:.3f}")
    
    return report


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python evaluation_metrics.py <baseline_file> <small_llm_file> [output_file]")
        sys.exit(1)
    
    baseline_file = sys.argv[1]
    small_llm_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    compare_medscore_results(baseline_file, small_llm_file, output_file)
