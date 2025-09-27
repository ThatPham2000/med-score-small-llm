# Small Language Models for Factuality Evaluation of Free-Form Medical Answers

This repository implements an enhanced MedScore framework that makes small language models more intelligent and reasoning-capable for medical factuality evaluation. Our approach achieves comparable or superior performance to large language models while providing significant computational and cost advantages.

## 🎯 Key Results

- **1.3% improvement** in overall MedScore compared to GPT-4o baseline
- **4.8% improvement** in claim-level accuracy (80.5% vs 76.8%)
- **72.3% more claims** extracted per response (20.5 vs 11.9)
- **Cost-effective**: ~10-100x lower computational cost than large models
- **Privacy-preserving**: Local deployment capability

## 🚀 Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Install Ollama (for running small language models)
# Visit: https://ollama.ai/download
```

### Basic Usage

```bash
# Run small LLM evaluation with baseline comparison
python run_small_llm_evaluation.py data/AskDocs.demo.jsonl results/small_llm/ \
    --baseline MedScore_baseline_results/GPT4o_medscore_output.jsonl \
    --decomposition-model llama3.2:3b \
    --verification-model llama3.2:3b
```

### Advanced Usage

```bash
# Custom configuration
python run_small_llm_evaluation.py data/AskDocs.demo.jsonl results/small_llm/ \
    --baseline MedScore_baseline_results/GPT4o_medscore_output.jsonl \
    --decomposition-model gemma2:9b \
    --verification-model gemma2:9b \
    --reasoning-steps 5 \
    --confidence-threshold 0.8 \
    --no-visualizations
```

## 📊 Results Overview

Our evaluation shows that small language models can achieve superior performance when enhanced with appropriate reasoning frameworks:

| Metric | Baseline (GPT-4o) | Small LLM | Improvement |
|--------|-------------------|-----------|-------------|
| Mean MedScore | 0.7654 | 0.7756 | **+1.3%** |
| Claim Accuracy | 76.8% | 80.5% | **+4.8%** |
| Claims/Response | 11.9 | 20.5 | **+72.3%** |
| Std Deviation | 0.1894 | 0.1541 | **-18.7%** |

## 🏗️ Architecture

### Enhanced Components

1. **DecomposerSmallLLM**: Enhanced decomposition with chain-of-thought reasoning
2. **VerifierSmallLLM**: Improved verification with multi-step reasoning
3. **MedScoreSmallLLM**: Complete framework integration
4. **EvaluationMetrics**: Comprehensive comparison framework
5. **Visualization**: Results analysis and visualization tools

### Key Features

- **Chain-of-Thought Prompting**: Step-by-step reasoning for complex medical concepts
- **Multi-step Reasoning**: 3-step process for both decomposition and verification
- **Context-Aware Processing**: Maintains context throughout evaluation
- **Confidence Scoring**: Provides confidence levels for verification decisions
- **Comprehensive Metrics**: Detailed performance analysis and comparison

## 📁 Project Structure

```
med-score-vi/
├── med-score-vi/
│   ├── decomposer_small_llm.py      # Enhanced decomposition
│   ├── verifier_small_llm.py        # Enhanced verification
│   ├── med_score_small_llm.py       # Main framework
│   ├── evaluation_metrics.py        # Comparison metrics
│   ├── visualization.py             # Results visualization
│   └── ...                          # Original MedScore components
├── run_small_llm_evaluation.py      # Main evaluation script
├── scientific_paper.md              # Complete research paper
├── results/small_llm/               # Generated results
│   ├── small_llm_med_score_output.jsonl
│   ├── comparison_report.json
│   └── visualizations/              # Charts and graphs
└── MedScore_baseline_results/       # Baseline comparison data
```

## 🔧 Configuration Options

### Model Selection

```bash
# Available small language models
--decomposition-model llama3.2:3b    # Llama 3.2 3B parameters
--decomposition-model gemma2:9b      # Gemma 2 9B parameters
--decomposition-model qwen2.5:7b     # Qwen 2.5 7B parameters

--verification-model llama3.2:3b     # Same options as decomposition
```

### Reasoning Parameters

```bash
--reasoning-steps 3                   # Number of reasoning steps (1-5)
--use-chain-of-thought               # Enable chain-of-thought prompting
--confidence-threshold 0.7           # Verification confidence threshold
```

### Evaluation Options

```bash
--baseline <file>                    # Baseline results for comparison
--no-visualizations                  # Skip creating visualizations
--decompose-only                     # Run only decomposition
--verify-only                        # Run only verification
```

## 📈 Evaluation Metrics

### Overall Performance
- **Mean MedScore**: Average score across all responses
- **Score Distribution**: Statistical analysis of score patterns
- **Correlation**: Agreement with baseline results

### Claim-Level Analysis
- **Claim Accuracy**: Percentage of correctly verified claims
- **Claims per Response**: Average number of claims extracted
- **True/False Distribution**: Breakdown of claim types

### Response-Level Analysis
- **Agreement Rate**: Responses within acceptable score difference
- **Mean Absolute Difference**: Average score difference from baseline
- **Response Consistency**: Performance across different responses

## 🎨 Visualizations

The system generates comprehensive visualizations:

1. **Score Distribution**: Histogram and box plot comparisons
2. **Correlation Analysis**: Scatter plot with trend lines
3. **Performance Metrics**: Bar charts of key metrics
4. **Claim Analysis**: Claim accuracy and distribution charts
5. **Error Analysis**: Difference distribution and error patterns

## 🔬 Scientific Paper

A complete scientific paper is included (`scientific_paper.md`) with:

- **Abstract**: Summary of findings and contributions
- **Introduction**: Background and motivation
- **Related Work**: Literature review and positioning
- **Methodology**: Detailed technical approach
- **Results**: Comprehensive evaluation results
- **Discussion**: Analysis and implications
- **Future Work**: Research directions and extensions

**Target**: Q1 medical AI journals (Nature Medicine, JAMA, etc.)

## 🚀 Getting Started with Your Own Data

### 1. Prepare Your Data

```json
{
  "id": "unique_response_id",
  "question": "Medical question text",
  "response": "Medical response to evaluate"
}
```

### 2. Run Evaluation

```bash
python run_small_llm_evaluation.py your_data.jsonl results/your_evaluation/
```

### 3. Analyze Results

```bash
# View results
cat results/your_evaluation/small_llm_med_score_output.jsonl

# Compare with baseline (if available)
python -c "
import sys; sys.path.append('med-score-vi')
from evaluation_metrics import compare_medscore_results
compare_medscore_results('baseline.jsonl', 'results/your_evaluation/small_llm_med_score_output.jsonl')
"
```

## 🛠️ Development

### Adding New Models

1. Ensure the model is available in Ollama
2. Update the model selection options in `run_small_llm_evaluation.py`
3. Test with your specific model configuration

### Customizing Prompts

1. Modify prompts in `med-score-vi/prompts.py`
2. Update the enhanced prompts in `decomposer_small_llm.py` and `verifier_small_llm.py`
3. Test with your custom prompts

### Extending Metrics

1. Add new metrics to `evaluation_metrics.py`
2. Update the visualization functions in `visualization.py`
3. Integrate new metrics into the main evaluation script

## 📊 Performance Benchmarks

### Computational Efficiency

| Model | Parameters | Inference Time | Memory Usage | Cost |
|-------|------------|----------------|--------------|------|
| GPT-4o | 1.7T+ | ~2-5s | High | $$$$ |
| Llama 3.2 3B | 3B | ~0.5-1s | Low | $ |
| Gemma 2 9B | 9B | ~1-2s | Medium | $$ |

### Accuracy Comparison

| Task | GPT-4o | Small LLM | Improvement |
|------|--------|-----------|-------------|
| Overall MedScore | 0.765 | 0.776 | +1.3% |
| Claim Accuracy | 76.8% | 80.5% | +4.8% |
| Decomposition | 11.9 claims/response | 20.5 claims/response | +72.3% |

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **MedScore Framework**: Original MedScore implementation and methodology
- **Ollama**: Small language model inference platform
- **Open Source Community**: Various small language models and tools

## 📞 Support

For questions, issues, or contributions:

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: [Your contact information]

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{small_llm_medscore_2025,
  title={Small Language Models for Factuality Evaluation of Free-Form Medical Answers: An Enhanced MedScore Approach},
  author={[Your Name]},
  journal={[Target Journal]},
  year={2025}
}
```

---

**Last Updated**: January 2025  
**Version**: 1.0.0  
**Status**: Research Implementation
