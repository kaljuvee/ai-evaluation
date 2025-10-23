# Ragas Evaluation System

A comprehensive evaluation framework for RAG (Retrieval-Augmented Generation) applications using [Ragas](https://docs.ragas.io/en/latest/references/) by Exploding Gradients.

## Overview

This module provides a complete evaluation pipeline for RAG systems, including:

- **Answer Correctness**: Evaluates how accurate the generated answers are
- **Faithfulness**: Measures if the answer is faithful to the retrieved context
- **Context Precision**: Assesses the relevance of retrieved context to the question
- **Context Recall**: Measures how much relevant information was retrieved
- **Context Relevancy**: Evaluates the overall relevance of retrieved context
- **Answer Relevancy**: Measures how relevant the answer is to the question

## Features

### 🎯 Core Evaluation Metrics
- **Answer Correctness**: Uses LLM-based evaluation to assess factual accuracy
- **Faithfulness**: Ensures answers don't hallucinate beyond retrieved context
- **Context Precision**: Measures retrieval quality and relevance
- **Context Recall**: Evaluates completeness of information retrieval
- **Synthetic Test Set Generation**: Automatically generates diverse test questions

### 🔧 Technical Features
- **LangChain Integration**: Seamless integration with LangChain components
- **HuggingFace Support**: Uses HuggingFace embeddings and models
<!-- FAISS Vector Store intentionally omitted in minimal OpenAI-only setup -->
- **Caching**: Intelligent caching for cost-effective evaluations
- **Batch Processing**: Efficient batch evaluation for large datasets

### 📊 Comprehensive Reporting
- Detailed metrics breakdown
- CSV exports of evaluation data
- JSON format for programmatic access
- Visual analysis capabilities

## Installation

```bash
pip install -r requirements.txt
```

## Environment Setup

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Quick Start

### Basic Usage

```python
from ragas_evaluator import RagasEvaluator, EvaluationConfig

# Configure evaluation
config = EvaluationConfig(
    model_name="gpt-3.5-turbo",
    temperature=0.0,
    batch_size=5
)

# Run evaluation
evaluator = RagasEvaluator(config)
results = evaluator.run_comprehensive_evaluation()

# View results
print("Answer Correctness:", results['metrics']['answer_correctness'])
print("Faithfulness:", results['metrics']['faithfulness'])
print("Context Precision:", results['metrics']['context_precision'])
```

### Custom RAG System Evaluation

```python
from ragas_evaluator import RAGSystem, TestSetGenerator, RagasEvaluator

# Your documents
documents = [
    "Your document content here...",
    "More document content..."
]

# Create RAG system
rag_system = RAGSystem(documents, config)

# Generate test set
generator = TestSetGenerator(config)
testset = generator.generate_from_documents(documents, num_questions=50)

# Evaluate
evaluator = RagasEvaluator(config)
results = evaluator.evaluate_rag_system(rag_system, testset)
```

## Test Set Generation

The system includes sophisticated test set generation capabilities:

### Synthetic Data Generation
```python
generator = TestSetGenerator(config)

# Generate from your documents
testset = generator.generate_from_documents(your_documents, num_questions=50)

# Or use built-in synthetic dataset
testset = generator.generate_synthetic_dataset()
```

### Question Types
The generator creates diverse question types:
- **Simple Questions**: Direct factual queries
- **Reasoning Questions**: Require logical reasoning
- **Multi-context Questions**: Span multiple documents

## Evaluation Metrics Explained

### Answer Correctness
- **What it measures**: Factual accuracy of generated answers
- **How it works**: Uses LLM-based evaluation comparing answer to ground truth
- **Score range**: 0-1 (higher is better)

### Faithfulness
- **What it measures**: Whether the answer stays true to retrieved context
- **How it works**: Evaluates if answer contains information not in context
- **Score range**: 0-1 (higher is better)

### Context Precision
- **What it measures**: Relevance of retrieved documents to the question
- **How it works**: Assesses how well retrieval system finds relevant context
- **Score range**: 0-1 (higher is better)

### Context Recall
- **What it measures**: Completeness of relevant information retrieval
- **How it works**: Evaluates if all relevant information was retrieved
- **Score range**: 0-1 (higher is better)

## Configuration Options

```python
@dataclass
class EvaluationConfig:
    model_name: str = "gpt-3.5-turbo"      # LLM for evaluation
    temperature: float = 0.0               # Generation temperature
    max_tokens: int = 1000                 # Max tokens per response
    batch_size: int = 10                   # Batch size for evaluation
    cache_dir: str = "./cache"             # Cache directory
    results_dir: str = "./eval-results"    # Results output directory
```

## Output Files

The evaluation generates several output files:

- `ragas_metrics.json`: Detailed metrics in JSON format
- `evaluation_data.csv`: Raw evaluation data with questions, answers, and contexts
- `testset.csv`: Generated test questions and ground truth

## Advanced Usage

### Custom Metrics
```python
from ragas.metrics import CustomMetric

# Add custom metrics to evaluation
evaluator = RagasEvaluator(config)
evaluator.metrics.append(your_custom_metric)
```

### Batch Evaluation
```python
# Evaluate multiple RAG systems
rag_systems = [rag1, rag2, rag3]
results = []

for rag in rag_systems:
    result = evaluator.evaluate_rag_system(rag, testset)
    results.append(result)
```

### Cost Optimization
```python
# Use caching to reduce API calls
config = EvaluationConfig(
    cache_dir="./cache",
    batch_size=20  # Larger batches reduce API overhead
)
```

## Integration Examples

### LangChain Integration (OpenAI-only)
Use retrieval backends that do not require FAISS or omit vectorstore examples in this minimal setup.

### HuggingFace Models
```python
from langchain.embeddings import HuggingFaceEmbeddings

# Use different embedding models
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
```

## Performance Tips

1. **Use Caching**: Enable caching to avoid re-evaluating same questions
2. **Batch Processing**: Increase batch size for better throughput
3. **Model Selection**: Use appropriate model size for your use case
4. **Test Set Size**: Balance between coverage and evaluation time

## Troubleshooting

### Common Issues

1. **OpenAI API Key Missing**
   ```bash
   export OPENAI_API_KEY="your-key"
   ```

2. **Memory Issues with Large Datasets**
   - Reduce batch size
   - Use smaller embedding models
   - Process in chunks

3. **Slow Evaluation**
   - Enable caching
   - Increase batch size
   - Use faster models

## Contributing

To extend the evaluation system:

1. Add new metrics to `RagasEvaluator.metrics`
2. Implement custom test set generators
3. Add new evaluation configurations
4. Extend the RAG system interface

## References

- [Ragas Documentation](https://docs.ragas.io/en/latest/references/)
- [LangChain Documentation](https://python.langchain.com/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

## License

This project is part of the AI Evaluation framework. See the main LICENSE file for details. 