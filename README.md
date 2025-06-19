# AI Evaluation Framework

A comprehensive collection of AI evaluation approaches using different frameworks and tools for assessing the performance of generative AI applications.

## Overview

This repository contains sample implementations for evaluating AI applications using four different approaches:

1. **LangChain Evaluation** - Using LangChain's built-in evaluation chains
2. **HuggingFace Evaluation** - Using HuggingFace's evaluation libraries
3. **Azure AI Evaluation** - Using Microsoft's Azure AI Evaluation SDK
4. **Ragas Evaluation** - Using Ragas by Exploding Gradients for RAG-specific evaluation

## Directory Structure

```
ai-evaluation/
├── langchain/                 # LangChain evaluation samples
│   ├── langchain_eval_sample.py
│   └── __init__.py
├── huggingface/              # HuggingFace evaluation samples
│   ├── llm_as_judge.py
│   ├── README.md
│   └── __init__.py
├── azure/                    # Azure AI evaluation samples
│   ├── azure_agent_eval_sample.py
│   ├── azure_rag_eval_sample.py
│   ├── synthetic_data_generator.py
│   ├── requirements.txt
│   └── README.md
├── ragas/                    # Ragas evaluation samples
│   ├── ragas_evaluator.py
│   ├── example_evaluations.py
│   ├── demo.py
│   ├── test_installation.py
│   ├── requirements.txt
│   ├── README.md
│   └── __init__.py
├── requirements.txt          # Global dependencies
└── README.md                # This file
```

## Evaluation Approaches

### 1. LangChain Evaluation (`langchain/`)

**Purpose**: Evaluate LLM outputs using LangChain's evaluation chains and criteria-based assessment.

**Features**:
- QA evaluation against ground truth
- Criteria-based evaluation (conciseness, relevance, correctness)
- Embedding distance evaluation
- String distance evaluation
- Pairwise string comparison
- Dataset loading from LangChain's collection

**Key Components**:
- `QAEvalChain` - Question-answering accuracy
- `CriteriaEvalChain` - Custom criteria evaluation
- `EmbeddingDistanceEvalChain` - Semantic similarity
- `StringDistanceEvalChain` - Text similarity metrics

**Usage**:
```bash
cd langchain
python langchain_eval_sample.py
```

### 2. HuggingFace Evaluation (`huggingface/`)

**Purpose**: Evaluate AI models using HuggingFace's evaluation libraries and LLM-as-a-judge approach.

**Features**:
- LLM-as-a-judge evaluation methodology
- Custom evaluation criteria
- Model comparison capabilities
- Integration with HuggingFace datasets

**Key Components**:
- LLM judge evaluation
- Custom evaluation metrics
- Dataset integration

**Usage**:
```bash
cd huggingface
python llm_as_judge.py
```

### 3. Azure AI Evaluation (`azure/`)

**Purpose**: Comprehensive evaluation using Microsoft's Azure AI Evaluation SDK with cloud integration.

**Features**:
- Agent evaluation with multiple evaluators
- RAG-specific evaluation (retrieval, groundedness)
- Content safety evaluation
- Synthetic data generation
- Azure AI Studio integration
- Batch evaluation capabilities

**Key Components**:
- **Agent Evaluation**: Quality, safety, and accuracy metrics
- **RAG Evaluation**: Retrieval accuracy, groundedness, response completeness
- **Synthetic Data Generation**: Automated test data creation
- **Azure Integration**: Results logging to Azure AI Studio

**Evaluators Included**:
- Quality: Relevance, Coherence, Fluency, Groundedness
- Safety: Violence, Sexual Content, Self-Harm, Hate/Unfairness
- Similarity: F1 Score, ROUGE, BLEU, Embedding Distance
- RAG: Retrieval, Document Retrieval, Response Completeness

**Usage**:
```bash
cd azure
# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python synthetic_data_generator.py

# Run agent evaluation
python azure_agent_eval_sample.py

# Run RAG evaluation
python azure_rag_eval_sample.py
```

### 4. Ragas Evaluation (`ragas/`)

**Purpose**: Comprehensive RAG (Retrieval-Augmented Generation) evaluation using [Ragas](https://docs.ragas.io/en/latest/references/) by Exploding Gradients.

**Features**:
- **Answer Correctness**: LLM-based factual accuracy assessment
- **Faithfulness**: Ensures answers don't hallucinate beyond retrieved context
- **Context Precision**: Measures retrieval quality and relevance
- **Context Recall**: Evaluates completeness of information retrieval
- **Synthetic Test Set Generation**: Automatically generates diverse test questions
- **LangChain Integration**: Seamless integration with LangChain components
- **HuggingFace Support**: Uses HuggingFace embeddings and models
- **FAISS Vector Store**: Efficient similarity search for retrieval
- **Caching**: Intelligent caching for cost-effective evaluations

**Key Metrics**:
- Answer Correctness (0-1): Factual accuracy of generated answers
- Faithfulness (0-1): Whether answer stays true to retrieved context
- Context Precision (0-1): Relevance of retrieved documents to question
- Context Recall (0-1): Completeness of relevant information retrieval
- Context Relevancy (0-1): Overall relevance of retrieved context
- Answer Relevancy (0-1): How relevant the answer is to the question

**Usage**:
```bash
cd ragas
# Install dependencies
pip install -r requirements.txt

# Test installation
python test_installation.py

# Run quick demo
python demo.py

# Run comprehensive evaluation
python ragas_evaluator.py

# Run examples with different scenarios
python example_evaluations.py
```

**Quick Start**:
```python
from ragas_evaluator import RagasEvaluator, EvaluationConfig

# Configure evaluation
config = EvaluationConfig(
    model_name="gpt-3.5-turbo",
    batch_size=5
)

# Run evaluation
evaluator = RagasEvaluator(config)
results = evaluator.run_comprehensive_evaluation()

# View results
print("Answer Correctness:", results['metrics']['answer_correctness'])
print("Faithfulness:", results['metrics']['faithfulness'])
```

## Setup and Installation

### Prerequisites

1. **Python 3.8+**
2. **Virtual Environment** (recommended)
3. **Azure AI Project** (for Azure evaluation)
4. **OpenAI API Key** (for some evaluators)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-evaluation
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install global dependencies:
```bash
pip install -r requirements.txt
```

4. Install framework-specific dependencies:
```bash
# For Azure evaluation
cd azure && pip install -r requirements.txt

# For LangChain evaluation
pip install langchain langchain-openai langchain-community python-dotenv rapidfuzz

# For HuggingFace evaluation
pip install transformers datasets evaluate

# For Ragas evaluation
cd ragas && pip install -r requirements.txt
```

## Configuration

### Environment Variables

Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your-openai-api-key
AZURE_AI_PROJECT_NAME=your-project-name
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=your-resource-group
AZURE_WORKSPACE_NAME=your-workspace-name
```

### Azure AI Configuration

For Azure evaluation, update the configuration in the scripts:
```python
azure_ai_config = AzureAIConfig(
    project_name="your-project-name",
    subscription_id="your-subscription-id",
    resource_group="your-resource-group",
    workspace_name="your-workspace-name"
)
```

### Ragas Configuration

For Ragas evaluation, configure the evaluation parameters:
```python
from ragas_evaluator import EvaluationConfig

config = EvaluationConfig(
    model_name="gpt-3.5-turbo",    # LLM for evaluation
    temperature=0.0,               # Generation temperature
    max_tokens=1000,               # Max tokens per response
    batch_size=10,                 # Batch size for evaluation
    cache_dir="./cache",           # Cache directory
    results_dir="./eval-results"   # Results output directory
)
```

## Data Formats

### LangChain
- Single query-response pairs
- Criteria-based evaluation
- Embedding and string distance metrics

### HuggingFace
- Dataset-based evaluation
- LLM-as-a-judge methodology
- Custom evaluation criteria

### Azure AI
- JSONL format for batch evaluation
- Conversation format for multi-turn evaluation

### Ragas
- DataFrame format with questions, answers, and contexts
- Synthetic test set generation
- Comprehensive metrics output in JSON and CSV formats

## Performance Comparison

| Framework | Best For | Key Strengths | Learning Curve |
|-----------|----------|---------------|----------------|
| **LangChain** | General LLM evaluation | Easy integration, multiple metrics | Low |
| **HuggingFace** | Model comparison | LLM-as-judge, custom metrics | Medium |
| **Azure AI** | Enterprise evaluation | Cloud integration, safety metrics | Medium |
| **Ragas** | RAG-specific evaluation | RAG metrics, synthetic data generation | Low-Medium |

## Use Cases

### When to Use Each Approach

**LangChain Evaluation**:
- Quick prototyping and testing
- Criteria-based evaluation
- Integration with LangChain applications
- Local evaluation without cloud dependencies

**HuggingFace Evaluation**:
- Model comparison and benchmarking
- Dataset-based evaluation
- LLM-as-a-judge methodology
- Research and academic applications

**Azure AI Evaluation**:
- Production-grade evaluation
- Comprehensive safety and quality assessment
- RAG application evaluation
- Enterprise applications with Azure integration
- Synthetic data generation for testing

**Ragas Evaluation**:
- Comprehensive RAG evaluation
- RAG-specific metrics
- Synthetic data generation
- LangChain and HuggingFace integration

## Contributing

To add new evaluation approaches:

1. Create a new directory for your framework
2. Include a `requirements.txt` file
3. Add a comprehensive `README.md`
4. Include example scripts and documentation
5. Update this main README with your approach

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- [LangChain Evaluation Documentation](https://python.langchain.com/api_reference/langchain/evaluation.html)
- [HuggingFace Evaluation](https://huggingface.co/docs/evaluate)
- [Azure AI Evaluation SDK](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/evaluate-sdk)
- [Azure AI Simulator](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/simulator-interaction-data)