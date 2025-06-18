# AI Evaluation Framework

A comprehensive collection of AI evaluation approaches using different frameworks and tools for assessing the performance of generative AI applications.

## Overview

This repository contains sample implementations for evaluating AI applications using three different approaches:

1. **LangChain Evaluation** - Using LangChain's built-in evaluation chains
2. **HuggingFace Evaluation** - Using HuggingFace's evaluation libraries
3. **Azure AI Evaluation** - Using Microsoft's Azure AI Evaluation SDK

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
- Structured data with query, response, context, and ground truth

## Output and Results

### LangChain
- Console output with evaluation scores
- Detailed reasoning for each evaluation

### HuggingFace
- Evaluation metrics and scores
- Model comparison results

### Azure AI
- JSON result files
- Azure AI Studio portal integration
- Comprehensive metrics dashboard
- Row-level evaluation data

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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your evaluation samples
4. Update documentation
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- [LangChain Evaluation Documentation](https://python.langchain.com/api_reference/langchain/evaluation.html)
- [HuggingFace Evaluation](https://huggingface.co/docs/evaluate)
- [Azure AI Evaluation SDK](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/evaluate-sdk)
- [Azure AI Simulator](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/simulator-interaction-data)