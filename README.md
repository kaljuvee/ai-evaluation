# AI Evaluation Framework

A comprehensive collection of AI evaluation approaches using different frameworks and tools for assessing the performance of generative AI applications.

## Overview

This repository contains sample implementations for evaluating AI applications using seven different approaches:

1. **TrueLens Evaluation** - Using TruLens for feedback-driven evaluation
2. **DeepEval Evaluation** - Using DeepEval for RAG and summarization evaluation
3. **MLflow Evaluation** - Using MLflow for end-to-end MLOps and evaluation
4. **LangFuse Evaluation** - Using LangFuse for tracing and observability
5. **DSPy Evaluation** - Using DSPy for programmatic optimization
6. **LangChain Evaluation** - Using LangChain's built-in evaluation chains
7. **HuggingFace Evaluation** - Using HuggingFace's evaluation libraries
8. **Azure AI Evaluation** - Using Microsoft's Azure AI Evaluation SDK
9. **Ragas Evaluation** - Using Ragas by Exploding Gradients for RAG-specific evaluation

## Directory Structure

```
ai-evaluation/
├── truelens/                  # TruLens evaluation samples
│   ├── trulens_eval_sample.py
│   ├── requirements.txt
│   └── README.md
├── deepeval/                  # DeepEval evaluation samples
│   ├── deepeval_rag_sample.py
│   ├── deepeval_summarization_sample.py
│   ├── requirements.txt
│   └── README.md
├── mlflow/                    # MLflow evaluation samples
│   ├── mlflow_eval_sample.py
│   ├── requirements.txt
│   └── README.md
├── langfuse/                  # LangFuse evaluation samples
│   ├── langfuse_eval_sample.py
│   ├── requirements.txt
│   └── README.md
├── dspy/                      # DSPy evaluation samples
│   ├── dspy_sample.py
│   ├── dspy_multihop_sample.py
│   ├── requirements.txt
│   └── README.md
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
│   ├── requirements.txt
│   └── README.md
├── ragas/                    # Ragas evaluation samples
│   ├── ragas_evaluator.py
│   ├── requirements.txt
│   └── README.md
├── requirements.txt          # Global dependencies
└── README.md                # This file
```

## Evaluation Approaches

### 1. TrueLens Evaluation (`truelens/`)

**Purpose**: Evaluate LLM applications using feedback functions for groundedness, relevance, and more.

**Features**:
- Feedback-driven evaluation
- OpenTelemetry integration
- RAG and agent evaluation

### 2. DeepEval Evaluation (`deepeval/`)

**Purpose**: Unit testing for LLM applications with a focus on RAG and summarization.

**Features**:
- RAG and summarization metrics
- Pytest integration
- Synthetic data generation

### 3. MLflow Evaluation (`mlflow/`)

**Purpose**: End-to-end MLOps platform with robust evaluation and tracking capabilities.

**Features**:
- Experiment tracking
- Model registry
- QA and other evaluation types

### 4. LangFuse Evaluation (`langfuse/`)

**Purpose**: Observability and analytics for LLM applications with detailed tracing.

**Features**:
- Detailed tracing
- Human-in-the-loop scoring
- Production monitoring

### 5. DSPy Evaluation (`dspy/`)

**Purpose**: Programmatic optimization of LLM prompts and programs.

**Features**:
- Automatic prompt optimization
- Multi-hop question answering
- Metric-driven compilation

### 6. LangChain Evaluation (`langchain/`)

**Purpose**: Evaluate LLM outputs using LangChain's evaluation chains and criteria-based assessment.

### 7. HuggingFace Evaluation (`huggingface/`)

**Purpose**: Evaluate AI models using HuggingFace's evaluation libraries and LLM-as-a-judge approach.

### 8. Azure AI Evaluation (`azure/`)

**Purpose**: Comprehensive evaluation using Microsoft's Azure AI Evaluation SDK with cloud integration.

### 9. Ragas Evaluation (`ragas/`)

**Purpose**: Comprehensive RAG (Retrieval-Augmented Generation) evaluation using Ragas.

## Performance Comparison

| Framework | Best For | Key Strengths | Learning Curve |
|-----------|----------|---------------|----------------|
| **TrueLens** | Agent & RAG evaluation | Feedback functions, OpenTelemetry | Medium |
| **DeepEval** | Unit testing LLMs | RAG & summarization metrics | Low |
| **MLflow** | End-to-end MLOps | Experiment tracking, model registry | Medium |
| **LangFuse** | Production monitoring | Tracing, human-in-the-loop scoring | Low-Medium |
| **DSPy** | Prompt optimization | Programmatic optimization, multi-hop QA | High |
| **LangChain** | General LLM evaluation | Easy integration, multiple metrics | Low |
| **HuggingFace** | Model comparison | LLM-as-a-judge, custom metrics | Medium |
| **Azure AI** | Enterprise evaluation | Cloud integration, safety metrics | Medium |
| **Ragas** | RAG-specific evaluation | RAG metrics, synthetic data generation | Low-Medium |

## Contributing

To add new evaluation approaches:

1. Create a new directory for your framework
2. Include a `requirements.txt` file
3. Add a comprehensive `README.md`
4. Include example scripts and documentation
5. Update this main README with your approach

## License

This project is licensed under the MIT License - see the LICENSE file for details.

