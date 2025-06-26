# DeepEval RAG Evaluation Sample

This sample demonstrates how to use DeepEval to evaluate RAG (Retrieval-Augmented Generation) pipelines with comprehensive metrics and testing approaches.

## 🚀 Features Demonstrated

### Core Evaluation Metrics
- **Faithfulness**: Measures factual consistency against provided context
- **Answer Relevancy**: Assesses how well responses align with input queries
- **Context Relevancy**: Evaluates if retrieved context is relevant to queries
- **Hallucination Detection**: Identifies incorrect or unsupported information
- **Toxicity Detection**: Evaluates for harmful or biased content

### Advanced Features
- **Custom Metrics**: Custom RAG quality evaluation using G-Eval
- **Conversational Testing**: Multi-turn interaction evaluation
- **Synthetic Dataset Generation**: Automatic test case generation
- **Batch Evaluation**: Comprehensive testing with multiple metrics
- **Dataset Management**: Save and load evaluation datasets

## 📋 Prerequisites

1. **Python 3.8+**
2. **OpenAI API Key** (for LLM-based evaluations)
3. **DeepEval Account** (optional, for cloud features)

## 🛠️ Installation

1. **Install DeepEval**:
   ```bash
   pip install deepeval
   ```

2. **Set up environment variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

3. **Optional: Set up cloud features**:
   ```bash
   deepeval login
   ```

## 🎯 Usage

### Quick Start

Run the complete sample:
```bash
python deepeval_rag_sample.py
```

### Individual Tests

Run specific evaluation tests:
```bash
# Test faithfulness
python -c "from deepeval_rag_sample import test_rag_faithfulness; test_rag_faithfulness()"

# Test answer relevancy
python -c "from deepeval_rag_sample import test_rag_answer_relevancy; test_rag_answer_relevancy()"

# Test custom metrics
python -c "from deepeval_rag_sample import test_custom_rag_metric; test_custom_rag_metric()"
```

### Using Pytest

Run tests with pytest:
```bash
pytest deepeval_rag_sample.py -v
```

## 📊 Evaluation Examples

### 1. Basic RAG Evaluation

```python
from deepeval import assert_test
from deepeval.metrics import Faithfulness
from deepeval.test_case import LLMTestCase

# Create test case
test_case = LLMTestCase(
    input="What are the benefits of RAG?",
    actual_output="RAG provides improved accuracy through context retrieval.",
    expected_output="RAG systems offer improved accuracy by retrieving relevant context.",
    retrieval_context=["RAG systems combine LLMs with external knowledge retrieval."]
)

# Evaluate with faithfulness metric
faithfulness_metric = Faithfulness(threshold=0.7)
assert_test(test_case, [faithfulness_metric])
```

### 2. Custom RAG Metric

```python
from deepeval.metrics import GEval

class CustomRAGMetric(GEval):
    def __init__(self):
        super().__init__(
            name="RAG Quality",
            criteria="Evaluate accuracy, completeness, context usage, and clarity",
            evaluation_params=["actual_output", "expected_output", "retrieval_context"],
            threshold=0.7
        )
```

### 3. Conversational RAG Testing

```python
from deepeval.test_case import ConversationalTestCase

test_case = ConversationalTestCase(
    messages=[
        {"role": "user", "content": "Tell me about RAG"},
        {"role": "assistant", "content": "RAG combines LLMs with retrieval."},
        {"role": "user", "content": "What are the advantages?"}
    ],
    expected_output="RAG provides improved accuracy and reduced hallucinations.",
    retrieval_context=["RAG systems improve accuracy through context retrieval."]
)
```

### 4. Synthetic Dataset Generation

```python
from deepeval.synthesizer import Synthesizer

synthesizer = Synthesizer()
synthetic_test_cases = synthesizer.generate_test_cases(
    num_test_cases=5,
    test_case_type="rag",
    topic="artificial intelligence",
    include_context=True
)
```

## 🔧 Configuration

### Metric Thresholds

Adjust thresholds based on your requirements:
```python
# Strict evaluation
faithfulness_metric = Faithfulness(threshold=0.8)

# Lenient evaluation
faithfulness_metric = Faithfulness(threshold=0.5)
```

### Custom Evaluation Criteria

Define custom evaluation criteria for G-Eval:
```python
custom_metric = GEval(
    name="Business Logic",
    criteria="""
    Evaluate if the response:
    1. Follows company policies
    2. Uses approved language
    3. Provides actionable information
    """,
    threshold=0.7
)
```

## 📈 Results Interpretation

### Score Ranges
- **0.0-0.3**: Poor quality, needs improvement
- **0.4-0.6**: Acceptable quality with issues
- **0.7-0.8**: Good quality, mostly accurate
- **0.9-1.0**: Excellent quality

### Common Issues
- **Low Faithfulness**: Response doesn't align with provided context
- **Low Answer Relevancy**: Response doesn't address the query
- **High Hallucination**: Response contains unsupported information
- **Low Context Relevancy**: Retrieved context isn't relevant

## 🔗 Integration Examples

### With LangChain
```python
from langchain.chains import RetrievalQA
from deepeval import evaluate

# Your RAG pipeline
qa_chain = RetrievalQA.from_chain_type(llm, retriever)

# Create test cases from your data
test_cases = [
    LLMTestCase(
        input=query,
        actual_output=qa_chain.run(query),
        expected_output=expected_answer,
        retrieval_context=retrieved_docs
    )
    for query, expected_answer, retrieved_docs in your_test_data
]

# Evaluate
results = evaluate(test_cases, [Faithfulness(), AnswerRelevancy()])
```

### With LlamaIndex
```python
from llama_index import VectorStoreIndex
from deepeval import evaluate

# Your LlamaIndex setup
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Evaluate responses
test_cases = []
for query, expected in test_data:
    response = query_engine.query(query)
    test_cases.append(LLMTestCase(
        input=query,
        actual_output=str(response),
        expected_output=expected,
        retrieval_context=response.source_nodes
    ))

results = evaluate(test_cases, [Faithfulness(), AnswerRelevancy()])
```

## 🚀 Advanced Features

### CI/CD Integration
```yaml
# .github/workflows/rag-evaluation.yml
name: RAG Evaluation
on: [push, pull_request]
jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install deepeval
      - name: Run evaluation
        run: python deepeval_rag_sample.py
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

### Cloud Dashboard
Enable cloud features for:
- Regression testing
- Performance tracking
- Team collaboration
- Historical analysis

```bash
deepeval login
deepeval test run deepeval_rag_sample.py
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure DeepEval is installed correctly
   ```bash
   pip install --upgrade deepeval
   ```

2. **API Key Issues**: Verify OpenAI API key is set
   ```bash
   echo $OPENAI_API_KEY
   ```

3. **Memory Issues**: Reduce batch size for large evaluations
   ```python
   # Process in smaller batches
   for batch in chunks(test_cases, 10):
       results = evaluate(batch, metrics)
   ```

### Performance Optimization

1. **Parallel Processing**: Use multiple workers for large datasets
2. **Caching**: Cache evaluation results for repeated tests
3. **Batch Processing**: Process test cases in batches

## 📚 Additional Resources

- [DeepEval Documentation](https://docs.confident-ai.com/)
- [DeepEval GitHub](https://github.com/confident-ai/deepeval)
- [Confident AI Platform](https://confident-ai.com/)
- [Community Discord](https://discord.gg/confident-ai)

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this sample!

## 📄 License

This sample is provided under the MIT License. See the main project LICENSE file for details. 