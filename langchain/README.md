# LangChain Evaluation Sample

This directory contains a comprehensive example of using LangChain's evaluation framework to assess the performance of language models and AI applications.

## Overview

The `langchain_eval_sample.py` script demonstrates six different types of evaluations that can be performed using LangChain's evaluation chains:

1. **QA Evaluation** - Assesses question-answering accuracy
2. **Criteria-based Evaluation** - Evaluates responses against specific criteria
3. **Embedding Distance Evaluation** - Measures semantic similarity
4. **String Distance Evaluation** - Calculates text similarity using edit distance
5. **Pairwise String Comparison** - Compares two different responses
6. **Dataset Loading** - Loads evaluation datasets from LangChain's collection

## Script Breakdown

### 1. QA Evaluation
```python
qa_evaluator = load_evaluator("qa")
qa_result = qa_evaluator.evaluate_strings(
    prediction="The capital of France is Paris",
    input="What is the capital of France?",
    reference="Paris is the capital of France"
)
```
- **Purpose**: Evaluates how well an AI response matches a reference answer
- **Input**: Question, AI prediction, and reference answer
- **Output**: Score indicating correctness (0-1) with reasoning

### 2. Criteria-based Evaluation
```python
criteria_evaluator = load_evaluator("criteria")
criteria_result = criteria_evaluator.evaluate_strings(
    prediction="The weather is sunny today with a high of 75°F",
    input="What's the weather like?",
    criteria=[Criteria.CONCISENESS, Criteria.RELEVANCE, Criteria.CORRECTNESS]
)
```
- **Purpose**: Evaluates responses against specific quality criteria
- **Available Criteria**: 
  - `CONCISENESS` - Response brevity and clarity
  - `RELEVANCE` - Response relevance to the question
  - `CORRECTNESS` - Factual accuracy
  - `HELPFULNESS` - Overall helpfulness
  - `HARMFULNESS` - Potential harm assessment
- **Output**: Binary score (Y/N) with detailed reasoning

### 3. Embedding Distance Evaluation
```python
embedding_evaluator = load_evaluator("embedding_distance")
embedding_result = embedding_evaluator.evaluate_strings(
    prediction="The cat sat on the mat",
    reference="A feline was resting on the carpet",
    distance_metric=EmbeddingDistance.COSINE
)
```
- **Purpose**: Measures semantic similarity between texts using embeddings
- **Distance Metrics**:
  - `COSINE` - Cosine similarity (0-1, higher is more similar)
  - `EUCLIDEAN` - Euclidean distance
  - `MANHATTAN` - Manhattan distance
- **Output**: Similarity score with distance metric

### 4. String Distance Evaluation
```python
string_evaluator = load_evaluator("string_distance")
string_result = string_evaluator.evaluate_strings(
    prediction="Hello World",
    reference="Hello World!",
    distance_metric=StringDistance.LEVENSHTEIN
)
```
- **Purpose**: Measures exact text similarity using string distance algorithms
- **Distance Metrics**:
  - `LEVENSHTEIN` - Edit distance (minimum edits to transform one string to another)
  - `JARO_WINKLER` - Similarity for short strings like names
  - `HAMMING` - Character-by-character comparison
- **Output**: Distance score (lower is more similar)

### 5. Pairwise String Comparison
```python
pairwise_evaluator = load_evaluator("pairwise_string")
pairwise_result = pairwise_evaluator.evaluate_string_pairs(
    prediction=(
        "The quick brown fox jumps over the lazy dog",
        "A fast brown fox leaps over a sleepy dog"
    )
)
```
- **Purpose**: Compares two different responses to the same input
- **Use Cases**: Model comparison, A/B testing, preference evaluation
- **Output**: Preference score indicating which response is better

### 6. Dataset Loading
```python
dataset = load_dataset("llm-math")
```
- **Purpose**: Loads evaluation datasets from LangChain's collection
- **Available Datasets**: Various datasets for different evaluation tasks
- **Output**: Dataset object for batch evaluation

## Setup and Installation

### Prerequisites
- Python 3.8+
- OpenAI API key (for LLM-based evaluators)
- Virtual environment (recommended)

### Installation
```bash
# Install required packages
pip install langchain langchain-openai langchain-community python-dotenv rapidfuzz

# Set up environment variables
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### Running the Script
```bash
# Activate virtual environment (if using)
source .venv/bin/activate

# Run the evaluation script
python langchain_eval_sample.py
```

## Expected Output

The script will output results for each evaluation type:

```
QA Evaluation Result: {'reasoning': 'CORRECT', 'value': 'CORRECT', 'score': 1}

Criteria Evaluation Result: {'reasoning': 'The response is helpful...', 'value': 'Y', 'score': 1}

Embedding Distance Result: {'score': 0.0802959301089411}

String Distance Result: {'score': 0.01666666666666672}

Pairwise Comparison Result: {'reasoning': 'Both responses...', 'value': 'A', 'score': 1}

Dataset loaded successfully: <dataset object>
```

## Configuration

### Environment Variables
Create a `.env` file in the langchain directory:
```env
OPENAI_API_KEY=your-openai-api-key-here
```

### Customizing Evaluators
You can customize evaluators by passing additional parameters:

```python
# Custom criteria evaluation
custom_criteria = ["helpfulness", "conciseness", "relevance"]
criteria_evaluator = load_evaluator("criteria", criteria=custom_criteria)

# Custom embedding model
embedding_evaluator = load_evaluator("embedding_distance", embeddings=your_embedding_model)
```

## Use Cases

### When to Use Each Evaluator

**QA Evaluator**:
- Fact-checking applications
- Educational content evaluation
- Information retrieval systems

**Criteria Evaluator**:
- Content quality assessment
- Response filtering
- User experience evaluation

**Embedding Distance**:
- Semantic similarity tasks
- Content clustering
- Duplicate detection

**String Distance**:
- Exact text matching
- Plagiarism detection
- Data cleaning

**Pairwise Comparison**:
- Model comparison
- A/B testing
- Preference learning

## Extending the Script

### Adding Custom Evaluators
```python
from langchain.evaluation import StringEvaluator

class CustomEvaluator(StringEvaluator):
    def _evaluate_strings(self, prediction, input=None, reference=None):
        # Your custom evaluation logic
        return {"score": custom_score, "reasoning": "Custom reasoning"}
```

### Batch Evaluation
```python
# Evaluate multiple examples
examples = [
    {"prediction": "Answer 1", "input": "Question 1", "reference": "Reference 1"},
    {"prediction": "Answer 2", "input": "Question 2", "reference": "Reference 2"}
]

for example in examples:
    result = qa_evaluator.evaluate_strings(**example)
    print(f"Result: {result}")
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'rapidfuzz'**
   ```bash
   pip install rapidfuzz
   ```

2. **OpenAI API Key Error**
   - Ensure your `.env` file contains the correct API key
   - Check that the API key has sufficient credits

3. **Dataset Loading Errors**
   - Some datasets may require additional authentication
   - Check internet connection for dataset downloads

### Performance Tips

- Use batch evaluation for large datasets
- Cache embedding models for repeated use
- Consider using local models for faster evaluation

## References

- [LangChain Evaluation Documentation](https://python.langchain.com/api_reference/langchain/evaluation.html)
- [LangChain Evaluation Guide](https://python.langchain.com/docs/guides/evaluation/)
- [OpenAI API Documentation](https://platform.openai.com/docs) 