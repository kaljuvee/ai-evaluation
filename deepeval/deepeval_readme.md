# DeepEval: A Simple Guide to LLM Evaluation with Ground Truth

## What is DeepEval?

**DeepEval** is a testing framework for LLMs (Large Language Models) that works like unit testing for your AI applications. Think of it as pytest, but specifically designed for evaluating AI outputs.

### Why Use DeepEval?

- ✅ **Easy to understand**: Uses familiar testing concepts
- ✅ **Ground truth comparison**: Compare AI outputs against expected answers
- ✅ **Custom metrics**: Create your own evaluation criteria with G-Eval
- ✅ **Pytest integration**: Works with your existing test suite
- ✅ **Multiple metrics**: Faithfulness, relevance, hallucination detection, and more

## Understanding Ground Truth

**Ground truth** (also called "expected output") is the ideal answer you want your AI to produce. It's like the answer key in a test.

### Example:
- **Input**: "What are the primary colors?"
- **Ground Truth**: "The primary colors are red, blue, and yellow."
- **Actual Output**: "The primary colors are green, orange, and purple." ❌
- **Score**: 0.0 (completely wrong!)

## Quick Start: Basic Ground Truth Comparison

Here's the simplest way to compare actual output against expected output:

```python
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

# Step 1: Define your metric
correctness_metric = GEval(
    name="Correctness",
    model="gpt-4o",  # The LLM that judges your output
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,   # What your AI produced
        LLMTestCaseParams.EXPECTED_OUTPUT  # What it should have produced
    ],
    threshold=0.5  # Minimum score to pass (0.0 to 1.0)
)

# Step 2: Create a test case
test_case = LLMTestCase(
    input="What are the main causes of deforestation?",
    actual_output="The main causes of deforestation include agricultural expansion and logging.",
    expected_output="The main causes of deforestation include agricultural expansion, logging, infrastructure development, and urbanization."
)

# Step 3: Run the test
assert_test(test_case, [correctness_metric])
```

### What Happens?
1. DeepEval sends both outputs to GPT-4o
2. GPT-4o compares them based on your criteria
3. You get a score (0.0 to 1.0) and an explanation

## Understanding G-Eval: Custom Metrics Made Simple

**G-Eval** lets you create custom evaluation criteria using natural language. Instead of writing complex code, you just describe what "good" looks like.

### Basic G-Eval Structure

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

metric = GEval(
    name="Your Metric Name",
    model="gpt-4o",  # The judge model
    criteria="What you're evaluating",  # OR use evaluation_steps
    evaluation_params=[  # What data to compare
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    threshold=0.7  # Pass/fail threshold
)
```

### Two Ways to Define Criteria

#### Option 1: Simple Criteria (G-Eval generates steps)
```python
correctness_metric = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT]
)
```

#### Option 2: Detailed Steps (You control the evaluation)
```python
correctness_metric = GEval(
    name="Correctness",
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradict any facts in 'expected output'",
        "Heavily penalize omission of important details",
        "Vague language or contradicting opinions are acceptable"
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT]
)
```

**💡 Tip**: Use `evaluation_steps` for more consistent results!

## Complete Working Example

Here's a full example with multiple test cases showing different score ranges:

```python
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.dataset import EvaluationDataset

# Define the metric once
correctness_metric = GEval(
    name="Correctness",
    model="gpt-4o-mini",
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradict any facts in 'expected output'",
        "Lightly penalize omission of detail, focus on the main idea",
        "Vague language or contradicting opinions are OK"
    ],
    evaluation_params=[
        LLMTestCaseParams.EXPECTED_OUTPUT, 
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    threshold=0.7
)

# Create test cases with ground truth
test_cases = [
    # Score: 1.0 (perfect match)
    LLMTestCase(
        input="What are the main causes of deforestation?",
        actual_output="The main causes of deforestation include agricultural expansion, logging, infrastructure development, and urbanization.",
        expected_output="The main causes of deforestation include agricultural expansion, logging, infrastructure development, and urbanization."
    ),
    
    # Score: ~0.5 (partial match - missing detail)
    LLMTestCase(
        input="Define the term 'artificial intelligence'.",
        actual_output="Artificial intelligence is the simulation of human intelligence by machines.",
        expected_output="Artificial intelligence refers to the simulation of human intelligence in machines that are programmed to think and learn like humans, including tasks such as problem-solving, decision-making, and language understanding."
    ),
    
    # Score: 0.0 (contradiction/factually wrong)
    LLMTestCase(
        input="List the primary colors.",
        actual_output="The primary colors are green, orange, and purple.",
        expected_output="The primary colors are red, blue, and yellow."
    )
]

# Run evaluation
dataset = EvaluationDataset(test_cases=test_cases)
results = dataset.evaluate([correctness_metric])

# Access scores
for i, test_case in enumerate(test_cases):
    print(f"Test {i+1}: Score = {results[i].metrics[0].score}")
    print(f"Reason: {results[i].metrics[0].reason}\n")
```

### Expected Output:
```
Test 1: Score = 1.0
Reason: The actual output perfectly matches the expected output.

Test 2: Score = 0.5
Reason: The actual output captures the basic concept but omits important details about thinking, learning, and specific tasks.

Test 3: Score = 0.0
Reason: The actual output contradicts the expected output with completely incorrect information.
```

## Key Parameters Reference

| Parameter | Where to Set | Purpose | Example |
|-----------|--------------|---------|---------|
| `actual_output` | `LLMTestCase` | What your LLM generated | `"AI is artificial intelligence."` |
| `expected_output` | `LLMTestCase` | Ground truth / ideal answer | `"AI stands for artificial intelligence..."` |
| `input` | `LLMTestCase` | The query/prompt | `"What is AI?"` |
| `context` | `LLMTestCase` | Source documents (for RAG) | `["AI is artificial intelligence..."]` |
| `retrieval_context` | `LLMTestCase` | Retrieved docs (for RAG) | `["Document 1", "Document 2"]` |
| `evaluation_params` | `GEval` | Which fields to compare | `[ACTUAL_OUTPUT, EXPECTED_OUTPUT]` |
| `threshold` | `GEval` | Minimum passing score | `0.7` (70%) |

## When to Use Expected Output vs Context

| Scenario | Use This | Example |
|----------|----------|---------|
| You have ideal answers | `expected_output` | Q&A datasets, benchmark tests |
| You have source documents only | `context` or `retrieval_context` | RAG evaluation (faithfulness) |
| You have both | Use both! | Comprehensive RAG evaluation |

### Example: RAG with Both
```python
test_case = LLMTestCase(
    input="What is quantum computing?",
    actual_output="Quantum computing uses quantum mechanics for computation.",
    expected_output="Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to perform computations.",
    retrieval_context=["Quantum computing is a type of computation that harnesses quantum mechanics..."]
)

# Evaluate both correctness AND faithfulness
correctness = GEval(...)  # Compare to expected_output
faithfulness = Faithfulness()  # Check against retrieval_context

evaluate([test_case], [correctness, faithfulness])
```

## Advanced: Flexible Correctness Metric

This metric works whether you have ground truth or just source documents:

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

flexible_correctness = GEval(
    name="Flexible Correctness",
    evaluation_steps=[
        "If expected_output is provided, compare actual_output against it",
        "If no expected_output, check if actual_output is supported by context",
        "Penalize contradictions and unsupported claims",
        "Allow opinions if clearly marked as such"
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,  # Optional
        LLMTestCaseParams.CONTEXT           # Fallback
    ]
)

# Test case WITH ground truth
test_with_truth = LLMTestCase(
    input="What is AI?",
    actual_output="AI is artificial intelligence.",
    expected_output="AI stands for artificial intelligence, simulating human cognition.",
    context=["AI is artificial intelligence..."]  # Optional backup
)

# Test case WITHOUT ground truth (uses context)
test_without_truth = LLMTestCase(
    input="What is AI?",
    actual_output="AI is artificial intelligence.",
    context=["AI is artificial intelligence..."]  # Required here
)
```

## Common Patterns and Use Cases

### 1. Testing a Chatbot
```python
# Test if chatbot gives correct answers
test_case = LLMTestCase(
    input="How do I reset my password?",
    actual_output=chatbot.respond("How do I reset my password?"),
    expected_output="Click 'Forgot Password' on the login page and follow the email instructions."
)
```

### 2. Evaluating Summaries
```python
# Test if summary captures key points
test_case = LLMTestCase(
    input=long_article,
    actual_output=summarizer.summarize(long_article),
    expected_output="Expected summary with key points..."
)
```

### 3. RAG System Evaluation
```python
# Test both correctness and faithfulness
test_case = LLMTestCase(
    input="What are the benefits of exercise?",
    actual_output=rag_system.query("What are the benefits of exercise?"),
    expected_output="Exercise improves cardiovascular health, mental well-being, and strength.",
    retrieval_context=retrieved_documents
)

evaluate([test_case], [
    GEval(...),           # Correctness vs expected
    Faithfulness(),       # Faithfulness to context
    AnswerRelevancy()     # Relevance to question
])
```

## Important Rules

### ⚠️ Must Match Parameters
If you include `EXPECTED_OUTPUT` in `evaluation_params`, you **must** provide `expected_output` in `LLMTestCase`, or you'll get an error.

```python
# ❌ This will error
metric = GEval(evaluation_params=[LLMTestCaseParams.EXPECTED_OUTPUT])
test = LLMTestCase(input="...", actual_output="...")  # Missing expected_output!

# ✅ This works
metric = GEval(evaluation_params=[LLMTestCaseParams.EXPECTED_OUTPUT])
test = LLMTestCase(input="...", actual_output="...", expected_output="...")
```

### ⚠️ Use Either Criteria OR Evaluation Steps
Don't use both at the same time:

```python
# ❌ Don't do this
GEval(
    criteria="...",
    evaluation_steps=[...],  # Pick one!
)

# ✅ Do this
GEval(criteria="...")
# OR
GEval(evaluation_steps=[...])
```

### ⚠️ Order Matters in Evaluation Steps
The judge LLM follows your steps in order:

```python
# ✅ Good order
evaluation_steps=[
    "First, check for factual errors",
    "Then, check for missing details",
    "Finally, check for clarity"
]

# ❌ Confusing order
evaluation_steps=[
    "Check for clarity",
    "But first check for errors",  # Contradicts previous step
]
```

## Score Interpretation

| Score Range | Meaning | Action |
|-------------|---------|--------|
| 0.9 - 1.0 | Excellent | No changes needed |
| 0.7 - 0.9 | Good | Minor improvements possible |
| 0.5 - 0.7 | Acceptable | Needs improvement |
| 0.3 - 0.5 | Poor | Significant issues |
| 0.0 - 0.3 | Very Poor | Major problems |

## Troubleshooting

### Issue: Low Scores Despite Good Outputs
**Solution**: Your criteria might be too strict. Try:
- Lowering the threshold
- Adjusting evaluation steps to be more lenient
- Checking if expected_output is too specific

### Issue: All Scores Are 0.0 or 1.0
**Solution**: Your criteria might be too binary. Try:
- Adding more nuanced evaluation steps
- Using multiple metrics
- Checking if the judge model is appropriate

### Issue: Inconsistent Scores
**Solution**: The evaluation might be too subjective. Try:
- Using `evaluation_steps` instead of `criteria`
- Making steps more specific and objective
- Using a more powerful judge model (e.g., gpt-4o instead of gpt-4o-mini)

## Installation and Setup

```bash
# Install DeepEval
pip install deepeval

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key"

# Optional: Login for cloud features
deepeval login
```

## Running Tests

```bash
# Run with pytest
pytest your_test_file.py -v

# Run with DeepEval CLI
deepeval test run your_test_file.py

# Run a single test
python -c "from your_module import test_function; test_function()"
```

## Additional Resources

- [DeepEval Documentation](https://docs.confident-ai.com/)
- [G-Eval Guide](https://www.confident-ai.com/blog/a-step-by-step-guide-to-evaluating-an-llm-text)
- [Answer Correctness Metric](https://docs.confident-ai.com/docs/metrics-answer-correctness)
- [DeepEval GitHub](https://github.com/confident-ai/deepeval)

## Summary

**DeepEval makes LLM evaluation simple:**

1. **Define your metric** with G-Eval (describe what "good" looks like)
2. **Create test cases** with input, actual output, and expected output
3. **Run evaluation** and get scores with explanations
4. **Iterate** based on results

The key insight: **Ground truth (expected output) lets you objectively measure how close your AI is to the ideal answer.**

Start simple, then add more sophisticated metrics as needed!
