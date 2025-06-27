# Ragas Summary Evaluation

This module provides a comprehensive implementation of text summarization evaluation using Ragas, following the official documentation examples.

## Features

- **Non-LLM Metrics**: BleuScore for traditional evaluation
- **LLM-based Metrics**: AspectCritic for more nuanced evaluation
- **Dataset Evaluation**: Support for both Hugging Face datasets and custom data
- **Results Export**: Export results to CSV and pandas DataFrame
- **Flexible Configuration**: Easy setup with OpenAI API

## Installation

The required dependencies are already included in `requirements.txt`. Make sure you have activated your virtual environment:

```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r ragas/requirements.txt
```

## Setup

### 1. OpenAI API Key

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or set it in your Python script:

```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

### 2. Test Installation

Run the test script to verify everything is working:

```bash
cd ragas
python test_summary_eval.py
```

## Usage

### Basic Usage

```python
import asyncio
from ragas_summary_eval import RagasSummaryEvaluator

async def main():
    # Initialize evaluator
    evaluator = RagasSummaryEvaluator()
    
    # Run full evaluation
    results = await evaluator.evaluate_dataset()
    print(f"Results: {results}")

# Run the evaluation
asyncio.run(main())
```

### Single Sample Evaluation

```python
from ragas_summary_eval import RagasSummaryEvaluator

# Initialize evaluator
evaluator = RagasSummaryEvaluator()

# Create sample data
sample_data = {
    "user_input": "summarise given text\nThe company reported an 8% rise in Q3 2024...",
    "response": "The company experienced an 8% increase in Q3 2024...",
    "reference": "The company reported an 8% growth in Q3 2024..."
}

# Evaluate with non-LLM metric
bleu_score = evaluator.evaluate_single_sample_non_llm(sample_data)
print(f"BleuScore: {bleu_score}")

# Evaluate with LLM-based metric
async def evaluate_llm():
    llm_sample = {k: v for k, v in sample_data.items() if k != 'reference'}
    aspect_score = await evaluator.evaluate_single_sample_llm(llm_sample)
    print(f"AspectCritic Score: {aspect_score}")

asyncio.run(evaluate_llm())
```

### Dataset Evaluation

```python
import asyncio
from ragas_summary_eval import RagasSummaryEvaluator

async def evaluate_dataset():
    evaluator = RagasSummaryEvaluator()
    
    # Option 1: Use sample data
    dataset = evaluator.create_evaluation_dataset_from_samples()
    
    # Option 2: Load from Hugging Face
    # dataset = evaluator.load_dataset_from_hf("explodinggradients/earning_report_summary")
    
    # Evaluate dataset
    results = await evaluator.evaluate_dataset(dataset)
    
    # Export results
    df = evaluator.export_results_to_pandas(results)
    evaluator.save_results_to_csv(results, "my_evaluation_results.csv")
    
    return results

results = asyncio.run(evaluate_dataset())
```

## Metrics Explained

### 1. BleuScore (Non-LLM Metric)

- **What it measures**: Text similarity between generated summary and reference
- **Pros**: Fast, no API costs, objective
- **Cons**: May not capture semantic meaning well
- **Use case**: Quick baseline evaluation

### 2. AspectCritic (LLM-based Metric)

- **What it measures**: Whether summary accurately captures key details
- **Pros**: More nuanced, understands context
- **Cons**: Requires API calls, subjective
- **Use case**: Production evaluation, quality assessment

## Example Output

```
🚀 Ragas Summary Evaluation Demo
==================================================

1. Non-LLM Metric Evaluation (BleuScore)
----------------------------------------
BleuScore: 0.137
Input: summarise given text\nThe company reported an 8% rise in Q3 2024, driven by strong performance in the Asian market...
Response: The company experienced an 8% increase in Q3 2024, largely due to effective marketing strategies...
Reference: The company reported an 8% growth in Q3 2024, primarily driven by strong sales in the Asian market...

2. LLM-based Metric Evaluation (AspectCritic)
---------------------------------------------
AspectCritic Score: 1 (PASS)
Input: summarise given text\nThe company reported an 8% rise in Q3 2024, driven by strong performance in the Asian market...
Response: The company experienced an 8% increase in Q3 2024, largely due to effective marketing strategies...

3. Dataset Evaluation
--------------------
Evaluating dataset with 3 samples...
Dataset features: ['user_input', 'response']
Overall Results: {'summary_accuracy': 0.84}

Sample-level results (first 3 rows):
    user_input                                          response                                            summary_accuracy
0   summarise given text\nThe Q2 earnings report r...   The Q2 earnings report showed a 15% revenue in...   1
1   summarise given text\nIn 2023, North American ...   Companies are strategizing to adapt to market ...   1
2   summarise given text\nIn 2022, European expans...   Many companies experienced a notable 15% growt...   1

Results saved to summary_evaluation_results.csv

✅ Evaluation completed successfully!
```

## Customization

### Custom Evaluation Criteria

```python
# Modify the AspectCritic definition
evaluator.aspect_critic_metric = AspectCritic(
    name="custom_summary_accuracy",
    llm=evaluator.evaluator_llm,
    definition="Your custom evaluation criteria here..."
)
```

### Custom Sample Data

```python
# Create your own test data
custom_data = [
    {
        "user_input": "Your input text here...",
        "response": "Your generated summary here...",
        "reference": "Your reference summary here..."  # Optional for LLM metrics
    }
]

# Create dataset from custom data
dataset = EvaluationDataset.from_dict({
    'user_input': [item['user_input'] for item in custom_data],
    'response': [item['response'] for item in custom_data]
})
```

## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   ```
   ValueError: OpenAI API key not found
   ```
   **Solution**: Set your `OPENAI_API_KEY` environment variable

2. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'ragas'
   ```
   **Solution**: Install Ragas: `pip install ragas`

3. **Dataset Loading Errors**
   ```
   Error loading dataset from Hugging Face
   ```
   **Solution**: The script will fall back to sample data automatically

### Performance Tips

- Use non-LLM metrics for quick iterations
- Use LLM-based metrics for final evaluation
- Cache results when evaluating large datasets
- Consider using batch processing for large datasets

## References

- [Ragas Documentation](https://docs.ragas.io/en/latest/getstarted/evals/#evaluating-on-a-dataset)
- [Ragas GitHub Repository](https://github.com/explodinggradients/ragas)
- [LangChain Integration](https://python.langchain.com/docs/integrations/llms/openai)

## License

This implementation follows the same license as the main project. 