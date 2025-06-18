# Azure AI Evaluation Samples

This directory contains sample scripts for evaluating AI applications using the Azure AI Evaluation SDK.

## Files

- `azure_agent_eval_sample.py` - Comprehensive agent evaluation with multiple evaluators
- `azure_rag_eval_sample.py` - RAG-specific evaluation focusing on retrieval and groundedness
- `synthetic_data_generator.py` - Generate synthetic test data for evaluation
- `requirements.txt` - Required dependencies

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure Azure AI project:
   - Update the `AzureAIConfig` in each script with your project details
   - Set up authentication (service principal or managed identity)

## Usage

### 1. Generate Synthetic Data
```bash
python synthetic_data_generator.py
```
This creates:
- `synthetic_qa_data.jsonl` - Q&A pairs for evaluation
- `synthetic_conversation_data.jsonl` - Multi-turn conversations

### 2. Run Agent Evaluation
```bash
python azure_agent_eval_sample.py
```
Evaluates:
- Relevance, coherence, fluency
- Groundedness and QA accuracy
- Content safety (violence, sexual content, etc.)

### 3. Run RAG Evaluation
```bash
python azure_rag_eval_sample.py
```
Evaluates:
- Retrieval accuracy
- Document retrieval
- Groundedness (standard and pro)
- Response completeness
- Text similarity metrics (F1, ROUGE, BLEU)

## Configuration

Update the Azure AI configuration in each script:

```python
azure_ai_config = AzureAIConfig(
    project_name="your-project-name",
    subscription_id="your-subscription-id",
    resource_group="your-resource-group",
    workspace_name="your-workspace-name"
)
```

## Data Format

The scripts expect JSONL format with the following fields:
- `query`: User question
- `response`: AI-generated response
- `context`: Source documents/context
- `ground_truth`: Expected answer

## Output

Results are saved as JSON files and can be viewed in the Azure AI Studio portal using the provided URL.

## References

- [Azure AI Evaluation SDK Documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/evaluate-sdk)
- [Simulator for Synthetic Data](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/simulator-interaction-data) 