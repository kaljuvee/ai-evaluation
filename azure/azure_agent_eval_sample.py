from azure.ai.evaluation import (
    evaluate, RelevanceEvaluator, CoherenceEvaluator, FluencyEvaluator,
    GroundednessEvaluator, QAEvaluator, ContentSafetyEvaluator,
    ViolenceEvaluator, SexualEvaluator, SelfHarmEvaluator, HateUnfairnessEvaluator
)
from azure.ai.evaluation import AzureAIConfig
import json

# Initialize Azure AI configuration
# You'll need to set up your Azure AI project and get the necessary credentials
azure_ai_config = AzureAIConfig(
    project_name="your-project-name",
    subscription_id="your-subscription-id",
    resource_group="your-resource-group",
    workspace_name="your-workspace-name"
)

def create_sample_data():
    """Create sample data for evaluation"""
    sample_data = [
        {
            "query": "What is the capital of France?",
            "response": "Paris is the capital of France.",
            "context": "Paris has been the capital of France since the 10th century and is known for its cultural and historical landmarks.",
            "ground_truth": "Paris"
        },
        {
            "query": "Who developed the theory of relativity?",
            "response": "Albert Einstein developed the theory of relativity.",
            "context": "Albert Einstein developed the theory of relativity, with his special relativity published in 1905 and general relativity in 1915.",
            "ground_truth": "Albert Einstein"
        },
        {
            "query": "What is the speed of light?",
            "response": "The speed of light is approximately 299,792,458 meters per second.",
            "context": "The exact speed of light in a vacuum is 299,792,458 meters per second, a constant used in physics to represent 'c'.",
            "ground_truth": "299,792,458 meters per second"
        }
    ]
    
    # Write to JSONL file
    with open("sample_data.jsonl", "w") as f:
        for item in sample_data:
            f.write(json.dumps(item) + "\n")
    
    return "sample_data.jsonl"

def run_agent_evaluation():
    """Run comprehensive agent evaluation"""
    
    # Create sample data
    data_file = create_sample_data()
    
    # Initialize evaluators
    relevance_eval = RelevanceEvaluator()
    coherence_eval = CoherenceEvaluator()
    fluency_eval = FluencyEvaluator()
    groundedness_eval = GroundednessEvaluator()
    qa_eval = QAEvaluator()
    
    # Safety evaluators
    violence_eval = ViolenceEvaluator()
    sexual_eval = SexualEvaluator()
    self_harm_eval = SelfHarmEvaluator()
    hate_unfairness_eval = HateUnfairnessEvaluator()
    content_safety_eval = ContentSafetyEvaluator()
    
    # Run evaluation
    result = evaluate(
        data=data_file,
        evaluators={
            "relevance": relevance_eval,
            "coherence": coherence_eval,
            "fluency": fluency_eval,
            "groundedness": groundedness_eval,
            "qa": qa_eval,
            "violence": violence_eval,
            "sexual": sexual_eval,
            "self_harm": self_harm_eval,
            "hate_unfairness": hate_unfairness_eval,
            "content_safety": content_safety_eval
        },
        evaluator_config={
            "default": {
                "column_mapping": {
                    "query": "${data.query}",
                    "response": "${data.response}",
                    "context": "${data.context}",
                    "ground_truth": "${data.ground_truth}"
                }
            }
        },
        azure_ai_project=azure_ai_config,
        output_path="./agent_eval_results.json"
    )
    
    print("Evaluation completed!")
    print(f"Results saved to: ./agent_eval_results.json")
    print(f"Studio URL: {result.studio_url}")
    
    # Print summary metrics
    print("\nSummary Metrics:")
    for metric, value in result.metrics.items():
        print(f"{metric}: {value}")

def run_single_evaluation():
    """Run evaluation on a single query-response pair"""
    
    from azure.ai.evaluation import RelevanceEvaluator
    
    # Initialize evaluator
    relevance_eval = RelevanceEvaluator()
    
    # Single evaluation
    query = "What is the capital of France?"
    response = "Paris is the capital of France."
    
    result = relevance_eval(query=query, response=response)
    print(f"Single evaluation result: {result}")

if __name__ == "__main__":
    print("Running Azure AI Agent Evaluation...")
    
    # Run single evaluation first
    print("\n1. Single Evaluation:")
    run_single_evaluation()
    
    # Run comprehensive evaluation
    print("\n2. Comprehensive Evaluation:")
    run_agent_evaluation()
