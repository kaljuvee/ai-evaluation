from azure.ai.evaluation import (
    evaluate, RetrievalEvaluator, DocumentRetrievalEvaluator,
    GroundednessEvaluator, GroundednessProEvaluator, RelevanceEvaluator,
    ResponseCompletenessEvaluator, SimilarityEvaluator, F1ScoreEvaluator,
    RougeScoreEvaluator, BleuScoreEvaluator
)
from azure.ai.evaluation import AzureAIConfig
import json

# Initialize Azure AI configuration
azure_ai_config = AzureAIConfig(
    project_name="your-project-name",
    subscription_id="your-subscription-id",
    resource_group="your-resource-group",
    workspace_name="your-workspace-name"
)

def create_rag_sample_data():
    """Create sample RAG data for evaluation"""
    sample_data = [
        {
            "query": "What are the benefits of renewable energy?",
            "response": "Renewable energy sources like solar and wind power offer several benefits including reduced greenhouse gas emissions, lower operating costs, and energy independence.",
            "context": "Renewable energy sources such as solar, wind, and hydroelectric power provide clean alternatives to fossil fuels. They help reduce greenhouse gas emissions and can lower energy costs over time. Solar panels and wind turbines are becoming increasingly cost-effective and efficient.",
            "ground_truth": "Renewable energy provides clean power, reduces emissions, and can lower costs while promoting energy independence."
        },
        {
            "query": "How does machine learning work?",
            "response": "Machine learning uses algorithms to learn patterns from data and make predictions or decisions without being explicitly programmed for each task.",
            "context": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses statistical techniques to give computers the ability to learn from data and make predictions or decisions.",
            "ground_truth": "Machine learning algorithms learn patterns from data to make predictions without explicit programming."
        },
        {
            "query": "What is the impact of climate change on oceans?",
            "response": "Climate change affects oceans through rising sea levels, ocean acidification, and changes in marine ecosystems and biodiversity.",
            "context": "Climate change has significant impacts on the world's oceans. Rising global temperatures cause sea levels to rise due to thermal expansion and melting ice. Increased CO2 levels lead to ocean acidification, which affects marine life. Changes in temperature and acidity disrupt marine ecosystems and biodiversity.",
            "ground_truth": "Climate change impacts oceans through sea level rise, acidification, and ecosystem disruption."
        }
    ]
    
    # Write to JSONL file
    with open("rag_sample_data.jsonl", "w") as f:
        for item in sample_data:
            f.write(json.dumps(item) + "\n")
    
    return "rag_sample_data.jsonl"

def run_rag_evaluation():
    """Run comprehensive RAG evaluation"""
    
    # Create sample data
    data_file = create_rag_sample_data()
    
    # Initialize RAG-specific evaluators
    retrieval_eval = RetrievalEvaluator()
    document_retrieval_eval = DocumentRetrievalEvaluator()
    groundedness_eval = GroundednessEvaluator()
    groundedness_pro_eval = GroundednessProEvaluator()
    relevance_eval = RelevanceEvaluator()
    response_completeness_eval = ResponseCompletenessEvaluator()
    
    # Text similarity evaluators
    similarity_eval = SimilarityEvaluator()
    f1_score_eval = F1ScoreEvaluator()
    rouge_eval = RougeScoreEvaluator()
    bleu_eval = BleuScoreEvaluator()
    
    # Run evaluation
    result = evaluate(
        data=data_file,
        evaluators={
            "retrieval": retrieval_eval,
            "document_retrieval": document_retrieval_eval,
            "groundedness": groundedness_eval,
            "groundedness_pro": groundedness_pro_eval,
            "relevance": relevance_eval,
            "response_completeness": response_completeness_eval,
            "similarity": similarity_eval,
            "f1_score": f1_score_eval,
            "rouge": rouge_eval,
            "bleu": bleu_eval
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
        output_path="./rag_eval_results.json"
    )
    
    print("RAG Evaluation completed!")
    print(f"Results saved to: ./rag_eval_results.json")
    print(f"Studio URL: {result.studio_url}")
    
    # Print summary metrics
    print("\nRAG Evaluation Summary Metrics:")
    for metric, value in result.metrics.items():
        print(f"{metric}: {value}")

def run_groundedness_evaluation():
    """Run focused groundedness evaluation"""
    
    groundedness_eval = GroundednessEvaluator()
    
    # Test groundedness with context
    query = "What is the capital of France?"
    response = "Paris is the capital of France."
    context = "Paris has been the capital of France since the 10th century and is known for its cultural and historical landmarks."
    
    result = groundedness_eval(
        query=query,
        response=response,
        context=context
    )
    
    print(f"Groundedness evaluation result: {result}")

def run_retrieval_evaluation():
    """Run retrieval evaluation"""
    
    retrieval_eval = RetrievalEvaluator()
    
    # Test retrieval with conversation format
    conversation = {
        "messages": [
            {
                "content": "What are the benefits of renewable energy?",
                "role": "user"
            },
            {
                "content": "Renewable energy sources like solar and wind power offer several benefits including reduced greenhouse gas emissions, lower operating costs, and energy independence.",
                "role": "assistant",
                "context": "Renewable energy sources such as solar, wind, and hydroelectric power provide clean alternatives to fossil fuels. They help reduce greenhouse gas emissions and can lower energy costs over time."
            }
        ]
    }
    
    result = retrieval_eval(conversation=conversation)
    print(f"Retrieval evaluation result: {result}")

if __name__ == "__main__":
    print("Running Azure AI RAG Evaluation...")
    
    # Run individual evaluations
    print("\n1. Groundedness Evaluation:")
    run_groundedness_evaluation()
    
    print("\n2. Retrieval Evaluation:")
    run_retrieval_evaluation()
    
    # Run comprehensive RAG evaluation
    print("\n3. Comprehensive RAG Evaluation:")
    run_rag_evaluation()
