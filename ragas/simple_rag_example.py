"""
Ragas Simple RAG Evaluation Example

This example demonstrates how to evaluate a RAG (Retrieval-Augmented Generation) system
using Ragas metrics. It shows all 6 core metrics with clear explanations.

Based on: ragas_readme.md
"""

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_relevancy,
    answer_correctness
)
from datasets import Dataset
import os


def main():
    print("=" * 80)
    print("Ragas Simple RAG Evaluation Example")
    print("=" * 80)
    print()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠ WARNING: OPENAI_API_KEY not found in environment variables")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        print()
        return

    # Step 1: Prepare evaluation data
    print("Step 1: Preparing Evaluation Data")
    print("-" * 80)
    print()
    
    # Simulated RAG system outputs
    # In a real scenario, these would come from your actual RAG system
    data = {
        "question": [
            "What is machine learning?",
            "How does photosynthesis work?",
            "What are the primary colors?"
        ],
        "answer": [
            # Answer 1: Good answer, faithful to context
            "Machine learning is a subset of AI that enables systems to learn from data.",
            
            # Answer 2: Good answer with all key points
            "Photosynthesis is the process where plants convert sunlight into energy using chlorophyll.",
            
            # Answer 3: Perfect answer
            "The primary colors are red, blue, and yellow."
        ],
        "contexts": [
            # Context 1: Relevant context for ML question
            ["Machine learning is a branch of artificial intelligence that focuses on learning from data."],
            
            # Context 2: Multiple relevant documents for photosynthesis
            [
                "Photosynthesis converts light energy into chemical energy in plants.",
                "Chlorophyll is the green pigment that captures sunlight."
            ],
            
            # Context 3: Relevant context for primary colors
            [
                "Primary colors are red, blue, and yellow.",
                "They cannot be created by mixing other colors."
            ]
        ],
        "ground_truth": [
            # Ground truth 1: Ideal answer for ML
            "Machine learning is a subset of artificial intelligence that allows systems to learn and improve from experience without being explicitly programmed.",
            
            # Ground truth 2: Ideal answer for photosynthesis
            "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of sugar.",
            
            # Ground truth 3: Ideal answer for primary colors
            "The primary colors are red, blue, and yellow."
        ]
    }
    
    print("✓ Created evaluation data with 3 test cases")
    for i, q in enumerate(data["question"], 1):
        print(f"  Test {i}: {q}")
    print()

    # Step 2: Create dataset
    print("Step 2: Creating Ragas Dataset")
    print("-" * 80)
    dataset = Dataset.from_dict(data)
    print(f"✓ Dataset created with {len(data['question'])} examples")
    print()

    # Step 3: Define metrics
    print("Step 3: Understanding the Metrics")
    print("-" * 80)
    print()
    
    metrics_info = {
        "faithfulness": {
            "emoji": "📝",
            "question": "Does the answer stick to the facts in the retrieved documents?",
            "needs_ground_truth": False
        },
        "answer_relevancy": {
            "emoji": "🎯",
            "question": "Does the answer actually address the user's question?",
            "needs_ground_truth": False
        },
        "context_precision": {
            "emoji": "🔍",
            "question": "Are the retrieved documents actually relevant to the question?",
            "needs_ground_truth": True
        },
        "context_recall": {
            "emoji": "📚",
            "question": "Did we retrieve all the relevant information needed?",
            "needs_ground_truth": True
        },
        "context_relevancy": {
            "emoji": "🎪",
            "question": "How much of the retrieved context is actually useful?",
            "needs_ground_truth": False
        },
        "answer_correctness": {
            "emoji": "✅",
            "question": "Is the answer factually correct compared to ground truth?",
            "needs_ground_truth": True
        }
    }
    
    for metric_name, info in metrics_info.items():
        gt_marker = "✅ Needs ground truth" if info["needs_ground_truth"] else "❌ No ground truth needed"
        print(f"{info['emoji']} {metric_name.upper()}")
        print(f"   Question: {info['question']}")
        print(f"   {gt_marker}")
        print()

    # Step 4: Run evaluation
    print("Step 4: Running Evaluation")
    print("-" * 80)
    print("Evaluating with all 6 metrics... (this may take a moment)")
    print()
    
    try:
        results = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                context_relevancy,
                answer_correctness
            ]
        )
        
        # Step 5: Display results
        print("Step 5: Results")
        print("=" * 80)
        print()
        
        print("Overall Scores:")
        print("-" * 80)
        for metric_name in metrics_info.keys():
            if metric_name in results:
                score = results[metric_name]
                emoji = metrics_info[metric_name]["emoji"]
                
                # Determine quality level
                if score >= 0.9:
                    quality = "Excellent ✅"
                elif score >= 0.7:
                    quality = "Good ✓"
                elif score >= 0.5:
                    quality = "Acceptable ⚠"
                else:
                    quality = "Needs Improvement ❌"
                
                print(f"{emoji} {metric_name:20s}: {score:.3f} - {quality}")
        
        print()
        print("=" * 80)
        print()
        
        # Interpretation guide
        print("Score Interpretation Guide:")
        print("-" * 80)
        print("0.9-1.0: Excellent - System is performing very well")
        print("0.7-0.9: Good - Minor improvements possible")
        print("0.5-0.7: Acceptable - Needs attention")
        print("0.3-0.5: Poor - Significant issues")
        print("0.0-0.3: Very Poor - Major problems, requires immediate action")
        print()
        
        # Metric-specific insights
        print("What Each Score Means:")
        print("-" * 80)
        
        if 'faithfulness' in results and results['faithfulness'] < 0.7:
            print("⚠ Low Faithfulness: Your AI might be hallucinating or adding unsupported information")
        
        if 'answer_relevancy' in results and results['answer_relevancy'] < 0.7:
            print("⚠ Low Answer Relevancy: Answers might be going off-topic")
        
        if 'context_precision' in results and results['context_precision'] < 0.7:
            print("⚠ Low Context Precision: Retrieval system is returning irrelevant documents")
        
        if 'context_recall' in results and results['context_recall'] < 0.7:
            print("⚠ Low Context Recall: Important information is being missed during retrieval")
        
        print()
        
        # Recommendations
        print("Recommendations Based on Scores:")
        print("-" * 80)
        
        avg_score = sum(results[m] for m in metrics_info.keys() if m in results) / len([m for m in metrics_info.keys() if m in results])
        
        if avg_score >= 0.8:
            print("✅ Your RAG system is performing well! Minor optimizations may help.")
        elif avg_score >= 0.6:
            print("⚠ Your RAG system needs improvement. Focus on metrics below 0.7.")
        else:
            print("❌ Your RAG system needs significant work. Review retrieval and generation components.")
        
        print()
        print("=" * 80)
        print("Example Complete!")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        print()
        print("Common issues:")
        print("- Missing OPENAI_API_KEY environment variable")
        print("- Network connectivity problems")
        print("- API rate limits")
        print()
        raise


def example_without_ground_truth():
    """
    Example showing evaluation WITHOUT ground truth.
    Only metrics that don't require ground truth can be used.
    """
    print("=" * 80)
    print("Ragas Evaluation WITHOUT Ground Truth")
    print("=" * 80)
    print()
    
    # Data without ground_truth field
    data = {
        "question": ["What is Python?"],
        "answer": ["Python is a high-level programming language known for its simplicity."],
        "contexts": [["Python is a programming language created by Guido van Rossum."]]
        # No ground_truth provided!
    }
    
    dataset = Dataset.from_dict(data)
    
    print("Metrics available WITHOUT ground truth:")
    print("  ✓ Faithfulness")
    print("  ✓ Answer Relevancy")
    print("  ✓ Context Relevancy")
    print()
    
    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_relevancy
        ]
    )
    
    print("Results:")
    for metric, score in results.items():
        print(f"  {metric}: {score:.3f}")
    print()


if __name__ == "__main__":
    # Run main example with ground truth
    main()
    
    # Uncomment to see example without ground truth
    # print("\n\n")
    # example_without_ground_truth()
