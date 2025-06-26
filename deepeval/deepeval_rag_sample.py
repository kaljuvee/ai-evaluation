"""
DeepEval RAG Evaluation Sample

This sample demonstrates how to use DeepEval to evaluate RAG (Retrieval-Augmented Generation) pipelines
with various metrics including faithfulness, answer relevancy, context relevancy, and custom evaluations.

Key Features Demonstrated:
- LLMTestCase for single query-response evaluation
- ConversationalTestCase for multi-turn interactions
- Multiple evaluation metrics (Faithfulness, Answer Relevancy, Context Relevancy, G-Eval)
- Custom metric creation
- Synthetic dataset generation
- Batch evaluation
- Integration with different RAG frameworks
"""

import os
import pytest
from typing import List, Dict, Any
from deepeval import assert_test, evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    GEval,
    HallucinationMetric,
    ToxicityMetric
)
from deepeval.test_case import LLMTestCase, ConversationalTestCase
from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset


class CustomRAGMetric(GEval):
    """Custom metric for RAG-specific evaluation"""
    
    def __init__(self):
        super().__init__(
            name="RAG Quality",
            criteria="""
            Evaluate the RAG response quality based on:
            1. Accuracy: Is the information factually correct?
            2. Completeness: Does it answer the full question?
            3. Context Usage: Does it properly use the provided context?
            4. Clarity: Is the response clear and well-structured?
            
            Score from 0-1 where:
            0.0-0.3: Poor quality, inaccurate or incomplete
            0.4-0.6: Acceptable quality with some issues
            0.7-0.8: Good quality, mostly accurate and complete
            0.9-1.0: Excellent quality, accurate, complete, and well-structured
            """,
            evaluation_params=["actual_output", "expected_output", "retrieval_context"],
            threshold=0.7
        )


def create_sample_rag_data() -> List[Dict[str, Any]]:
    """Create sample RAG test data"""
    return [
        {
            "query": "What are the benefits of using RAG systems?",
            "expected_output": "RAG systems provide several benefits including improved accuracy through context retrieval, reduced hallucinations, and the ability to access up-to-date information from external sources.",
            "actual_output": "RAG systems offer improved accuracy by retrieving relevant context, help reduce hallucinations, and allow access to current information from external knowledge bases.",
            "retrieval_context": [
                "RAG (Retrieval-Augmented Generation) systems combine the power of large language models with external knowledge retrieval.",
                "Key benefits include improved accuracy through context retrieval, reduced hallucinations, and access to up-to-date information.",
                "RAG systems can access external knowledge bases, databases, and documents to provide more accurate responses."
            ]
        },
        {
            "query": "How does vector similarity work in RAG?",
            "expected_output": "Vector similarity in RAG works by converting text into numerical vectors and finding the most similar vectors using distance metrics like cosine similarity.",
            "actual_output": "Vector similarity converts text to numerical vectors and finds the most similar ones using distance metrics such as cosine similarity.",
            "retrieval_context": [
                "Vector similarity is a core component of RAG systems.",
                "Text is converted into numerical vectors using embedding models.",
                "Similarity is calculated using distance metrics like cosine similarity or Euclidean distance.",
                "The most similar vectors are retrieved as relevant context."
            ]
        },
        {
            "query": "What is the difference between RAG and fine-tuning?",
            "expected_output": "RAG retrieves external context at inference time while fine-tuning updates model weights during training. RAG is more flexible for dynamic information while fine-tuning requires retraining for new data.",
            "actual_output": "RAG retrieves external context during inference, while fine-tuning updates the model's weights during training. RAG is more flexible for dynamic information.",
            "retrieval_context": [
                "RAG and fine-tuning are two different approaches to improving LLM performance.",
                "RAG retrieves external context at inference time without changing model weights.",
                "Fine-tuning updates the model's weights during training on specific data.",
                "RAG is more flexible for dynamic information while fine-tuning requires retraining."
            ]
        }
    ]


def create_conversational_data() -> List[Dict[str, Any]]:
    """Create sample conversational test data"""
    return [
        {
            "messages": [
                {"role": "user", "content": "Tell me about RAG systems"},
                {"role": "assistant", "content": "RAG systems combine language models with external knowledge retrieval to improve accuracy and reduce hallucinations."},
                {"role": "user", "content": "What are their main advantages?"},
                {"role": "assistant", "content": "The main advantages include improved accuracy through context retrieval, reduced hallucinations, and access to up-to-date information from external sources."}
            ],
            "expected_output": "The main advantages include improved accuracy through context retrieval, reduced hallucinations, and access to up-to-date information from external sources.",
            "retrieval_context": [
                "RAG systems provide improved accuracy through context retrieval.",
                "They help reduce hallucinations by grounding responses in retrieved information.",
                "RAG systems can access external knowledge bases for current information."
            ]
        }
    ]


def test_rag_faithfulness():
    """Test RAG response faithfulness to provided context"""
    faithfulness_metric = FaithfulnessMetric(threshold=0.7)
    
    test_cases = []
    sample_data = create_sample_rag_data()
    
    for data in sample_data:
        test_case = LLMTestCase(
            input=data["query"],
            actual_output=data["actual_output"],
            expected_output=data["expected_output"],
            retrieval_context=data["retrieval_context"]
        )
        test_cases.append(test_case)
    
    # Run evaluation
    results = evaluate(test_cases, [faithfulness_metric])
    print(f"Faithfulness Evaluation Results: {results}")
    
    # Assert individual test cases
    for test_case in test_cases:
        assert_test(test_case, [faithfulness_metric])


def test_rag_answer_relevancy():
    """Test RAG response relevancy to the query"""
    relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
    
    test_cases = []
    sample_data = create_sample_rag_data()
    
    for data in sample_data:
        test_case = LLMTestCase(
            input=data["query"],
            actual_output=data["actual_output"],
            expected_output=data["expected_output"],
            retrieval_context=data["retrieval_context"]
        )
        test_cases.append(test_case)
    
    results = evaluate(test_cases, [relevancy_metric])
    print(f"Answer Relevancy Evaluation Results: {results}")
    
    for test_case in test_cases:
        assert_test(test_case, [relevancy_metric])


def test_rag_context_relevancy():
    """Test if retrieved context is relevant to the query"""
    context_relevancy_metric = ContextualRelevancyMetric(threshold=0.7)
    
    test_cases = []
    sample_data = create_sample_rag_data()
    
    for data in sample_data:
        test_case = LLMTestCase(
            input=data["query"],
            actual_output=data["actual_output"],
            expected_output=data["expected_output"],
            retrieval_context=data["retrieval_context"]
        )
        test_cases.append(test_case)
    
    results = evaluate(test_cases, [context_relevancy_metric])
    print(f"Context Relevancy Evaluation Results: {results}")
    
    for test_case in test_cases:
        assert_test(test_case, [context_relevancy_metric])


def test_custom_rag_metric():
    """Test using custom RAG evaluation metric"""
    custom_metric = CustomRAGMetric()
    
    test_cases = []
    sample_data = create_sample_rag_data()
    
    for data in sample_data:
        test_case = LLMTestCase(
            input=data["query"],
            actual_output=data["actual_output"],
            expected_output=data["expected_output"],
            retrieval_context=data["retrieval_context"]
        )
        test_cases.append(test_case)
    
    results = evaluate(test_cases, [custom_metric])
    print(f"Custom RAG Metric Evaluation Results: {results}")
    
    for test_case in test_cases:
        assert_test(test_case, [custom_metric])


def test_conversational_rag():
    """Test conversational RAG interactions"""
    faithfulness_metric = FaithfulnessMetric(threshold=0.7)
    relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
    
    conversational_data = create_conversational_data()
    
    for data in conversational_data:
        test_case = ConversationalTestCase(
            messages=data["messages"],
            expected_output=data["expected_output"],
            retrieval_context=data["retrieval_context"]
        )
        
        # Test both metrics
        assert_test(test_case, [faithfulness_metric, relevancy_metric])


def test_hallucination_detection():
    """Test hallucination detection in RAG responses"""
    hallucination_metric = HallucinationMetric(threshold=0.3)  # Lower threshold for hallucination
    
    # Test case with potential hallucination
    test_case = LLMTestCase(
        input="What is the latest version of Python?",
        actual_output="The latest version of Python is 3.12.5, released in March 2024.",
        expected_output="The latest version of Python is 3.12.5, released in March 2024.",
        retrieval_context=[
            "Python is a programming language.",
            "Python 3.12 was released in October 2023.",
            "Python versions are released regularly."
        ]
    )
    
    assert_test(test_case, [hallucination_metric])


def test_toxicity_detection():
    """Test toxicity detection in RAG responses"""
    toxicity_metric = ToxicityMetric(threshold=0.1)  # Very low threshold for toxicity
    
    # Test case with non-toxic content
    test_case = LLMTestCase(
        input="What is machine learning?",
        actual_output="Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
        expected_output="Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
        retrieval_context=[
            "Machine learning is a subset of artificial intelligence.",
            "It enables computers to learn from experience.",
            "ML algorithms improve performance over time."
        ]
    )
    
    assert_test(test_case, [toxicity_metric])


def test_synthetic_dataset_generation():
    """Demonstrate synthetic dataset generation for RAG evaluation"""
    synthesizer = Synthesizer()
    
    # Generate synthetic test cases for RAG evaluation
    synthetic_test_cases = synthesizer.generate_test_cases(
        num_test_cases=5,
        test_case_type="rag",
        topic="artificial intelligence and machine learning",
        include_context=True
    )
    
    print(f"Generated {len(synthetic_test_cases)} synthetic test cases")
    
    # Evaluate synthetic test cases
    faithfulness_metric = FaithfulnessMetric(threshold=0.7)
    relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
    
    results = evaluate(synthetic_test_cases, [faithfulness_metric, relevancy_metric])
    print(f"Synthetic Dataset Evaluation Results: {results}")


def test_comprehensive_rag_evaluation():
    """Comprehensive RAG evaluation with multiple metrics"""
    # Define multiple metrics
    metrics = [
        FaithfulnessMetric(threshold=0.7),
        AnswerRelevancyMetric(threshold=0.7),
        ContextualRelevancyMetric(threshold=0.7),
        CustomRAGMetric(),
        HallucinationMetric(threshold=0.3),
        ToxicityMetric(threshold=0.1)
    ]
    
    # Create test cases
    test_cases = []
    sample_data = create_sample_rag_data()
    
    for data in sample_data:
        test_case = LLMTestCase(
            input=data["query"],
            actual_output=data["actual_output"],
            expected_output=data["expected_output"],
            retrieval_context=data["retrieval_context"]
        )
        test_cases.append(test_case)
    
    # Run comprehensive evaluation
    results = evaluate(test_cases, metrics)
    print(f"Comprehensive RAG Evaluation Results: {results}")
    
    # Print detailed results
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"Query: {test_case.input}")
        print(f"Expected: {test_case.expected_output}")
        print(f"Actual: {test_case.actual_output}")
        print(f"Context: {test_case.retrieval_context}")


def create_evaluation_dataset():
    """Create and save an evaluation dataset"""
    dataset = EvaluationDataset()
    
    # Add test cases to dataset
    sample_data = create_sample_rag_data()
    for data in sample_data:
        test_case = LLMTestCase(
            input=data["query"],
            actual_output=data["actual_output"],
            expected_output=data["expected_output"],
            retrieval_context=data["retrieval_context"]
        )
        dataset.add_test_case(test_case)
    
    # Save dataset
    dataset.save("rag_evaluation_dataset.json")
    print("Evaluation dataset saved as 'rag_evaluation_dataset.json'")


def main():
    """Run all RAG evaluation examples"""
    print("🚀 DeepEval RAG Evaluation Sample")
    print("=" * 50)
    
    try:
        # Test individual metrics
        print("\n1. Testing Faithfulness...")
        test_rag_faithfulness()
        
        print("\n2. Testing Answer Relevancy...")
        test_rag_answer_relevancy()
        
        print("\n3. Testing Context Relevancy...")
        test_rag_context_relevancy()
        
        print("\n4. Testing Custom RAG Metric...")
        test_custom_rag_metric()
        
        print("\n5. Testing Conversational RAG...")
        test_conversational_rag()
        
        print("\n6. Testing Hallucination Detection...")
        test_hallucination_detection()
        
        print("\n7. Testing Toxicity Detection...")
        test_toxicity_detection()
        
        print("\n8. Testing Synthetic Dataset Generation...")
        test_synthetic_dataset_generation()
        
        print("\n9. Running Comprehensive Evaluation...")
        test_comprehensive_rag_evaluation()
        
        print("\n10. Creating Evaluation Dataset...")
        create_evaluation_dataset()
        
        print("\n✅ All evaluations completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        print("\n💡 Make sure you have:")
        print("   - Installed deepeval: pip install deepeval")
        print("   - Set up your OpenAI API key: export OPENAI_API_KEY='your-key'")
        print("   - For cloud features: deepeval login")


if __name__ == "__main__":
    main()
