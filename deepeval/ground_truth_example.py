"""
DeepEval Ground Truth Example

This example demonstrates how to use ground truth (expected output) with DeepEval's G-Eval metric.
It shows the complete workflow from defining metrics to evaluating test cases with different quality levels.

Based on: deepeval_readme.md
"""

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.dataset import EvaluationDataset


def main():
    print("=" * 80)
    print("DeepEval Ground Truth Example")
    print("=" * 80)
    print()

    # Step 1: Define the correctness metric using G-Eval
    # This metric compares actual output against expected output (ground truth)
    print("Step 1: Defining Correctness Metric")
    print("-" * 80)
    
    correctness_metric = GEval(
        name="Correctness",
        model="gpt-4o-mini",  # The LLM judge that evaluates outputs
        evaluation_steps=[
            "Check whether the facts in 'actual output' contradict any facts in 'expected output'",
            "Lightly penalize omission of detail, focus on the main idea",
            "Vague language or contradicting opinions are OK"
        ],
        evaluation_params=[
            LLMTestCaseParams.EXPECTED_OUTPUT,  # Ground truth
            LLMTestCaseParams.ACTUAL_OUTPUT     # What the AI produced
        ],
        threshold=0.7  # Minimum score to pass (0.0 to 1.0)
    )
    print(f"✓ Created metric: {correctness_metric.name}")
    print(f"  Threshold: {correctness_metric.threshold}")
    print()

    # Step 2: Create test cases with different quality levels
    print("Step 2: Creating Test Cases")
    print("-" * 80)
    
    test_cases = [
        # Test Case 1: Perfect match (Score: 1.0)
        LLMTestCase(
            input="What are the main causes of deforestation?",
            actual_output="The main causes of deforestation include agricultural expansion, logging, infrastructure development, and urbanization.",
            expected_output="The main causes of deforestation include agricultural expansion, logging, infrastructure development, and urbanization."
        ),
        
        # Test Case 2: Partial match - missing details (Score: ~0.5-0.7)
        LLMTestCase(
            input="Define the term 'artificial intelligence'.",
            actual_output="Artificial intelligence is the simulation of human intelligence by machines.",
            expected_output="Artificial intelligence refers to the simulation of human intelligence in machines that are programmed to think and learn like humans, including tasks such as problem-solving, decision-making, and language understanding."
        ),
        
        # Test Case 3: Factually incorrect (Score: 0.0)
        LLMTestCase(
            input="List the primary colors.",
            actual_output="The primary colors are green, orange, and purple.",
            expected_output="The primary colors are red, blue, and yellow."
        ),
        
        # Test Case 4: Good summary with slight variation (Score: ~0.8-0.9)
        LLMTestCase(
            input="What are the benefits of regular exercise?",
            actual_output="Regular exercise improves cardiovascular health, strengthens muscles, and enhances mental well-being.",
            expected_output="Exercise provides cardiovascular benefits, increases muscle strength, and improves mental health."
        )
    ]
    
    print(f"✓ Created {len(test_cases)} test cases")
    for i, tc in enumerate(test_cases, 1):
        print(f"  Test {i}: {tc.input[:50]}...")
    print()

    # Step 3: Create evaluation dataset
    print("Step 3: Creating Evaluation Dataset")
    print("-" * 80)
    dataset = EvaluationDataset(test_cases=test_cases)
    print(f"✓ Dataset created with {len(test_cases)} test cases")
    print()

    # Step 4: Run evaluation
    print("Step 4: Running Evaluation")
    print("-" * 80)
    print("Evaluating test cases... (this may take a moment)")
    print()
    
    results = dataset.evaluate([correctness_metric])
    
    # Step 5: Display results
    print("Step 5: Results")
    print("=" * 80)
    print()
    
    for i, test_case in enumerate(test_cases):
        score = results[i].metrics[0].score
        reason = results[i].metrics[0].reason
        
        # Determine status emoji
        if score >= 0.9:
            status = "✅ EXCELLENT"
        elif score >= 0.7:
            status = "✓ GOOD"
        elif score >= 0.5:
            status = "⚠ ACCEPTABLE"
        else:
            status = "❌ POOR"
        
        print(f"Test Case {i+1}: {status}")
        print(f"Score: {score:.2f}")
        print(f"Question: {test_case.input}")
        print(f"Expected: {test_case.expected_output[:80]}...")
        print(f"Actual: {test_case.actual_output[:80]}...")
        print(f"Reason: {reason}")
        print("-" * 80)
        print()
    
    # Step 6: Summary statistics
    print("Summary Statistics")
    print("=" * 80)
    scores = [results[i].metrics[0].score for i in range(len(test_cases))]
    avg_score = sum(scores) / len(scores)
    passed = sum(1 for s in scores if s >= correctness_metric.threshold)
    
    print(f"Average Score: {avg_score:.2f}")
    print(f"Passed: {passed}/{len(test_cases)} ({passed/len(test_cases)*100:.1f}%)")
    print(f"Threshold: {correctness_metric.threshold}")
    print()
    
    # Interpretation guide
    print("Score Interpretation Guide:")
    print("  0.9-1.0: Excellent - Perfect or near-perfect match")
    print("  0.7-0.9: Good - Minor differences, main ideas captured")
    print("  0.5-0.7: Acceptable - Some issues, needs improvement")
    print("  0.3-0.5: Poor - Significant problems")
    print("  0.0-0.3: Very Poor - Major errors or contradictions")
    print()
    
    print("=" * 80)
    print("Example Complete!")
    print("=" * 80)


if __name__ == "__main__":
    # Check for API key
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠ WARNING: OPENAI_API_KEY not found in environment variables")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        print()
    
    main()
