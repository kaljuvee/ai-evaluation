"""
DSPy Ground Truth Optimization Example

This example demonstrates how to optimize DSPy programs using LABELED EXAMPLES (ground truth).
It uses BootstrapFewShot to learn from examples with known correct outputs.

Based on: dspy_readme.md
"""

import dspy
from dspy.teleprompt import BootstrapFewShot
import os


def main():
    print("=" * 80)
    print("DSPy Optimization WITH Ground Truth")
    print("=" * 80)
    print()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠ WARNING: OPENAI_API_KEY not found in environment variables")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        print()
        return

    # Step 1: Configure DSPy
    print("Step 1: Configuring DSPy")
    print("-" * 80)
    lm = dspy.OpenAI(model="gpt-3.5-turbo")
    dspy.settings.configure(lm=lm)
    print("✓ Configured with gpt-3.5-turbo")
    print()

    # Step 2: Define the task
    print("Step 2: Defining the Task")
    print("-" * 80)
    print("Task: Question Answering based on context")
    print()
    
    class QA(dspy.Signature):
        """Answer the question based on context."""
        context = dspy.InputField()
        question = dspy.InputField()
        answer = dspy.OutputField()

    class QuestionAnswerer(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate_answer = dspy.ChainOfThought(QA)
        
        def forward(self, context, question):
            return self.generate_answer(context=context, question=question)
    
    print("✓ Defined QA signature and module")
    print()

    # Step 3: Create training data WITH ground truth
    print("Step 3: Creating Training Data (WITH Ground Truth)")
    print("-" * 80)
    print("Ground truth = the ideal answer we want the AI to produce")
    print()
    
    trainset = [
        dspy.Example(
            context="Paris is the capital of France.",
            question="What is the capital of France?",
            answer="Paris"  # ← Ground truth (ideal answer)
        ).with_inputs("context", "question"),
        
        dspy.Example(
            context="Python was created by Guido van Rossum.",
            question="Who created Python?",
            answer="Guido van Rossum"  # ← Ground truth
        ).with_inputs("context", "question"),
        
        dspy.Example(
            context="The Eiffel Tower is located in Paris, France.",
            question="Where is the Eiffel Tower?",
            answer="Paris, France"  # ← Ground truth
        ).with_inputs("context", "question"),
        
        dspy.Example(
            context="Machine learning is a subset of artificial intelligence.",
            question="What is machine learning?",
            answer="Machine learning is a subset of artificial intelligence"  # ← Ground truth
        ).with_inputs("context", "question"),
    ]
    
    print(f"✓ Created {len(trainset)} training examples with ground truth")
    for i, ex in enumerate(trainset, 1):
        print(f"  Example {i}: {ex.question}")
        print(f"    Ground truth: {ex.answer}")
    print()

    # Step 4: Define success metric
    print("Step 4: Defining Success Metric")
    print("-" * 80)
    print("Metric: Check if the answer matches the ground truth")
    print()
    
    def validate_answer(example, pred, trace=None):
        """
        Validation metric that compares predicted answer to ground truth.
        Returns True if the ground truth answer is contained in the prediction.
        """
        return example.answer.lower() in pred.answer.lower()
    
    print("✓ Metric defined: Answer must contain ground truth")
    print()

    # Step 5: Test BEFORE optimization
    print("Step 5: Testing BEFORE Optimization")
    print("-" * 80)
    
    unoptimized_qa = QuestionAnswerer()
    
    test_example = dspy.Example(
        context="The Great Wall of China is over 13,000 miles long.",
        question="How long is the Great Wall of China?"
    ).with_inputs("context", "question")
    
    before_result = unoptimized_qa(
        context=test_example.context,
        question=test_example.question
    )
    
    print(f"Question: {test_example.question}")
    print(f"Answer (before optimization): {before_result.answer}")
    print()

    # Step 6: Optimize with BootstrapFewShot
    print("Step 6: Optimizing with BootstrapFewShot")
    print("-" * 80)
    print("BootstrapFewShot learns from your labeled examples...")
    print("This may take a moment...")
    print()
    
    try:
        optimizer = BootstrapFewShot(
            metric=validate_answer,
            max_bootstrapped_demos=2  # Number of examples to use
        )
        
        optimized_qa = optimizer.compile(
            QuestionAnswerer(),
            trainset=trainset
        )
        
        print("✓ Optimization complete!")
        print()
        
        # Step 7: Test AFTER optimization
        print("Step 7: Testing AFTER Optimization")
        print("-" * 80)
        
        after_result = optimized_qa(
            context=test_example.context,
            question=test_example.question
        )
        
        print(f"Question: {test_example.question}")
        print(f"Answer (after optimization): {after_result.answer}")
        print()
        
        # Step 8: Compare results
        print("Step 8: Comparison")
        print("=" * 80)
        print(f"Before: {before_result.answer}")
        print(f"After:  {after_result.answer}")
        print()
        
        # Test on all training examples
        print("Testing on Training Examples:")
        print("-" * 80)
        
        correct_before = 0
        correct_after = 0
        
        for i, ex in enumerate(trainset, 1):
            # Test unoptimized
            pred_before = unoptimized_qa(context=ex.context, question=ex.question)
            match_before = validate_answer(ex, pred_before)
            
            # Test optimized
            pred_after = optimized_qa(context=ex.context, question=ex.question)
            match_after = validate_answer(ex, pred_after)
            
            if match_before:
                correct_before += 1
            if match_after:
                correct_after += 1
            
            status_before = "✓" if match_before else "✗"
            status_after = "✓" if match_after else "✗"
            
            print(f"Example {i}: {ex.question}")
            print(f"  Before: {status_before} {pred_before.answer[:50]}...")
            print(f"  After:  {status_after} {pred_after.answer[:50]}...")
        
        print()
        print(f"Accuracy Before: {correct_before}/{len(trainset)} ({correct_before/len(trainset)*100:.1f}%)")
        print(f"Accuracy After:  {correct_after}/{len(trainset)} ({correct_after/len(trainset)*100:.1f}%)")
        print()
        
        if correct_after > correct_before:
            print("✅ Optimization improved performance!")
        elif correct_after == correct_before:
            print("⚠ Optimization maintained performance")
        else:
            print("❌ Optimization decreased performance (may need more examples)")
        
        print()
        print("=" * 80)
        print("Example Complete!")
        print("=" * 80)
        print()
        print("Key Takeaways:")
        print("- BootstrapFewShot learns from labeled examples (ground truth)")
        print("- It automatically creates better prompts based on your data")
        print("- More training examples generally lead to better results")
        print("- The metric guides what 'good' looks like")
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        print()
        print("Common issues:")
        print("- API key not set or invalid")
        print("- Network connectivity problems")
        print("- Insufficient training examples")
        raise


if __name__ == "__main__":
    main()
