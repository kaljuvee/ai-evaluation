"""
DSPy Metric-Only Optimization Example

This example demonstrates how to optimize DSPy programs WITHOUT labeled examples.
It uses quality metrics to evaluate and improve prompts automatically.

Based on: dspy_readme.md
"""

import dspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
import os


def main():
    print("=" * 80)
    print("DSPy Optimization WITHOUT Ground Truth (Metric-Only)")
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
    print("Task: Summarize text concisely")
    print()
    
    class Summarize(dspy.Signature):
        """Summarize the text concisely."""
        text = dspy.InputField()
        summary = dspy.OutputField()

    class Summarizer(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate_summary = dspy.ChainOfThought(Summarize)
        
        def forward(self, text):
            return self.generate_summary(text=text)
    
    print("✓ Defined Summarize signature and module")
    print()

    # Step 3: Create training data WITHOUT ground truth
    print("Step 3: Creating Training Data (WITHOUT Ground Truth)")
    print("-" * 80)
    print("Notice: We only provide the INPUT text, no ideal summaries!")
    print()
    
    trainset = [
        dspy.Example(
            text="Climate change is one of the most pressing issues of our time. Rising global temperatures are causing ice caps to melt, sea levels to rise, and weather patterns to become more extreme. Scientists agree that human activities, particularly the burning of fossil fuels, are the primary cause."
        ).with_inputs("text"),
        
        dspy.Example(
            text="Artificial intelligence has made remarkable progress in recent years. Machine learning algorithms can now recognize images, understand speech, and even generate human-like text. However, concerns about AI safety and ethics remain important topics of discussion."
        ).with_inputs("text"),
        
        dspy.Example(
            text="The human brain is an incredibly complex organ containing approximately 86 billion neurons. These neurons communicate through electrical and chemical signals, forming the basis of all our thoughts, memories, and behaviors. Neuroscientists continue to uncover new insights into how the brain works."
        ).with_inputs("text"),
        
        dspy.Example(
            text="Renewable energy sources like solar and wind power are becoming increasingly cost-effective. As technology improves and economies of scale kick in, clean energy is now competitive with traditional fossil fuels in many markets. This transition is crucial for reducing carbon emissions."
        ).with_inputs("text"),
    ]
    
    print(f"✓ Created {len(trainset)} training examples WITHOUT ground truth")
    for i, ex in enumerate(trainset, 1):
        print(f"  Example {i}: {ex.text[:60]}...")
    print()

    # Step 4: Define quality metric (NO ground truth needed!)
    print("Step 4: Defining Quality Metric (No Ground Truth Needed)")
    print("-" * 80)
    print("Metric: Evaluate summary quality based on objective criteria")
    print()
    
    def quality_metric(example, pred, trace=None):
        """
        Quality metric that evaluates summaries WITHOUT ground truth.
        Checks multiple quality criteria and returns a score.
        """
        summary = pred.summary
        
        # Criterion 1: Summary should be concise (< 100 chars)
        is_concise = len(summary) < 100
        
        # Criterion 2: Summary should have substance (> 20 chars)
        has_content = len(summary) > 20
        
        # Criterion 3: Summary should not be too repetitive
        words = summary.split()
        unique_ratio = len(set(words)) / len(words) if words else 0
        no_repetition = unique_ratio > 0.7
        
        # Criterion 4: Summary should contain key words from original
        original_words = set(example.text.lower().split()[:20])  # First 20 words
        summary_words = set(summary.lower().split())
        has_key_words = len(original_words & summary_words) > 2
        
        # Calculate overall score (0.0 to 1.0)
        score = (is_concise + has_content + no_repetition + has_key_words) / 4
        
        return score
    
    print("✓ Metric defined with 4 quality criteria:")
    print("  1. Concise (< 100 characters)")
    print("  2. Has content (> 20 characters)")
    print("  3. Not repetitive (unique word ratio > 0.7)")
    print("  4. Contains key words from original")
    print()

    # Step 5: Test BEFORE optimization
    print("Step 5: Testing BEFORE Optimization")
    print("-" * 80)
    
    unoptimized_summarizer = Summarizer()
    
    test_text = "Quantum computing represents a paradigm shift in computation. Unlike classical computers that use bits, quantum computers use qubits that can exist in multiple states simultaneously. This property, called superposition, allows quantum computers to solve certain problems exponentially faster than classical computers."
    
    before_result = unoptimized_summarizer(text=test_text)
    before_score = quality_metric(
        dspy.Example(text=test_text).with_inputs("text"),
        before_result
    )
    
    print(f"Test text: {test_text[:80]}...")
    print(f"Summary (before): {before_result.summary}")
    print(f"Quality score: {before_score:.2f}")
    print()

    # Step 6: Optimize with metric-only approach
    print("Step 6: Optimizing with Metric-Only Approach")
    print("-" * 80)
    print("Using BootstrapFewShotWithRandomSearch...")
    print("This tries different prompt variations and keeps the best ones")
    print("This may take a moment...")
    print()
    
    try:
        # Note: MIPRO would be ideal here, but BootstrapFewShotWithRandomSearch
        # also works well for metric-only optimization
        optimizer = BootstrapFewShotWithRandomSearch(
            metric=quality_metric,
            max_bootstrapped_demos=2,
            num_candidate_programs=3  # Try 3 different prompt variations
        )
        
        optimized_summarizer = optimizer.compile(
            Summarizer(),
            trainset=trainset
        )
        
        print("✓ Optimization complete!")
        print()
        
        # Step 7: Test AFTER optimization
        print("Step 7: Testing AFTER Optimization")
        print("-" * 80)
        
        after_result = optimized_summarizer(text=test_text)
        after_score = quality_metric(
            dspy.Example(text=test_text).with_inputs("text"),
            after_result
        )
        
        print(f"Test text: {test_text[:80]}...")
        print(f"Summary (after): {after_result.summary}")
        print(f"Quality score: {after_score:.2f}")
        print()
        
        # Step 8: Compare results
        print("Step 8: Comparison")
        print("=" * 80)
        print(f"Before: {before_result.summary}")
        print(f"Score:  {before_score:.2f}")
        print()
        print(f"After:  {after_result.summary}")
        print(f"Score:  {after_score:.2f}")
        print()
        
        improvement = after_score - before_score
        if improvement > 0.1:
            print(f"✅ Significant improvement: +{improvement:.2f}")
        elif improvement > 0:
            print(f"✓ Slight improvement: +{improvement:.2f}")
        else:
            print(f"⚠ No improvement: {improvement:.2f}")
        
        print()
        
        # Test on training examples
        print("Testing on Training Examples:")
        print("-" * 80)
        
        total_score_before = 0
        total_score_after = 0
        
        for i, ex in enumerate(trainset, 1):
            # Test unoptimized
            pred_before = unoptimized_summarizer(text=ex.text)
            score_before = quality_metric(ex, pred_before)
            
            # Test optimized
            pred_after = optimized_summarizer(text=ex.text)
            score_after = quality_metric(ex, pred_after)
            
            total_score_before += score_before
            total_score_after += score_after
            
            print(f"Example {i}:")
            print(f"  Before: {score_before:.2f} - {pred_before.summary[:60]}...")
            print(f"  After:  {score_after:.2f} - {pred_after.summary[:60]}...")
        
        avg_before = total_score_before / len(trainset)
        avg_after = total_score_after / len(trainset)
        
        print()
        print(f"Average Quality Before: {avg_before:.2f}")
        print(f"Average Quality After:  {avg_after:.2f}")
        print(f"Improvement: {avg_after - avg_before:+.2f}")
        print()
        
        print("=" * 80)
        print("Example Complete!")
        print("=" * 80)
        print()
        print("Key Takeaways:")
        print("- Metric-only optimization works WITHOUT labeled examples")
        print("- Define quality criteria that capture what 'good' looks like")
        print("- DSPy automatically finds prompts that maximize your metric")
        print("- More diverse training examples lead to better generalization")
        print("- This approach is great for creative tasks like summarization")
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        print()
        print("Common issues:")
        print("- API key not set or invalid")
        print("- Network connectivity problems")
        print("- Metric returning invalid scores (must be 0.0 to 1.0)")
        raise


if __name__ == "__main__":
    main()
