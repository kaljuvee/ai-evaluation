"""
DSPy Optimization Examples: Advanced Techniques for Summarization

This file demonstrates various DSPy optimization techniques:
1. BootstrapFewShot - Few-shot learning with examples
2. BootstrapFinetune - Fine-tuning approach
3. MIPROv2 - Advanced prompt optimization
4. BetterTogether - Combined optimization strategies
5. Custom optimization with multiple metrics
"""

import dspy
from dspy_sample import (
    SummarizationSignature, 
    SyntheticDataGenerator, 
    Article,
    combined_metric
)
import random
from typing import List, Dict, Any

def setup_for_optimization():
    """Setup DSPy and generate data for optimization examples"""
    print("🔧 Setting up for optimization examples...")
    
    # Setup DSPy (you'll need API keys for full functionality)
    try:
        lm = dspy.LM("openai/gpt-4o-mini")
        dspy.configure(lm=lm)
        print("✅ Configured with OpenAI GPT-4o-mini")
    except Exception as e:
        print(f"⚠️  Using mock configuration: {e}")
        dspy.configure(lm=None)
    
    # Generate synthetic data
    data_generator = SyntheticDataGenerator()
    articles = data_generator.generate_synthetic_articles(12)
    
    # Create DSPy examples
    train_examples = []
    for article in articles:
        example = dspy.Example(
            article_title=article.title,
            article_content=article.content,
            summary=article.expected_summary
        ).with_inputs("article_title", "article_content")
        train_examples.append(example)
    
    print(f"📚 Created {len(train_examples)} training examples")
    return train_examples

def example_1_bootstrap_fewshot():
    """Example 1: BootstrapFewShot optimization"""
    print("\n🎯 Example 1: BootstrapFewShot Optimization")
    print("=" * 50)
    
    train_examples = setup_for_optimization()
    
    # Create base summarizer
    base_summarizer = dspy.ChainOfThought(SummarizationSignature())
    
    # Define optimization metric
    def fewshot_metric(gold, pred, trace=None):
        """Metric for few-shot optimization"""
        # Simple keyword overlap metric
        gold_words = set(gold.summary.lower().split())
        pred_words = set(pred.summary.lower().split())
        
        if not gold_words:
            return 0.0
        
        overlap = len(gold_words.intersection(pred_words))
        return overlap / len(gold_words)
    
    # Create BootstrapFewShot optimizer
    optimizer = dspy.BootstrapFewShot(
        metric=fewshot_metric,
        max_bootstrapped_demos=4,  # Number of examples to use
        max_labeled_demos=2,       # Number of labeled examples
        num_candidate_programs=3,  # Number of candidate programs to generate
        num_threads=1              # Number of parallel threads
    )
    
    print("🔄 Running BootstrapFewShot optimization...")
    print("This will:")
    print("  - Generate few-shot examples from training data")
    print("  - Create multiple candidate programs")
    print("  - Select the best performing program")
    
    try:
        # Compile the optimized model
        optimized_model = optimizer.compile(
            base_summarizer, 
            trainset=train_examples[:8]  # Use subset for optimization
        )
        
        print("✅ Optimization completed!")
        print("Optimized model ready for use")
        
        # Test the optimized model
        test_article = train_examples[8]
        print(f"\n📝 Testing optimized model:")
        print(f"Title: {test_article.article_title}")
        print(f"Expected: {test_article.summary}")
        
        try:
            prediction = optimized_model(
                article_title=test_article.article_title,
                article_content=test_article.article_content
            )
            print(f"Predicted: {prediction.summary}")
        except Exception as e:
            print(f"⚠️  Prediction failed: {e}")
        
        return optimized_model
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        print("This might be due to missing API keys or network issues")
        return None

def example_2_bootstrap_finetune():
    """Example 2: BootstrapFinetune optimization"""
    print("\n🎯 Example 2: BootstrapFinetune Optimization")
    print("=" * 50)
    
    train_examples = setup_for_optimization()
    
    # Create base summarizer
    base_summarizer = dspy.ChainOfThought(SummarizationSignature())
    
    # Create BootstrapFinetune optimizer
    optimizer = dspy.BootstrapFinetune(
        metric=combined_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=2,
        num_candidate_programs=2,
        num_threads=1
    )
    
    print("🔄 Running BootstrapFinetune optimization...")
    print("This will:")
    print("  - Generate training examples")
    print("  - Fine-tune the model on these examples")
    print("  - Optimize prompts and weights")
    
    try:
        # Compile the optimized model
        optimized_model = optimizer.compile(
            base_summarizer, 
            trainset=train_examples[:6]
        )
        
        print("✅ Fine-tuning completed!")
        
        return optimized_model
        
    except Exception as e:
        print(f"❌ Fine-tuning failed: {e}")
        print("This might be due to missing API keys or network issues")
        return None

def example_3_mipro_optimization():
    """Example 3: MIPROv2 optimization (advanced)"""
    print("\n🎯 Example 3: MIPROv2 Optimization (Advanced)")
    print("=" * 50)
    
    train_examples = setup_for_optimization()
    
    # Create base summarizer
    base_summarizer = dspy.ChainOfThought(SummarizationSignature())
    
    # Create MIPROv2 optimizer
    optimizer = dspy.MIPROv2(
        metric=combined_metric,
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
        num_candidate_programs=2,
        num_threads=1
    )
    
    print("🔄 Running MIPROv2 optimization...")
    print("This advanced optimizer will:")
    print("  - Use multi-stage optimization")
    print("  - Generate and refine prompts")
    print("  - Optimize program structure")
    
    try:
        # Compile the optimized model
        optimized_model = optimizer.compile(
            base_summarizer, 
            trainset=train_examples[:6]
        )
        
        print("✅ MIPROv2 optimization completed!")
        
        return optimized_model
        
    except Exception as e:
        print(f"❌ MIPROv2 optimization failed: {e}")
        print("This might be due to missing API keys or network issues")
        return None

def example_4_better_together():
    """Example 4: BetterTogether optimization (combines multiple strategies)"""
    print("\n🎯 Example 4: BetterTogether Optimization")
    print("=" * 50)
    
    train_examples = setup_for_optimization()
    
    # Create base summarizer
    base_summarizer = dspy.ChainOfThought(SummarizationSignature())
    
    # Create BetterTogether optimizer
    optimizer = dspy.BetterTogether(
        metric=combined_metric,
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
        num_candidate_programs=2,
        num_threads=1
    )
    
    print("🔄 Running BetterTogether optimization...")
    print("This optimizer combines multiple strategies:")
    print("  - Few-shot learning")
    print("  - Fine-tuning")
    print("  - Advanced prompt optimization")
    
    try:
        # Compile the optimized model
        optimized_model = optimizer.compile(
            base_summarizer, 
            trainset=train_examples[:6]
        )
        
        print("✅ BetterTogether optimization completed!")
        
        return optimized_model
        
    except Exception as e:
        print(f"❌ BetterTogether optimization failed: {e}")
        print("This might be due to missing API keys or network issues")
        return None

def example_5_custom_optimization():
    """Example 5: Custom optimization with multiple metrics"""
    print("\n🎯 Example 5: Custom Optimization with Multiple Metrics")
    print("=" * 50)
    
    train_examples = setup_for_optimization()
    
    # Define custom metrics
    def length_metric(gold, pred, trace=None):
        """Prefer summaries of appropriate length"""
        gold_length = len(gold.summary.split())
        pred_length = len(pred.summary.split())
        
        # Prefer summaries within 50% of expected length
        min_length = gold_length * 0.5
        max_length = gold_length * 1.5
        
        if min_length <= pred_length <= max_length:
            return 1.0
        else:
            return 0.0
    
    def keyword_metric(gold, pred, trace=None):
        """Prefer summaries with keyword overlap"""
        gold_words = set(gold.summary.lower().split())
        pred_words = set(pred.summary.lower().split())
        
        if not gold_words:
            return 0.0
        
        overlap = len(gold_words.intersection(pred_words))
        return overlap / len(gold_words)
    
    def custom_combined_metric(gold, pred, trace=None):
        """Custom combined metric"""
        length_score = length_metric(gold, pred, trace)
        keyword_score = keyword_metric(gold, pred, trace)
        
        # Weighted combination
        return length_score * 0.4 + keyword_score * 0.6
    
    # Create base summarizer
    base_summarizer = dspy.ChainOfThought(SummarizationSignature())
    
    # Create optimizer with custom metric
    optimizer = dspy.BootstrapFewShot(
        metric=custom_combined_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=2,
        num_candidate_programs=2,
        num_threads=1
    )
    
    print("🔄 Running custom optimization...")
    print("Using custom metric that considers:")
    print("  - Summary length appropriateness (40%)")
    print("  - Keyword overlap (60%)")
    
    try:
        # Compile the optimized model
        optimized_model = optimizer.compile(
            base_summarizer, 
            trainset=train_examples[:6]
        )
        
        print("✅ Custom optimization completed!")
        
        return optimized_model
        
    except Exception as e:
        print(f"❌ Custom optimization failed: {e}")
        print("This might be due to missing API keys or network issues")
        return None

def example_6_ensemble_optimization():
    """Example 6: Ensemble optimization with multiple models"""
    print("\n🎯 Example 6: Ensemble Optimization")
    print("=" * 50)
    
    train_examples = setup_for_optimization()
    
    # Create multiple base models
    model1 = dspy.ChainOfThought(SummarizationSignature())
    model2 = dspy.ChainOfThought(SummarizationSignature())
    model3 = dspy.ChainOfThought(SummarizationSignature())
    
    # Create ensemble
    ensemble = dspy.Ensemble([model1, model2, model3])
    
    # Create optimizer
    optimizer = dspy.BootstrapFewShot(
        metric=combined_metric,
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
        num_candidate_programs=2,
        num_threads=1
    )
    
    print("🔄 Running ensemble optimization...")
    print("This will optimize an ensemble of 3 models:")
    print("  - Model 1: Basic chain-of-thought")
    print("  - Model 2: Basic chain-of-thought")
    print("  - Model 3: Basic chain-of-thought")
    
    try:
        # Compile the optimized ensemble
        optimized_ensemble = optimizer.compile(
            ensemble, 
            trainset=train_examples[:6]
        )
        
        print("✅ Ensemble optimization completed!")
        
        return optimized_ensemble
        
    except Exception as e:
        print(f"❌ Ensemble optimization failed: {e}")
        print("This might be due to missing API keys or network issues")
        return None

def compare_optimization_results():
    """Compare results from different optimization techniques"""
    print("\n📊 Comparing Optimization Results")
    print("=" * 50)
    
    # Run all optimization examples
    results = {}
    
    optimizers = [
        ("BootstrapFewShot", example_1_bootstrap_fewshot),
        ("BootstrapFinetune", example_2_bootstrap_finetune),
        ("MIPROv2", example_3_mipro_optimization),
        ("BetterTogether", example_4_better_together),
        ("Custom", example_5_custom_optimization),
        ("Ensemble", example_6_ensemble_optimization)
    ]
    
    for name, optimizer_func in optimizers:
        print(f"\n🔄 Testing {name}...")
        try:
            result = optimizer_func()
            results[name] = "✅ Success" if result else "❌ Failed"
        except Exception as e:
            results[name] = f"❌ Error: {str(e)[:50]}..."
    
    # Print comparison
    print("\n📈 Optimization Results Summary:")
    print("-" * 40)
    for name, result in results.items():
        print(f"{name:<20} {result}")
    
    print("\n💡 Notes:")
    print("- Success depends on API key availability")
    print("- Different optimizers may work better for different tasks")
    print("- Consider cost and time when choosing optimization strategy")

def run_all_examples():
    """Run all optimization examples"""
    print("🚀 DSPy Optimization Examples")
    print("This demonstrates various DSPy optimization techniques")
    print("=" * 60)
    
    try:
        compare_optimization_results()
        
        print("\n✅ All examples completed!")
        print("\n📝 Key Takeaways:")
        print("  1. BootstrapFewShot: Good for few-shot learning")
        print("  2. BootstrapFinetune: Good for fine-tuning")
        print("  3. MIPROv2: Advanced optimization for complex tasks")
        print("  4. BetterTogether: Combines multiple strategies")
        print("  5. Custom metrics: Tailor optimization to your needs")
        print("  6. Ensembles: Improve robustness with multiple models")
        
    except Exception as e:
        print(f"\n❌ Examples failed: {e}")
        print("This might be due to missing API keys or network issues")

if __name__ == "__main__":
    run_all_examples() 