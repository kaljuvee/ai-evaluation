"""
Simple Demo of Ragas Evaluation System

This script demonstrates the basic usage of the Ragas evaluation system
with a minimal example that can run quickly.
"""

import os
from ragas_evaluator import RagasEvaluator, EvaluationConfig

def simple_demo():
    """Run a simple demonstration of the Ragas evaluation system"""
    
    print("🚀 Ragas Evaluation System Demo")
    print("=" * 40)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set")
        print("   Set it with: export OPENAI_API_KEY='your-key'")
        print("   Some features may not work without this key")
        print()
    
    # Configuration for quick demo
    config = EvaluationConfig(
        model_name="gpt-3.5-turbo",
        temperature=0.0,
        batch_size=3,  # Small batch for quick demo
        cache_dir="./cache",
        results_dir="./eval-results/demo"
    )
    
    print("📋 Configuration:")
    print(f"   Model: {config.model_name}")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Results Directory: {config.results_dir}")
    print()
    
    try:
        print("🔄 Running evaluation...")
        evaluator = RagasEvaluator(config)
        results = evaluator.run_comprehensive_evaluation()
        
        print("\n📊 Results:")
        print("-" * 20)
        for metric_name, metric_value in results['metrics'].items():
            print(f"  {metric_name}: {metric_value:.4f}")
        
        print(f"\n📈 Test Set Size: {results['testset_size']} questions")
        print(f"💾 Results saved to: {config.results_dir}")
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have set OPENAI_API_KEY")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Check your internet connection")
        print("4. Ensure you have sufficient disk space")

if __name__ == "__main__":
    simple_demo() 