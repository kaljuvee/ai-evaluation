"""
Test script for Ragas Summary Evaluation

This script tests the basic functionality of the Ragas summary evaluator
without requiring an OpenAI API key.
"""

import asyncio
from ragas_summary_eval import RagasSummaryEvaluator


async def test_basic_functionality():
    """
    Test basic functionality without requiring API keys.
    """
    print("🧪 Testing Ragas Summary Evaluation")
    print("=" * 40)
    
    try:
        # Test 1: Check if we can create sample data
        print("\n1. Testing sample data creation...")
        evaluator = RagasSummaryEvaluator()
        sample_data = evaluator.create_sample_data()
        print(f"✅ Created {len(sample_data)} sample data points")
        
        # Test 2: Test non-LLM metric (should work without API key)
        print("\n2. Testing non-LLM metric (BleuScore)...")
        bleu_score = evaluator.evaluate_single_sample_non_llm(sample_data[0])
        print(f"✅ BleuScore: {bleu_score:.3f}")
        
        # Test 3: Test dataset creation
        print("\n3. Testing dataset creation...")
        dataset = evaluator.create_evaluation_dataset_from_samples()
        print(f"✅ Created dataset with {len(dataset)} samples")
        print(f"   Dataset features: {dataset.features()}")
        
        # Test 4: Test LLM-based metric (will fail without API key, but that's expected)
        print("\n4. Testing LLM-based metric (AspectCritic)...")
        try:
            llm_sample = {k: v for k, v in sample_data[0].items() if k != 'reference'}
            aspect_score = await evaluator.evaluate_single_sample_llm(llm_sample)
            print(f"✅ AspectCritic Score: {aspect_score}")
        except Exception as e:
            print(f"⚠️  LLM-based metric failed (expected without API key): {str(e)[:100]}...")
        
        print("\n✅ Basic functionality tests completed!")
        print("\nTo run full evaluation with LLM-based metrics:")
        print("1. Set your OPENAI_API_KEY environment variable")
        print("2. Run: python ragas_summary_eval.py")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(test_basic_functionality()) 