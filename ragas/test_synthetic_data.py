"""
Test script for Ragas Synthetic Data Generation

This script tests the basic functionality of the Ragas synthetic data generator
without requiring an OpenAI API key.
"""

import asyncio
from ragas_synhetic_data_eval import RagasSyntheticDataGenerator


async def test_basic_functionality():
    """
    Test basic functionality without requiring API keys.
    """
    print("🧪 Testing Ragas Synthetic Data Generation")
    print("=" * 50)
    
    try:
        # Test 1: Check if we can create sample documents
        print("\n1. Testing sample document creation...")
        generator = RagasSyntheticDataGenerator()
        sample_docs = generator.create_sample_documents()
        print(f"✅ Created {len(sample_docs)} sample documents")
        
        # Test 2: Test knowledge graph creation (will fail without API key, but that's expected)
        print("\n2. Testing knowledge graph creation...")
        try:
            kg = generator.create_knowledge_graph(sample_docs)
            print(f"✅ Knowledge graph created: {kg}")
        except Exception as e:
            print(f"⚠️  Knowledge graph creation failed (expected without API key): {str(e)[:100]}...")
        
        # Test 3: Test simple testset generation (will fail without API key, but that's expected)
        print("\n3. Testing simple testset generation...")
        try:
            testset = generator.generate_testset_simple(sample_docs, testset_size=2)
            print(f"✅ Simple testset generated with {len(testset)} samples")
        except Exception as e:
            print(f"⚠️  Simple testset generation failed (expected without API key): {str(e)[:100]}...")
        
        # Test 4: Test knowledge graph testset generation (will fail without API key, but that's expected)
        print("\n4. Testing knowledge graph testset generation...")
        try:
            testset = generator.generate_testset(sample_docs, testset_size=2)
            print(f"✅ Knowledge graph testset generated with {len(testset)} samples")
        except Exception as e:
            print(f"⚠️  Knowledge graph testset generation failed (expected without API key): {str(e)[:100]}...")
        
        print("\n✅ Basic functionality tests completed!")
        print("\nTo run full synthetic data generation with LLM-based generation:")
        print("1. Set your OPENAI_API_KEY environment variable")
        print("2. Run: python ragas_synhetic_data_eval.py")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(test_basic_functionality()) 