#!/usr/bin/env python3
"""
DeepEval Installation Test

This script verifies that DeepEval is properly installed and can perform basic operations.
Run this before using the main RAG evaluation sample.
"""

import sys
import os

def test_imports():
    """Test if DeepEval can be imported"""
    print("🔍 Testing DeepEval imports...")
    
    try:
        import deepeval
        print("✅ DeepEval imported successfully")
        
        from deepeval import assert_test, evaluate
        print("✅ Core functions imported successfully")
        
        from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric
        print("✅ Metrics imported successfully")
        
        from deepeval.test_case import LLMTestCase, ConversationalTestCase
        print("✅ Test cases imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic DeepEval functionality"""
    print("\n🔍 Testing basic functionality...")
    
    try:
        from deepeval.metrics import FaithfulnessMetric
        from deepeval.test_case import LLMTestCase
        
        # Create a simple test case
        test_case = LLMTestCase(
            input="What is AI?",
            actual_output="Artificial Intelligence is a field of computer science.",
            expected_output="AI is a branch of computer science.",
            retrieval_context=["Artificial Intelligence is a field of computer science."]
        )
        
        # Create a metric
        faithfulness_metric = FaithfulnessMetric(threshold=0.5)
        
        print("✅ Test case and metric created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def test_environment():
    """Test environment setup"""
    print("\n🔍 Testing environment setup...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version is compatible")
    
    # Check OpenAI API key
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print("✅ OpenAI API key is set")
    else:
        print("⚠️  OpenAI API key not set (required for LLM-based evaluations)")
        print("   Set it with: export OPENAI_API_KEY='your-key'")
    
    return True

def main():
    """Run all tests"""
    print("🚀 DeepEval Installation Test")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 3
    
    # Run tests
    if test_imports():
        tests_passed += 1
    
    if test_basic_functionality():
        tests_passed += 1
    
    if test_environment():
        tests_passed += 1
    
    # Summary
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! DeepEval is ready to use.")
        print("\nNext steps:")
        print("1. Run the RAG sample: python deepeval_rag_sample.py")
        print("2. Check the README.md for detailed usage instructions")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Install DeepEval: pip install deepeval")
        print("2. Set OpenAI API key: export OPENAI_API_KEY='your-key'")
        print("3. Check Python version (3.8+ required)")

if __name__ == "__main__":
    main() 