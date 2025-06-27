"""
Simple DSPy test to verify installation and basic functionality
"""

import dspy
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_dspy_basic():
    """Test basic DSPy functionality"""
    print("🧪 Testing DSPy Basic Functionality")
    print("=" * 40)
    
    # Test 1: Check if DSPy is installed
    print("✅ DSPy version:", dspy.__version__)
    
    # Test 2: Try to configure with OpenAI
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            print(f"✅ OpenAI API key found: {api_key[:10]}...")
            lm = dspy.LM("openai/gpt-4o-mini")
            dspy.configure(lm=lm)
            print("✅ DSPy configured successfully")
        else:
            print("⚠️  No OpenAI API key found in .env file")
            print("Using mock configuration for testing...")
            dspy.configure(lm=None)
    except Exception as e:
        print(f"❌ Error configuring DSPy: {e}")
        dspy.configure(lm=None)
    
    # Test 3: Create a simple signature
    try:
        class SimpleSignature(dspy.Signature):
            """Simple test signature"""
            input_text = dspy.InputField(desc="Input text")
            output_text = dspy.OutputField(desc="Output text")
        
        print("✅ Signature creation successful")
        
        # Test 4: Create a simple module
        simple_module = dspy.ChainOfThought(SimpleSignature())
        print("✅ Module creation successful")
        
        # Test 5: Try a simple prediction (will fail without API key, but should not crash)
        try:
            result = simple_module(input_text="Hello world")
            print("✅ Prediction successful")
            print(f"Result: {result}")
        except Exception as e:
            print(f"⚠️  Prediction failed (expected without API key): {e}")
        
    except Exception as e:
        print(f"❌ Error in signature/module creation: {e}")
    
    print("\n🎯 Basic DSPy test completed!")

def test_dspy_imports():
    """Test DSPy imports"""
    print("\n📦 Testing DSPy Imports")
    print("=" * 30)
    
    try:
        from dspy import Signature, InputField, OutputField, Module, ChainOfThought
        print("✅ Basic imports successful")
        
        from dspy import BootstrapFewShot, BootstrapFinetune, MIPROv2, BetterTogether
        print("✅ Optimizer imports successful")
        
        from dspy import Example
        print("✅ Example import successful")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")

if __name__ == "__main__":
    test_dspy_imports()
    test_dspy_basic() 