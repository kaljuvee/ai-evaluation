"""
Simple DSPy example that works with the current version
"""

import dspy
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_dspy():
    """Setup DSPy with OpenAI"""
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            lm = dspy.LM("openai/gpt-4o-mini")
            dspy.configure(lm=lm)
            print("✅ Configured DSPy with OpenAI GPT-4o-mini")
            return True
        else:
            print("⚠️  No OpenAI API key found")
            return False
    except Exception as e:
        print(f"❌ Error configuring DSPy: {e}")
        return False

def simple_summarization_example():
    """Simple summarization example"""
    print("\n📝 Simple Summarization Example")
    print("=" * 40)
    
    # Define signature
    class SimpleSummarize(dspy.Signature):
        """Simple summarization signature"""
        text = dspy.InputField(desc="Text to summarize")
        summary = dspy.OutputField(desc="Summary of the text")
    
    # Create module
    summarizer = dspy.ChainOfThought(SimpleSummarize())
    
    # Test text
    test_text = """
    Artificial Intelligence (AI) has transformed the landscape of modern computing in unprecedented ways. 
    From machine learning algorithms that power recommendation systems to natural language processing 
    models that enable human-like text generation, AI technologies are becoming increasingly sophisticated. 
    The development of large language models like GPT-4 and Claude has demonstrated the potential for 
    AI to understand and generate human language with remarkable accuracy.
    """
    
    print(f"Input text: {test_text[:100]}...")
    
    try:
        # Make prediction
        result = summarizer(text=test_text)
        print(f"✅ Summary: {result.summary}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def article_summarization_example():
    """Article summarization example"""
    print("\n📰 Article Summarization Example")
    print("=" * 40)
    
    # Define signature
    class ArticleSummarize(dspy.Signature):
        """Article summarization signature"""
        title = dspy.InputField(desc="Article title")
        content = dspy.InputField(desc="Article content")
        summary = dspy.OutputField(desc="Article summary")
    
    # Create module
    summarizer = dspy.ChainOfThought(ArticleSummarize())
    
    # Test article
    title = "The Rise of Artificial Intelligence"
    content = """
    Artificial Intelligence (AI) has transformed the landscape of modern computing in unprecedented ways. 
    From machine learning algorithms that power recommendation systems to natural language processing 
    models that enable human-like text generation, AI technologies are becoming increasingly sophisticated. 
    The development of large language models like GPT-4 and Claude has demonstrated the potential for 
    AI to understand and generate human language with remarkable accuracy. These advancements have 
    applications across various industries, including healthcare, finance, education, and entertainment.
    """
    
    print(f"Title: {title}")
    print(f"Content: {content[:100]}...")
    
    try:
        # Make prediction
        result = summarizer(title=title, content=content)
        print(f"✅ Summary: {result.summary}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_examples():
    """Run all examples"""
    print("🚀 DSPy Simple Examples")
    print("This demonstrates basic DSPy functionality")
    print("=" * 50)
    
    # Setup DSPy
    if not setup_dspy():
        print("⚠️  Running in demo mode without API access")
    
    # Run examples
    success_count = 0
    
    if simple_summarization_example():
        success_count += 1
    
    if article_summarization_example():
        success_count += 1
    
    print(f"\n📊 Results: {success_count}/2 examples successful")
    
    if success_count > 0:
        print("✅ DSPy is working correctly!")
    else:
        print("❌ DSPy examples failed. Check API key and network connection.")

if __name__ == "__main__":
    run_examples() 