"""
Test Installation Script for Ragas Evaluation System

This script verifies that all dependencies are properly installed
and the system can perform basic operations.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        from ragas import evaluate
        print("✓ ragas imported successfully")
    except ImportError as e:
        print(f"✗ ragas import failed: {e}")
        return False
    
    try:
        from langchain.schema import Document
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.embeddings import HuggingFaceEmbeddings
        print("✓ langchain components imported successfully")
    except ImportError as e:
        print(f"✗ langchain import failed: {e}")
        return False
    
    # Skip FAISS and sentence-transformers checks in minimal OpenAI-only setup
    
    return True

def test_basic_ragas_functionality():
    """Test basic Ragas functionality"""
    print("\nTesting basic Ragas functionality...")
    
    try:
        from ragas.metrics import answer_correctness, faithfulness
        print("✓ Ragas metrics imported successfully")
        
        # Test that we can create metric instances
        ac_metric = answer_correctness
        faith_metric = faithfulness
        print("✓ Ragas metrics instantiated successfully")
        
        return True
    except Exception as e:
        print(f"✗ Ragas functionality test failed: {e}")
        return False

def test_langchain_functionality():
    """Test basic LangChain functionality"""
    print("\nTesting LangChain functionality...")
    
    try:
        from langchain.schema import Document
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # Test document creation
        doc = Document(page_content="Test document", metadata={"source": "test"})
        print("✓ Document creation successful")
        
        # Test text splitting
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        splits = splitter.split_documents([doc])
        print(f"✓ Text splitting successful (created {len(splits)} splits)")
        
        return True
    except Exception as e:
        print(f"✗ LangChain functionality test failed: {e}")
        return False

def test_embeddings():
    """Test HuggingFace embeddings"""
    print("\nTesting HuggingFace embeddings...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Test embedding model loading
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("✓ Embedding model loaded successfully")
        
        # Test embedding generation
        embeddings = model.encode(["Test sentence"])
        print(f"✓ Embedding generation successful (shape: {embeddings.shape})")
        
        return True
    except Exception as e:
        print(f"✗ Embeddings test failed: {e}")
        return False

def test_openai_config():
    """Test OpenAI configuration"""
    print("\nTesting OpenAI configuration...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("✓ OpenAI API key found")
        return True
    else:
        print("⚠ OpenAI API key not found (set OPENAI_API_KEY environment variable)")
        print("  Some features may not work without this key")
        return False

def test_directory_structure():
    """Test that required directories can be created"""
    print("\nTesting directory structure...")
    
    try:
        test_dirs = ["./cache", "./eval-results", "./test-output"]
        
        for dir_path in test_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"✓ Directory {dir_path} created/verified")
        
        # Clean up test directory
        import shutil
        if Path("./test-output").exists():
            shutil.rmtree("./test-output")
        
        return True
    except Exception as e:
        print(f"✗ Directory structure test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Ragas Evaluation System - Installation Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Ragas Functionality", test_basic_ragas_functionality),
        ("LangChain Functionality", test_langchain_functionality),
        ("Embeddings", test_embeddings),
        ("OpenAI Configuration", test_openai_config),
        ("Directory Structure", test_directory_structure)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your installation is ready.")
        print("\nNext steps:")
        print("1. Set your OpenAI API key: export OPENAI_API_KEY='your-key'")
        print("2. Run: python ragas_evaluator.py")
        print("3. Or run examples: python example_evaluations.py")
    else:
        print("⚠ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Check your Python environment")
        print("3. Ensure you have sufficient disk space for models")

if __name__ == "__main__":
    main() 