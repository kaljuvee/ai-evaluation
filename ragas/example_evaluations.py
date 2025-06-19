"""
Example Evaluations with Ragas

This script demonstrates various evaluation scenarios and use cases
for the Ragas evaluation system.
"""

import os
import pandas as pd
from pathlib import Path
from ragas_evaluator import (
    RagasEvaluator, 
    RAGSystem, 
    TestSetGenerator, 
    EvaluationConfig
)

def example_1_basic_evaluation():
    """Example 1: Basic RAG evaluation with synthetic data"""
    print("=== Example 1: Basic RAG Evaluation ===")
    
    config = EvaluationConfig(
        model_name="gpt-3.5-turbo",
        batch_size=5,
        results_dir="./eval-results/example1"
    )
    
    evaluator = RagasEvaluator(config)
    results = evaluator.run_comprehensive_evaluation()
    
    print("Results:")
    for metric, value in results['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    return results

def example_2_custom_documents():
    """Example 2: Evaluation with custom documents"""
    print("\n=== Example 2: Custom Documents Evaluation ===")
    
    # Custom documents about climate change
    documents = [
        "Climate change refers to long-term shifts in global weather patterns and average temperatures. The primary driver of recent climate change is the increase in greenhouse gas concentrations in the atmosphere, particularly carbon dioxide from burning fossil fuels.",
        
        "The Paris Agreement, adopted in 2015, aims to limit global warming to well below 2 degrees Celsius above pre-industrial levels. Countries have committed to reducing their greenhouse gas emissions and adapting to climate impacts.",
        
        "Renewable energy sources like solar, wind, and hydroelectric power produce electricity without emitting greenhouse gases. These technologies have become increasingly cost-competitive with fossil fuels in recent years.",
        
        "Climate change impacts include rising sea levels, more frequent extreme weather events, changes in precipitation patterns, and threats to biodiversity. These effects are already being observed worldwide.",
        
        "Mitigation strategies include transitioning to renewable energy, improving energy efficiency, protecting forests, and developing carbon capture technologies. Adaptation measures include building resilient infrastructure and developing early warning systems."
    ]
    
    config = EvaluationConfig(
        model_name="gpt-3.5-turbo",
        batch_size=5,
        results_dir="./eval-results/example2"
    )
    
    # Create RAG system
    rag_system = RAGSystem(documents, config)
    
    # Generate test set
    generator = TestSetGenerator(config)
    testset = generator.generate_from_documents(documents, num_questions=20)
    
    # Evaluate
    evaluator = RagasEvaluator(config)
    results = evaluator.evaluate_rag_system(rag_system, testset)
    
    print("Climate Change RAG Results:")
    for metric, value in results['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    return results

def example_3_multiple_rag_systems():
    """Example 3: Comparing multiple RAG systems"""
    print("\n=== Example 3: Multiple RAG Systems Comparison ===")
    
    # Different document sets
    tech_docs = [
        "Artificial Intelligence (AI) is transforming industries worldwide. Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions.",
        "Cloud computing provides scalable computing resources over the internet. Major providers include AWS, Google Cloud, and Microsoft Azure.",
        "Cybersecurity is crucial in the digital age. Threats include malware, phishing attacks, and data breaches."
    ]
    
    health_docs = [
        "Public health focuses on protecting and improving community health. Key areas include disease prevention, health promotion, and emergency response.",
        "Vaccines have been one of the most successful public health interventions. They prevent millions of deaths annually from infectious diseases.",
        "Mental health is as important as physical health. Conditions like depression and anxiety affect millions of people worldwide."
    ]
    
    config = EvaluationConfig(
        model_name="gpt-3.5-turbo",
        batch_size=5,
        results_dir="./eval-results/example3"
    )
    
    # Create RAG systems
    tech_rag = RAGSystem(tech_docs, config)
    health_rag = RAGSystem(health_docs, config)
    
    # Generate test sets
    generator = TestSetGenerator(config)
    tech_testset = generator.generate_from_documents(tech_docs, num_questions=15)
    health_testset = generator.generate_from_documents(health_docs, num_questions=15)
    
    # Evaluate both systems
    evaluator = RagasEvaluator(config)
    
    tech_results = evaluator.evaluate_rag_system(tech_rag, tech_testset)
    health_results = evaluator.evaluate_rag_system(health_rag, health_testset)
    
    print("Comparison Results:")
    print("\nTechnology RAG:")
    for metric, value in tech_results['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nHealthcare RAG:")
    for metric, value in health_results['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    return {
        'tech': tech_results,
        'health': health_results
    }

def example_4_parameter_sensitivity():
    """Example 4: Testing different parameters"""
    print("\n=== Example 4: Parameter Sensitivity Analysis ===")
    
    documents = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
        "Deep learning uses neural networks with multiple layers to model complex patterns in data.",
        "Natural language processing helps computers understand and generate human language."
    ]
    
    # Test different batch sizes
    batch_sizes = [1, 5, 10]
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        
        config = EvaluationConfig(
            model_name="gpt-3.5-turbo",
            batch_size=batch_size,
            results_dir=f"./eval-results/example4_batch_{batch_size}"
        )
        
        rag_system = RAGSystem(documents, config)
        generator = TestSetGenerator(config)
        testset = generator.generate_from_documents(documents, num_questions=10)
        
        evaluator = RagasEvaluator(config)
        result = evaluator.evaluate_rag_system(rag_system, testset)
        
        results[f'batch_{batch_size}'] = result
        
        print(f"  Answer Correctness: {result['metrics']['answer_correctness']:.4f}")
        print(f"  Faithfulness: {result['metrics']['faithfulness']:.4f}")
    
    return results

def example_5_custom_questions():
    """Example 5: Evaluation with custom questions"""
    print("\n=== Example 5: Custom Questions Evaluation ===")
    
    documents = [
        "Python is a high-level programming language known for its simplicity and readability. It's widely used in data science, web development, and automation.",
        "The Python ecosystem includes popular libraries like NumPy for numerical computing, Pandas for data manipulation, and Matplotlib for visualization.",
        "Python's package manager, pip, makes it easy to install and manage third-party libraries and dependencies."
    ]
    
    # Custom questions
    custom_questions = [
        "What is Python and what is it used for?",
        "Name some popular Python libraries for data science.",
        "How do you install packages in Python?",
        "What makes Python different from other programming languages?",
        "Can you explain the Python ecosystem?"
    ]
    
    config = EvaluationConfig(
        model_name="gpt-3.5-turbo",
        batch_size=5,
        results_dir="./eval-results/example5"
    )
    
    # Create RAG system
    rag_system = RAGSystem(documents, config)
    
    # Create test set from custom questions
    testset_data = []
    for question in custom_questions:
        result = rag_system.query(question)
        testset_data.append({
            'question': question,
            'answer': result['answer'],
            'contexts': [result['context']]
        })
    
    testset = pd.DataFrame(testset_data)
    
    # Evaluate
    evaluator = RagasEvaluator(config)
    results = evaluator.evaluate_rag_system(rag_system, testset)
    
    print("Custom Questions Results:")
    for metric, value in results['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    # Show some example Q&A pairs
    print("\nExample Q&A Pairs:")
    for i, row in testset.iterrows():
        print(f"\nQ: {row['question']}")
        print(f"A: {row['answer'][:100]}...")
    
    return results

def example_6_error_analysis():
    """Example 6: Error analysis and detailed inspection"""
    print("\n=== Example 6: Error Analysis ===")
    
    documents = [
        "The Earth orbits the Sun in approximately 365.25 days. This orbital period is called a year.",
        "The Moon orbits the Earth in about 27.3 days. This is called the sidereal month.",
        "The solar system consists of the Sun and the objects that orbit it, including planets, moons, asteroids, and comets."
    ]
    
    config = EvaluationConfig(
        model_name="gpt-3.5-turbo",
        batch_size=5,
        results_dir="./eval-results/example6"
    )
    
    rag_system = RAGSystem(documents, config)
    generator = TestSetGenerator(config)
    testset = generator.generate_from_documents(documents, num_questions=10)
    
    evaluator = RagasEvaluator(config)
    results = evaluator.evaluate_rag_system(rag_system, testset)
    
    # Analyze individual responses
    print("Detailed Analysis:")
    for i, row in results['eval_data'].iterrows():
        print(f"\nQuestion {i+1}: {row['question']}")
        print(f"Answer: {row['answer'][:150]}...")
        print(f"Context length: {len(row['contexts'][0])} characters")
    
    return results

def main():
    """Run all examples"""
    print("Ragas Evaluation Examples")
    print("=" * 50)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Some examples may not work.")
        print("Set it with: export OPENAI_API_KEY='your-key'")
    
    # Run examples
    try:
        example_1_basic_evaluation()
        example_2_custom_documents()
        example_3_multiple_rag_systems()
        example_4_parameter_sensitivity()
        example_5_custom_questions()
        example_6_error_analysis()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("Check the ./eval-results/ directory for detailed outputs.")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have set your OPENAI_API_KEY and installed all dependencies.")

if __name__ == "__main__":
    main() 