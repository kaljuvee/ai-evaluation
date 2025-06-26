"""
DeepEval RAG Integration Example

This example shows how to integrate DeepEval with a real RAG pipeline using LangChain.
It demonstrates evaluating actual RAG responses against expected outputs.
"""

import os
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase


class RAGPipeline:
    """Simple RAG pipeline for demonstration"""
    
    def __init__(self, documents: List[str]):
        self.documents = documents
        self.vectorstore = None
        self.qa_chain = None
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Set up the RAG pipeline"""
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.create_documents(self.documents)
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.from_documents(texts, embeddings)
        
        # Create QA chain
        llm = ChatOpenAI(temperature=0)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3})
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG pipeline"""
        # Get response
        response = self.qa_chain.run(question)
        
        # Get retrieved documents
        docs = self.qa_chain.retriever.get_relevant_documents(question)
        context = [doc.page_content for doc in docs]
        
        return {
            "answer": response,
            "context": context,
            "sources": [doc.metadata for doc in docs]
        }


def create_test_dataset() -> List[Dict[str, Any]]:
    """Create a test dataset for RAG evaluation"""
    return [
        {
            "question": "What is the main benefit of RAG systems?",
            "expected_answer": "RAG systems provide improved accuracy by retrieving relevant context from external knowledge sources.",
            "category": "accuracy"
        },
        {
            "question": "How does vector similarity work in retrieval?",
            "expected_answer": "Vector similarity works by converting text into numerical vectors and finding the most similar vectors using distance metrics like cosine similarity.",
            "category": "technical"
        },
        {
            "question": "What are the key components of a RAG pipeline?",
            "expected_answer": "A RAG pipeline typically consists of a retriever to find relevant documents, a generator to create responses, and a knowledge base to store information.",
            "category": "architecture"
        }
    ]


def create_sample_documents() -> List[str]:
    """Create sample documents for the RAG pipeline"""
    return [
        """
        RAG (Retrieval-Augmented Generation) systems combine the power of large language models 
        with external knowledge retrieval to improve accuracy and reduce hallucinations. 
        The main benefit of RAG systems is their ability to provide more accurate responses 
        by retrieving relevant context from external knowledge sources.
        """,
        """
        Vector similarity is a core component of RAG systems. Text is converted into numerical 
        vectors using embedding models, and similarity is calculated using distance metrics 
        like cosine similarity or Euclidean distance. The most similar vectors are retrieved 
        as relevant context for the generation process.
        """,
        """
        A typical RAG pipeline consists of several key components: a retriever that finds 
        relevant documents from a knowledge base, a generator that creates responses based 
        on the retrieved context, and a knowledge base that stores the information to be 
        retrieved. This architecture allows for more accurate and up-to-date responses.
        """
    ]


def evaluate_rag_pipeline():
    """Evaluate a RAG pipeline using DeepEval"""
    print("🚀 DeepEval RAG Pipeline Evaluation")
    print("=" * 50)
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OpenAI API key not set. Please set OPENAI_API_KEY environment variable.")
        return
    
    try:
        # Create RAG pipeline
        print("📚 Setting up RAG pipeline...")
        documents = create_sample_documents()
        rag_pipeline = RAGPipeline(documents)
        print("✅ RAG pipeline created successfully")
        
        # Create test dataset
        print("\n📋 Creating test dataset...")
        test_data = create_test_dataset()
        print(f"✅ Created {len(test_data)} test cases")
        
        # Generate test cases
        print("\n🔍 Generating test cases...")
        test_cases = []
        
        for i, data in enumerate(test_data):
            print(f"Processing test case {i+1}/{len(test_data)}: {data['question']}")
            
            # Query the RAG pipeline
            result = rag_pipeline.query(data['question'])
            
            # Create test case
            test_case = LLMTestCase(
                input=data['question'],
                actual_output=result['answer'],
                expected_output=data['expected_answer'],
                retrieval_context=result['context']
            )
            test_cases.append(test_case)
            
            print(f"  Actual: {result['answer'][:100]}...")
            print(f"  Expected: {data['expected_answer'][:100]}...")
            print(f"  Context retrieved: {len(result['context'])} documents")
        
        # Define metrics
        print("\n📊 Setting up evaluation metrics...")
        metrics = [
            FaithfulnessMetric(threshold=0.7),
            AnswerRelevancyMetric(threshold=0.7),
            ContextualRelevancyMetric(threshold=0.7)
        ]
        
        # Run evaluation
        print("\n🎯 Running evaluation...")
        results = evaluate(test_cases, metrics)
        
        # Display results
        print("\n📈 Evaluation Results:")
        print("=" * 30)
        
        for i, test_case in enumerate(test_cases):
            print(f"\nTest Case {i+1}: {test_data[i]['question']}")
            print(f"Category: {test_data[i]['category']}")
            print(f"Actual Output: {test_case.actual_output}")
            print(f"Expected Output: {test_case.expected_output}")
            print(f"Retrieval Context: {len(test_case.retrieval_context)} documents")
        
        print(f"\nOverall Results: {results}")
        
        # Detailed metric analysis
        print("\n🔍 Detailed Metric Analysis:")
        print("=" * 30)
        
        for metric_name, metric_results in results.items():
            if hasattr(metric_results, 'score'):
                print(f"{metric_name}: {metric_results.score:.3f}")
            else:
                print(f"{metric_name}: {metric_results}")
        
        print("\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        print("\n💡 Make sure you have:")
        print("   - Set OPENAI_API_KEY environment variable")
        print("   - Installed all required packages: pip install -r requirements.txt")


def evaluate_with_custom_metrics():
    """Example of using custom metrics for RAG evaluation"""
    print("\n🎨 Custom Metrics Example")
    print("=" * 30)
    
    from deepeval.metrics import GEval
    
    # Create custom RAG evaluation metric
    custom_metric = GEval(
        name="RAG Quality Score",
        criteria="""
        Evaluate the RAG response quality considering:
        1. Accuracy: Is the information factually correct?
        2. Completeness: Does it answer the full question?
        3. Context Usage: Does it properly use the provided context?
        4. Clarity: Is the response clear and well-structured?
        5. Relevance: Is the response relevant to the question?
        
        Score from 0-1 where:
        0.0-0.3: Poor quality, inaccurate or incomplete
        0.4-0.6: Acceptable quality with some issues
        0.7-0.8: Good quality, mostly accurate and complete
        0.9-1.0: Excellent quality, accurate, complete, and well-structured
        """,
        evaluation_params=["actual_output", "expected_output", "retrieval_context"],
        threshold=0.7
    )
    
    # Create a simple test case
    test_case = LLMTestCase(
        input="What is the main benefit of RAG?",
        actual_output="RAG systems provide improved accuracy by retrieving relevant context from external sources.",
        expected_output="RAG systems provide improved accuracy by retrieving relevant context from external knowledge sources.",
        retrieval_context=[
            "RAG systems combine LLMs with external knowledge retrieval.",
            "The main benefit is improved accuracy through context retrieval.",
            "RAG reduces hallucinations by grounding responses in retrieved information."
        ]
    )
    
    # Evaluate with custom metric
    results = evaluate([test_case], [custom_metric])
    print(f"Custom RAG Quality Score: {results}")


def main():
    """Run the RAG integration example"""
    print("🚀 DeepEval RAG Integration Example")
    print("=" * 50)
    
    # Run main evaluation
    evaluate_rag_pipeline()
    
    # Run custom metrics example
    evaluate_with_custom_metrics()
    
    print("\n🎉 Example completed!")
    print("\nNext steps:")
    print("1. Modify the test dataset for your specific use case")
    print("2. Adjust metric thresholds based on your requirements")
    print("3. Integrate with your own RAG pipeline")
    print("4. Set up CI/CD for automated evaluation")


if __name__ == "__main__":
    main() 