"""
Ragas Evaluation System for RAG Applications

This module provides a comprehensive evaluation framework using Ragas (by Exploding Gradients)
to assess RAG system performance across multiple metrics including:
- Answer Correctness
- Faithfulness  
- Context Precision
- Context Recall

Features:
- Synthetic test set generation
- LangChain integration
- HuggingFace model support
- Comprehensive evaluation pipeline
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    faithfulness,
    context_precision,
    context_recall,
    context_relevancy,
    answer_relevancy
)
from ragas.metrics.critique import CritiqueTone
from ragas.metrics.answer_correctness import AnswerCorrectness
from ragas.metrics.faithfulness import Faithfulness
from ragas.metrics.context_precision import ContextPrecision
from ragas.metrics.context_recall import ContextRecall

from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import Simple, Reasoning
from ragas.testset.graph import RAGASGraph
from ragas.testset.graphs import entity, keyword, question_type
from ragas.testset.synthesizer import Synthesizer

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import VectorStoreRetriever
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

@dataclass
class EvaluationConfig:
    """Configuration for RAG evaluation"""
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.0
    max_tokens: int = 1000
    batch_size: int = 10
    cache_dir: str = "./cache"
    results_dir: str = "./eval-results"

class RAGSystem:
    """Simple RAG system for demonstration"""
    
    def __init__(self, documents: List[str], config: EvaluationConfig):
        self.config = config
        self.documents = documents
        self.vectorstore = None
        self.retriever = None
        self.llm = ChatOpenAI(
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        self._setup_rag()
    
    def _setup_rag(self):
        """Setup the RAG system with documents"""
        # Create documents
        docs = [Document(page_content=doc, metadata={"source": f"doc_{i}"}) 
                for i, doc in enumerate(self.documents)]
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        
        # Create embeddings and vectorstore
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = FAISS.from_documents(splits, embeddings)
        self.retriever = VectorStoreRetriever(
            vectorstore=self.vectorstore,
            search_type="similarity",
            search_kwargs={"k": 5}
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system"""
        # Retrieve relevant documents
        retrieved_docs = self.retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        
        # Generate answer
        messages = [
            SystemMessage(content="You are a helpful assistant. Answer the question based on the provided context. If the context doesn't contain enough information, say so."),
            HumanMessage(content=f"Context: {context}\n\nQuestion: {question}")
        ]
        
        response = self.llm.invoke(messages)
        
        return {
            "question": question,
            "answer": response.content,
            "context": context,
            "retrieved_docs": [doc.page_content for doc in retrieved_docs]
        }

class TestSetGenerator:
    """Generate synthetic test sets for RAG evaluation"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.generator = TestsetGenerator.with_openai(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-3.5-turbo"
        )
    
    def generate_from_documents(self, documents: List[str], num_questions: int = 50) -> pd.DataFrame:
        """Generate test questions from documents"""
        # Create a simple graph for question generation
        graph = RAGASGraph()
        
        # Add entities and keywords
        for i, doc in enumerate(documents):
            graph.add_node(entity(f"doc_{i}", doc))
        
        # Generate questions
        testset = self.generator.generate(
            graph,
            num_questions=num_questions,
            distributions={
                "question_type": {
                    "simple": 0.3,
                    "reasoning": 0.4,
                    "multi_context": 0.3
                }
            }
        )
        
        return testset.to_pandas()
    
    def generate_synthetic_dataset(self) -> pd.DataFrame:
        """Generate a synthetic dataset for demonstration"""
        # Sample documents about AI and machine learning
        documents = [
            "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines. Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. Deep learning, a subset of machine learning, uses neural networks with multiple layers to model complex patterns.",
            
            "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It includes tasks like text classification, sentiment analysis, machine translation, and question answering. Modern NLP systems often use transformer architectures like BERT and GPT.",
            
            "Retrieval-Augmented Generation (RAG) combines information retrieval with text generation. It first retrieves relevant documents from a knowledge base, then uses that context to generate more accurate and factual responses. RAG systems are particularly useful for question-answering tasks where up-to-date information is crucial.",
            
            "Vector databases store and retrieve high-dimensional vectors representing text embeddings. They enable efficient similarity search, making them essential for modern information retrieval systems. Popular vector databases include Pinecone, Weaviate, and FAISS.",
            
            "Evaluation metrics for AI systems include accuracy, precision, recall, and F1-score. For RAG systems, additional metrics like faithfulness, answer correctness, and context precision are important to assess the quality of generated responses."
        ]
        
        return self.generate_from_documents(documents, num_questions=30)

class RagasEvaluator:
    """Main evaluator class using Ragas metrics"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.metrics = [
            answer_correctness,
            faithfulness,
            context_precision,
            context_recall,
            context_relevancy,
            answer_relevancy
        ]
        
        # Ensure results directory exists
        Path(config.results_dir).mkdir(parents=True, exist_ok=True)
        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def evaluate_rag_system(self, rag_system: RAGSystem, testset: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate a RAG system using the provided test set"""
        
        # Prepare evaluation data
        eval_data = []
        for _, row in testset.iterrows():
            result = rag_system.query(row['question'])
            eval_data.append({
                'question': row['question'],
                'answer': result['answer'],
                'contexts': [result['context']],
                'ground_truth': row.get('answer', '')  # If available
            })
        
        # Convert to DataFrame
        eval_df = pd.DataFrame(eval_data)
        
        # Run evaluation
        results = evaluate(
            eval_df,
            metrics=self.metrics,
            run_config={
                "cache_dir": self.config.cache_dir,
                "batch_size": self.config.batch_size
            }
        )
        
        return {
            'metrics': results,
            'eval_data': eval_df,
            'testset_size': len(testset)
        }
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run a comprehensive evaluation with synthetic data"""
        
        # Generate test set
        print("Generating synthetic test set...")
        generator = TestSetGenerator(self.config)
        testset = generator.generate_synthetic_dataset()
        
        # Create RAG system
        print("Setting up RAG system...")
        documents = [
            "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines. Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.",
            "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It includes tasks like text classification, sentiment analysis, and machine translation.",
            "Retrieval-Augmented Generation (RAG) combines information retrieval with text generation. It first retrieves relevant documents from a knowledge base, then uses that context to generate more accurate responses.",
            "Vector databases store and retrieve high-dimensional vectors representing text embeddings. They enable efficient similarity search for modern information retrieval systems.",
            "Evaluation metrics for AI systems include accuracy, precision, recall, and F1-score. For RAG systems, additional metrics like faithfulness and context precision are important."
        ]
        
        rag_system = RAGSystem(documents, self.config)
        
        # Run evaluation
        print("Running evaluation...")
        results = self.evaluate_rag_system(rag_system, testset)
        
        # Save results
        self._save_results(results, testset)
        
        return results
    
    def _save_results(self, results: Dict[str, Any], testset: pd.DataFrame):
        """Save evaluation results to files"""
        
        # Save metrics
        metrics_file = Path(self.config.results_dir) / "ragas_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(results['metrics'], f, indent=2)
        
        # Save evaluation data
        eval_data_file = Path(self.config.results_dir) / "evaluation_data.csv"
        results['eval_data'].to_csv(eval_data_file, index=False)
        
        # Save test set
        testset_file = Path(self.config.results_dir) / "testset.csv"
        testset.to_csv(testset_file, index=False)
        
        print(f"Results saved to {self.config.results_dir}")

def main():
    """Main function to run the evaluation"""
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Some features may not work.")
    
    # Configuration
    config = EvaluationConfig(
        model_name="gpt-3.5-turbo",
        temperature=0.0,
        max_tokens=1000,
        batch_size=5,
        cache_dir="./cache",
        results_dir="./eval-results"
    )
    
    # Run evaluation
    evaluator = RagasEvaluator(config)
    results = evaluator.run_comprehensive_evaluation()
    
    # Print results
    print("\n=== RAGAS Evaluation Results ===")
    for metric_name, metric_value in results['metrics'].items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    print(f"\nTest set size: {results['testset_size']}")
    print(f"Results saved to: {config.results_dir}")

if __name__ == "__main__":
    main()
