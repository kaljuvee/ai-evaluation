"""
Ragas Synthetic Data Generation for RAG Evaluation

This script demonstrates how to generate synthetic test datasets for RAG evaluation
using Ragas, following the official documentation examples.
"""

import os
import asyncio
import sys
import io
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

# Ragas imports
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.testset.synthesizers import default_query_distribution

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


class RagasSyntheticDataGenerator:
    """
    A class to generate synthetic test datasets for RAG evaluation using Ragas.
    """
    
    def __init__(self, openai_api_key: str = None):
        """
        Initialize the synthetic data generator with OpenAI API key.
        
        Args:
            openai_api_key: OpenAI API key. If None, will try to get from environment.
        """
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        elif not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
        # Initialize LLM and embeddings for generation
        self.generator_llm = LangchainLLMWrapper(ChatOpenAI(
            model="gpt-4o",
            temperature=0.4
        ))
        
        self.generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
        
        # Initialize testset generator
        self.generator = TestsetGenerator(
            llm=self.generator_llm, 
            embedding_model=self.generator_embeddings
        )
    
    def load_sample_documents(self, path: str = "Sample_Docs_Markdown/") -> List:
        """
        Load documents from a directory using LangChain DirectoryLoader.
        
        Args:
            path: Path to the directory containing documents
            
        Returns:
            List of loaded documents
        """
        try:
            loader = DirectoryLoader(path, glob="**/*.md")
            docs = loader.load()
            print(f"✅ Loaded {len(docs)} documents from {path}")
            return docs
        except Exception as e:
            print(f"❌ Error loading documents: {e}")
            print("Creating sample documents instead...")
            return self.create_sample_documents()
    
    def create_sample_documents(self) -> List:
        """
        Create sample documents for testing when no external documents are available.
        
        Returns:
            List of sample documents
        """
        from langchain.schema import Document
        
        sample_docs = [
            Document(
                page_content="""
                # Machine Learning Fundamentals
                
                Machine learning is a subset of artificial intelligence that enables computers to learn 
                and make decisions without being explicitly programmed. It involves algorithms that can 
                identify patterns in data and make predictions or decisions based on those patterns.
                
                ## Types of Machine Learning
                
                There are three main types of machine learning:
                1. **Supervised Learning**: Uses labeled training data to learn patterns
                2. **Unsupervised Learning**: Finds hidden patterns in unlabeled data
                3. **Reinforcement Learning**: Learns through interaction with an environment
                
                ## Key Concepts
                
                - **Training Data**: The dataset used to train the model
                - **Features**: Input variables used for prediction
                - **Labels**: Output variables we want to predict
                - **Model**: The algorithm that learns from the data
                """,
                metadata={"source": "ml_fundamentals.md", "title": "Machine Learning Fundamentals"}
            ),
            Document(
                page_content="""
                # Deep Learning and Neural Networks
                
                Deep learning is a subset of machine learning that uses artificial neural networks 
                with multiple layers to model and understand complex patterns in data.
                
                ## Neural Network Architecture
                
                Neural networks consist of:
                - **Input Layer**: Receives the input data
                - **Hidden Layers**: Process the data through multiple transformations
                - **Output Layer**: Produces the final prediction
                
                ## Popular Deep Learning Frameworks
                
                1. **TensorFlow**: Developed by Google, widely used in production
                2. **PyTorch**: Developed by Facebook, popular in research
                3. **Keras**: High-level API for building neural networks
                
                ## Applications
                
                Deep learning is used in:
                - Computer Vision
                - Natural Language Processing
                - Speech Recognition
                - Autonomous Vehicles
                """,
                metadata={"source": "deep_learning.md", "title": "Deep Learning and Neural Networks"}
            ),
            Document(
                page_content="""
                # Natural Language Processing (NLP)
                
                Natural Language Processing is a field of AI that focuses on the interaction 
                between computers and human language. It enables machines to understand, 
                interpret, and generate human language.
                
                ## Core NLP Tasks
                
                1. **Text Classification**: Categorizing text into predefined classes
                2. **Named Entity Recognition**: Identifying entities like names, organizations
                3. **Machine Translation**: Translating text between languages
                4. **Question Answering**: Answering questions based on context
                5. **Text Generation**: Creating human-like text
                
                ## Key Technologies
                
                - **Word Embeddings**: Vector representations of words
                - **Transformers**: Architecture for processing sequential data
                - **BERT**: Bidirectional Encoder Representations from Transformers
                - **GPT**: Generative Pre-trained Transformer
                
                ## Applications
                
                NLP is used in:
                - Chatbots and Virtual Assistants
                - Sentiment Analysis
                - Document Summarization
                - Language Translation Services
                """,
                metadata={"source": "nlp_basics.md", "title": "Natural Language Processing"}
            )
        ]
        
        print(f"✅ Created {len(sample_docs)} sample documents")
        return sample_docs
    
    def create_knowledge_graph(self, docs: List) -> KnowledgeGraph:
        """
        Create a knowledge graph from documents and enrich it with transformations.
        
        Args:
            docs: List of documents to process
            
        Returns:
            Enriched KnowledgeGraph object
        """
        print("🔧 Creating Knowledge Graph...")
        
        # Initialize empty knowledge graph
        kg = KnowledgeGraph()
        print(f"Initial KnowledgeGraph: {kg}")
        
        # Add documents to knowledge graph
        for doc in docs:
            kg.nodes.append(
                Node(
                    type=NodeType.DOCUMENT,
                    properties={
                        "page_content": doc.page_content, 
                        "document_metadata": doc.metadata
                    }
                )
            )
        
        print(f"KnowledgeGraph after adding documents: {kg}")
        
        # Apply default transformations to enrich the knowledge graph
        print("🔄 Applying transformations to enrich knowledge graph...")
        trans = default_transforms(
            documents=docs, 
            llm=self.generator_llm, 
            embedding_model=self.generator_embeddings
        )
        apply_transforms(kg, trans)
        
        print(f"Enriched KnowledgeGraph: {kg}")
        return kg
    
    def save_knowledge_graph(self, kg: KnowledgeGraph, filename: str = "knowledge_graph.json"):
        """
        Save knowledge graph to file.
        
        Args:
            kg: KnowledgeGraph to save
            filename: Name of the file to save to
        """
        kg.save(filename)
        print(f"✅ Knowledge graph saved to {filename}")
    
    def load_knowledge_graph(self, filename: str = "knowledge_graph.json") -> KnowledgeGraph:
        """
        Load knowledge graph from file.
        
        Args:
            filename: Name of the file to load from
            
        Returns:
            Loaded KnowledgeGraph object
        """
        kg = KnowledgeGraph.load(filename)
        print(f"✅ Knowledge graph loaded from {filename}: {kg}")
        return kg
    
    def generate_testset(self, 
                        docs: List, 
                        testset_size: int = 10, 
                        save_kg: bool = True,
                        kg_filename: str = "knowledge_graph.json") -> Any:
        """
        Generate a synthetic testset for RAG evaluation.
        
        Args:
            docs: List of documents to use for generation
            testset_size: Number of test samples to generate
            save_kg: Whether to save the knowledge graph
            kg_filename: Filename for saving knowledge graph
            
        Returns:
            Generated testset dataset
        """
        print(f"🚀 Generating testset with {testset_size} samples...")
        
        # Create knowledge graph
        kg = self.create_knowledge_graph(docs)
        
        # Save knowledge graph if requested
        if save_kg:
            self.save_knowledge_graph(kg, kg_filename)
        
        # Create testset generator with knowledge graph
        generator_with_kg = TestsetGenerator(
            llm=self.generator_llm, 
            embedding_model=self.generator_embeddings, 
            knowledge_graph=kg
        )
        
        # Get default query distribution
        query_distribution = default_query_distribution(self.generator_llm)
        print(f"Query distribution: {query_distribution}")
        
        # Generate testset
        testset = generator_with_kg.generate(
            testset_size=testset_size, 
            query_distribution=query_distribution
        )
        
        print(f"✅ Generated testset with {len(testset)} samples")
        return testset
    
    def generate_testset_simple(self, docs: List, testset_size: int = 10) -> Any:
        """
        Generate a testset using the simple method (without knowledge graph).
        
        Args:
            docs: List of documents to use for generation
            testset_size: Number of test samples to generate
            
        Returns:
            Generated testset dataset
        """
        print(f"🚀 Generating simple testset with {testset_size} samples...")
        
        dataset = self.generator.generate_with_langchain_docs(docs, testset_size=testset_size)
        
        print(f"✅ Generated testset with {len(dataset)} samples")
        return dataset
    
    def analyze_testset(self, testset) -> pd.DataFrame:
        """
        Analyze the generated testset and return as pandas DataFrame.
        
        Args:
            testset: Generated testset to analyze
            
        Returns:
            Pandas DataFrame with testset analysis
        """
        df = testset.to_pandas()
        print(f"📊 Testset analysis:")
        print(f"   - Total samples: {len(df)}")
        print(f"   - Columns: {list(df.columns)}")
        print(f"   - Sample data types: {df.dtypes.to_dict()}")
        return df
    
    def save_testset(self, testset, filename: str = "generated_testset.csv"):
        """
        Save testset to CSV file.
        
        Args:
            testset: Testset to save
            filename: Name of the CSV file
        """
        df = self.analyze_testset(testset)
        df.to_csv(filename, index=False)
        print(f"✅ Testset saved to {filename}")
        return df
    
    def display_sample_queries(self, testset, num_samples: int = 3):
        """
        Display sample queries from the generated testset.
        
        Args:
            testset: Generated testset
            num_samples: Number of sample queries to display
        """
        df = testset.to_pandas()
        
        print(f"\n📝 Sample Queries (showing {min(num_samples, len(df))} of {len(df)}):")
        print("=" * 80)
        
        for i, row in df.head(num_samples).iterrows():
            print(f"\nQuery {i+1}:")
            print(f"Question: {row.get('question', 'N/A')}")
            print(f"Ground Truth: {row.get('ground_truth', 'N/A')}")
            if 'contexts' in row:
                print(f"Contexts: {len(row['contexts'])} context(s)")
            print("-" * 40)


async def main():
    """
    Main function to demonstrate Ragas synthetic data generation.
    """
    # Prepare to capture all console output
    output_buffer = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = output_buffer

    try:
        print("🚀 Ragas Synthetic Data Generation Demo")
        print("=" * 60)
        
        # Initialize generator
        generator = RagasSyntheticDataGenerator()
        
        # Load or create documents
        print("\n1. Loading Documents")
        print("-" * 30)
        docs = generator.load_sample_documents()
        
        # Generate testset using simple method
        print("\n2. Simple Testset Generation")
        print("-" * 30)
        simple_testset = generator.generate_testset_simple(docs, testset_size=5)
        
        # Analyze and display simple testset
        print("\n3. Simple Testset Analysis")
        print("-" * 30)
        generator.display_sample_queries(simple_testset, num_samples=3)
        
        # Generate testset using knowledge graph method
        print("\n4. Knowledge Graph Testset Generation")
        print("-" * 30)
        kg_testset = generator.generate_testset(docs, testset_size=5, save_kg=True)
        
        # Analyze and display knowledge graph testset
        print("\n5. Knowledge Graph Testset Analysis")
        print("-" * 30)
        generator.display_sample_queries(kg_testset, num_samples=3)
        
        # Save results
        print("\n6. Saving Results")
        print("-" * 30)
        simple_df = generator.save_testset(simple_testset, "simple_testset.csv")
        kg_df = generator.save_testset(kg_testset, "knowledge_graph_testset.csv")
        
        print("\n✅ Synthetic data generation completed successfully!")
        print(f"📁 Files created:")
        print(f"   - knowledge_graph.json")
        print(f"   - simple_testset.csv")
        print(f"   - knowledge_graph_testset.csv")
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        raise
    finally:
        # Restore stdout
        sys.stdout = sys_stdout
        # Write output to markdown file
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        output_path = f"../test-data/ragas/synthetic_data_results_{timestamp}.md"
        with open(output_path, "w") as f:
            f.write("```markdown\n")
            f.write(output_buffer.getvalue())
            f.write("\n```")
        print(f"\n[Console output written to {output_path}]")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
