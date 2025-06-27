# Ragas Synthetic Data Generation

This module provides a comprehensive implementation of synthetic test dataset generation for RAG evaluation using Ragas, following the official documentation examples.

## Features

- **Document Loading**: Support for loading documents from directories or creating sample documents
- **Knowledge Graph Creation**: Build and enrich knowledge graphs from documents
- **Simple Testset Generation**: Generate testsets directly from documents
- **Advanced Testset Generation**: Use knowledge graphs for more sophisticated testset generation
- **Query Distribution**: Control the types of queries generated (single-hop, multi-hop, etc.)
- **Results Export**: Export results to CSV and analyze generated testsets
- **Flexible Configuration**: Easy setup with OpenAI API

## Installation

The required dependencies are already included in `requirements.txt`. Make sure you have activated your virtual environment:

```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r ragas/requirements.txt
```

## Setup

### 1. OpenAI API Key

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or set it in your Python script:

```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

### 2. Test Installation

Run the test script to verify everything is working:

```bash
cd ragas
python test_synthetic_data.py
```

## Usage

### Basic Usage

```python
import asyncio
from ragas_synhetic_data_eval import RagasSyntheticDataGenerator

async def main():
    # Initialize generator
    generator = RagasSyntheticDataGenerator()
    
    # Load or create documents
    docs = generator.load_sample_documents()
    
    # Generate simple testset
    simple_testset = generator.generate_testset_simple(docs, testset_size=10)
    
    # Generate advanced testset with knowledge graph
    kg_testset = generator.generate_testset(docs, testset_size=10)
    
    # Analyze and save results
    generator.save_testset(simple_testset, "simple_testset.csv")
    generator.save_testset(kg_testset, "knowledge_graph_testset.csv")

# Run the generation
asyncio.run(main())
```

### Document Loading

```python
from ragas_synhetic_data_eval import RagasSyntheticDataGenerator

generator = RagasSyntheticDataGenerator()

# Option 1: Load from directory
docs = generator.load_sample_documents("path/to/your/documents/")

# Option 2: Use sample documents (created automatically)
docs = generator.create_sample_documents()
```

### Knowledge Graph Creation

```python
# Create knowledge graph from documents
kg = generator.create_knowledge_graph(docs)

# Save knowledge graph
generator.save_knowledge_graph(kg, "my_knowledge_graph.json")

# Load knowledge graph
loaded_kg = generator.load_knowledge_graph("my_knowledge_graph.json")
```

### Testset Generation

```python
# Simple method (without knowledge graph)
simple_testset = generator.generate_testset_simple(docs, testset_size=10)

# Advanced method (with knowledge graph)
kg_testset = generator.generate_testset(docs, testset_size=10, save_kg=True)

# Analyze generated testset
df = generator.analyze_testset(kg_testset)
generator.display_sample_queries(kg_testset, num_samples=5)
```

### Custom Query Distribution

```python
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer
)

# Define custom query distribution
custom_distribution = [
    (SingleHopSpecificQuerySynthesizer(generator.generator_llm), 0.6),
    (MultiHopAbstractQuerySynthesizer(generator.generator_llm), 0.2),
    (MultiHopSpecificQuerySynthesizer(generator.generator_llm), 0.2),
]

# Use custom distribution in generation
testset = generator.generator.generate(
    testset_size=10, 
    query_distribution=custom_distribution
)
```

## Methods Explained

### 1. Simple Testset Generation

- **What it does**: Generates testsets directly from documents without knowledge graph
- **Pros**: Faster, simpler, less resource-intensive
- **Cons**: May generate less sophisticated queries
- **Use case**: Quick prototyping, simple RAG evaluation

### 2. Knowledge Graph Testset Generation

- **What it does**: Creates a knowledge graph, enriches it with transformations, then generates testsets
- **Pros**: More sophisticated queries, better coverage, multi-hop capabilities
- **Cons**: Slower, more resource-intensive, requires API calls
- **Use case**: Production evaluation, complex RAG systems

### 3. Query Types

- **Single-hop Specific**: Direct questions about specific facts
- **Multi-hop Abstract**: Questions requiring reasoning across multiple documents
- **Multi-hop Specific**: Complex questions about specific relationships

## Example Output

```
🚀 Ragas Synthetic Data Generation Demo
============================================================

1. Loading Documents
------------------------------
✅ Created 3 sample documents

2. Simple Testset Generation
------------------------------
🚀 Generating simple testset with 5 samples...
✅ Generated testset with 5 samples

3. Simple Testset Analysis
------------------------------
📝 Sample Queries (showing 3 of 5):
================================================================================

Query 1:
Question: What are the three main types of machine learning?
Ground Truth: The three main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning.
Contexts: 2 context(s)
----------------------------------------

Query 2:
Question: What is the difference between TensorFlow and PyTorch?
Ground Truth: TensorFlow was developed by Google and is widely used in production, while PyTorch was developed by Facebook and is popular in research.
Contexts: 1 context(s)
----------------------------------------

4. Knowledge Graph Testset Generation
------------------------------
🔧 Creating Knowledge Graph...
Initial KnowledgeGraph: KnowledgeGraph(nodes: 0, relationships: 0)
KnowledgeGraph after adding documents: KnowledgeGraph(nodes: 3, relationships: 0)
🔄 Applying transformations to enrich knowledge graph...
Enriched KnowledgeGraph: KnowledgeGraph(nodes: 45, relationships: 523)
✅ Knowledge graph saved to knowledge_graph.json
🚀 Generating testset with 5 samples...
Query distribution: [(SingleHopSpecificQuerySynthesizer(llm=llm), 0.5), (MultiHopAbstractQuerySynthesizer(llm=llm), 0.25), (MultiHopSpecificQuerySynthesizer(llm=llm), 0.25)]
✅ Generated testset with 5 samples

5. Knowledge Graph Testset Analysis
------------------------------
📝 Sample Queries (showing 3 of 5):
================================================================================

Query 1:
Question: How do neural networks process data through their architecture?
Ground Truth: Neural networks process data through an input layer that receives the data, hidden layers that process it through multiple transformations, and an output layer that produces the final prediction.
Contexts: 3 context(s)
----------------------------------------

6. Saving Results
------------------------------
📊 Testset analysis:
   - Total samples: 5
   - Columns: ['question', 'ground_truth', 'contexts']
   - Sample data types: {'question': object, 'ground_truth': object, 'contexts': object}
✅ Testset saved to simple_testset.csv
✅ Testset saved to knowledge_graph_testset.csv

✅ Synthetic data generation completed successfully!
📁 Files created:
   - knowledge_graph.json
   - simple_testset.csv
   - knowledge_graph_testset.csv
```

## Customization

### Custom Documents

```python
from langchain.schema import Document

custom_docs = [
    Document(
        page_content="Your document content here...",
        metadata={"source": "custom_doc.md", "title": "Custom Document"}
    )
]

# Use custom documents
testset = generator.generate_testset_simple(custom_docs, testset_size=10)
```

### Custom Transformations

```python
from ragas.testset.transforms import apply_transforms

# Apply custom transformations to knowledge graph
custom_transforms = [
    # Add your custom transformations here
]

apply_transforms(kg, custom_transforms)
```

### Custom LLM Configuration

```python
# Modify LLM settings
generator.generator_llm = LangchainLLMWrapper(ChatOpenAI(
    model="gpt-4o-mini",  # Use different model
    temperature=0.7,      # Adjust creativity
    max_tokens=1000       # Adjust response length
))
```

## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   ```
   ValueError: OpenAI API key not found
   ```
   **Solution**: Set your `OPENAI_API_KEY` environment variable

2. **Document Loading Errors**
   ```
   Error loading documents from path
   ```
   **Solution**: The script will automatically create sample documents

3. **Knowledge Graph Creation Errors**
   ```
   Error during knowledge graph creation
   ```
   **Solution**: Check your API key and internet connection

### Performance Tips

- Use simple testset generation for quick iterations
- Use knowledge graph generation for final evaluation
- Adjust `testset_size` based on your needs
- Cache knowledge graphs for reuse
- Use appropriate query distributions for your use case

## File Structure

After running the script, you'll have:

```
ragas/
├── knowledge_graph.json          # Saved knowledge graph
├── simple_testset.csv           # Simple testset results
├── knowledge_graph_testset.csv  # Advanced testset results
└── ../test-data/ragas/
    └── synthetic_data_results_<timestamp>.md  # Console output
```

## References

- [Ragas Testset Generation Documentation](https://docs.ragas.io/en/latest/getstarted/rag_testset_generation/#testset-generation)
- [Ragas GitHub Repository](https://github.com/explodinggradients/ragas)
- [LangChain Document Loaders](https://python.langchain.com/docs/modules/data_connection/document_loaders/)

## License

This implementation follows the same license as the main project. 