# Ragas: A Simple Guide to RAG Evaluation

## What is Ragas?

**Ragas** (RAG Assessment) is a framework specifically designed to evaluate **RAG (Retrieval-Augmented Generation)** systems. Think of it as a specialized toolkit for measuring how well your AI retrieves information and generates answers.

### What is RAG?

**RAG** combines two steps:
1. **Retrieval**: Finding relevant documents from a knowledge base
2. **Generation**: Using those documents to generate an answer

**Example RAG Flow:**
```
User Question: "What are the benefits of exercise?"
    ↓
Retrieval: Find relevant documents about exercise
    ↓
Retrieved Docs: ["Exercise improves heart health...", "Regular activity boosts mood..."]
    ↓
Generation: LLM creates answer using these documents
    ↓
Answer: "Exercise improves cardiovascular health and mental well-being..."
```

### Why Use Ragas?

- ✅ **RAG-specific metrics**: Designed for retrieval + generation systems
- ✅ **Multiple perspectives**: Evaluates both retrieval quality and answer quality
- ✅ **Simple to use**: Easy setup with clear metrics
- ✅ **Synthetic data generation**: Create test cases automatically
- ✅ **LangChain integration**: Works seamlessly with popular frameworks

## Core Metrics Explained (In Simple Terms)

Ragas provides **6 main metrics** to evaluate different aspects of your RAG system:

### 1. **Faithfulness** 📝
**Question**: Does the answer stick to the facts in the retrieved documents?

**Simple Explanation**: Checks if your AI is making things up or staying true to the source documents.

**Example:**
```python
Question: "What is the capital of France?"
Retrieved Context: ["Paris is the capital and largest city of France."]
Answer: "Paris is the capital of France and has a population of 50 million."
                                                    ↑ HALLUCINATION! ↑
Faithfulness Score: Low (0.5) - The population claim isn't in the context
```

**When to use**: To prevent hallucinations and ensure factual accuracy.

---

### 2. **Answer Relevancy** 🎯
**Question**: Does the answer actually address the user's question?

**Simple Explanation**: Checks if the AI is answering what was asked, not going off-topic.

**Example:**
```python
Question: "How do I reset my password?"
Answer: "Our company was founded in 2010 and has great customer service."
Answer Relevancy Score: Low (0.2) - Doesn't answer the question!

Question: "How do I reset my password?"
Answer: "Click 'Forgot Password' on the login page and follow the email instructions."
Answer Relevancy Score: High (0.95) - Directly answers the question!
```

**When to use**: To ensure your AI stays on topic and provides useful answers.

---

### 3. **Context Precision** 🔍
**Question**: Are the retrieved documents actually relevant to the question?

**Simple Explanation**: Measures how good your retrieval system is at finding the right documents.

**Example:**
```python
Question: "What are the health benefits of green tea?"

Retrieved Documents:
1. "Green tea contains antioxidants that may reduce cancer risk..." ✅ Relevant
2. "Coffee is a popular morning beverage..." ❌ Not relevant
3. "Green tea may improve brain function..." ✅ Relevant

Context Precision Score: 0.67 (2 out of 3 documents are relevant)
```

**When to use**: To improve your document retrieval system.

---

### 4. **Context Recall** 📚
**Question**: Did we retrieve all the relevant information needed to answer the question?

**Simple Explanation**: Checks if important information was missed during retrieval.

**Example:**
```python
Question: "What are the main causes of climate change?"
Ground Truth: "The main causes are fossil fuel burning, deforestation, and industrial processes."

Retrieved Context: ["Fossil fuel burning releases CO2..."]
Context Recall Score: Low (0.33) - Missing deforestation and industrial processes!

Retrieved Context: ["Fossil fuels...", "Deforestation...", "Industrial processes..."]
Context Recall Score: High (1.0) - All key topics covered!
```

**When to use**: To ensure your retrieval isn't missing important information.

---

### 5. **Context Relevancy** 🎪
**Question**: How much of the retrieved context is actually useful?

**Simple Explanation**: Similar to Context Precision, but focuses on the proportion of useful content.

**Example:**
```python
Question: "What is Python?"

Retrieved Context: 
"Python is a programming language. It was created by Guido van Rossum. 
The weather today is sunny. Python is used for web development, data science..."

Context Relevancy Score: 0.75 (weather sentence is irrelevant noise)
```

**When to use**: To reduce noise in your retrieved documents.

---

### 6. **Answer Correctness** ✅
**Question**: Is the answer factually correct compared to the ground truth?

**Simple Explanation**: Compares the AI's answer to the ideal answer (if you have one).

**Example:**
```python
Question: "What is 2+2?"
Ground Truth: "4"
Answer: "4"
Answer Correctness Score: 1.0 (Perfect!)

Question: "What is 2+2?"
Ground Truth: "4"
Answer: "5"
Answer Correctness Score: 0.0 (Wrong!)
```

**When to use**: When you have ground truth answers to compare against.

---

## Quick Start: Basic RAG Evaluation

Here's a simple example to get started:

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

# Your RAG system's outputs
data = {
    "question": ["What are the benefits of exercise?"],
    "answer": ["Exercise improves cardiovascular health and mental well-being."],
    "contexts": [["Exercise improves heart health and reduces stress.", 
                  "Regular physical activity boosts mood."]],
    "ground_truth": ["Exercise improves cardiovascular health, mental well-being, and physical strength."]
}

# Convert to dataset
dataset = Dataset.from_dict(data)

# Evaluate
results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
)

print(results)
```

### Output:
```
{
    'faithfulness': 1.0,          # Answer is faithful to context
    'answer_relevancy': 0.95,     # Answer is highly relevant
    'context_precision': 1.0,     # Retrieved docs are relevant
    'context_recall': 0.85        # Most ground truth info was retrieved
}
```

## Complete Working Example

Here's a full example evaluating a RAG system:

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)
from datasets import Dataset

# Simulate RAG system outputs
data = {
    "question": [
        "What is machine learning?",
        "How does photosynthesis work?",
        "What are the primary colors?"
    ],
    "answer": [
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "Photosynthesis is the process where plants convert sunlight into energy using chlorophyll.",
        "The primary colors are red, blue, and yellow."
    ],
    "contexts": [
        ["Machine learning is a branch of artificial intelligence that focuses on learning from data."],
        ["Photosynthesis converts light energy into chemical energy in plants.", 
         "Chlorophyll is the green pigment that captures sunlight."],
        ["Primary colors are red, blue, and yellow.", 
         "They cannot be created by mixing other colors."]
    ],
    "ground_truth": [
        "Machine learning is a subset of artificial intelligence that allows systems to learn and improve from experience without being explicitly programmed.",
        "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of sugar.",
        "The primary colors are red, blue, and yellow."
    ]
}

# Create dataset
dataset = Dataset.from_dict(data)

# Evaluate with all metrics
results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness
    ]
)

# Print results
print("Evaluation Results:")
print(f"Faithfulness: {results['faithfulness']:.2f}")
print(f"Answer Relevancy: {results['answer_relevancy']:.2f}")
print(f"Context Precision: {results['context_precision']:.2f}")
print(f"Context Recall: {results['context_recall']:.2f}")
print(f"Answer Correctness: {results['answer_correctness']:.2f}")
```

## Using Ground Truth for Answer Correctness

**Ground truth** is the ideal answer you expect. It's optional but highly recommended for objective evaluation.

### With Ground Truth:
```python
data = {
    "question": ["What is the speed of light?"],
    "answer": ["The speed of light is approximately 300,000 km/s."],
    "contexts": [["Light travels at 299,792 km/s in a vacuum."]],
    "ground_truth": ["The speed of light is 299,792 kilometers per second."]
}

# Evaluate answer correctness
results = evaluate(dataset, metrics=[answer_correctness])
# Score: ~0.95 (close but not exact)
```

### Without Ground Truth:
```python
data = {
    "question": ["What is the speed of light?"],
    "answer": ["The speed of light is approximately 300,000 km/s."],
    "contexts": [["Light travels at 299,792 km/s in a vacuum."]]
    # No ground_truth provided
}

# Can still evaluate faithfulness and relevancy
results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
```

## Synthetic Test Generation

Ragas can automatically generate test questions from your documents:

```python
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Your documents
documents = [
    "Python is a high-level programming language known for its simplicity.",
    "Machine learning is a subset of AI that learns from data.",
    "Neural networks are inspired by the human brain structure."
]

# Setup generator
generator = TestsetGenerator.with_openai()

# Generate test set
testset = generator.generate_with_langchain_docs(
    documents,
    test_size=10,  # Generate 10 questions
    distributions={
        simple: 0.5,      # 50% simple questions
        reasoning: 0.25,  # 25% reasoning questions
        multi_context: 0.25  # 25% multi-context questions
    }
)

# Use generated testset
print(testset.to_pandas())
```

## Real-World Scenarios

### Scenario 1: Customer Support Chatbot
```python
# Evaluate if chatbot gives accurate, relevant answers
data = {
    "question": ["How do I track my order?"],
    "answer": [chatbot.get_answer("How do I track my order?")],
    "contexts": [retrieved_docs],
    "ground_truth": ["Log into your account and click 'Order History' to track your order."]
}

results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, answer_correctness])
```

### Scenario 2: Documentation Q&A
```python
# Evaluate technical documentation RAG system
data = {
    "question": ["How do I install the library?"],
    "answer": [doc_qa_system.query("How do I install the library?")],
    "contexts": [retrieved_doc_sections],
    "ground_truth": ["Run 'pip install library-name' in your terminal."]
}

results = evaluate(dataset, metrics=[context_precision, context_recall, faithfulness])
```

### Scenario 3: Research Assistant
```python
# Evaluate research paper summarization
data = {
    "question": ["What are the key findings of this paper?"],
    "answer": [research_assistant.summarize(paper)],
    "contexts": [paper_sections],
    "ground_truth": ["The paper found that X leads to Y under conditions Z."]
}

results = evaluate(dataset, metrics=[answer_correctness, faithfulness])
```

## When to Use Which Metrics

| Metric | Use When | Don't Use When |
|--------|----------|----------------|
| **Faithfulness** | Always (prevents hallucinations) | - |
| **Answer Relevancy** | Always (ensures on-topic answers) | - |
| **Context Precision** | Improving retrieval quality | Retrieval is already perfect |
| **Context Recall** | You have ground truth | No ground truth available |
| **Answer Correctness** | You have ground truth answers | No ground truth available |

## Metric Combinations for Different Goals

### Goal: Prevent Hallucinations
```python
metrics = [faithfulness, context_precision]
```

### Goal: Improve Answer Quality
```python
metrics = [answer_relevancy, answer_correctness, faithfulness]
```

### Goal: Optimize Retrieval
```python
metrics = [context_precision, context_recall, context_relevancy]
```

### Goal: Comprehensive Evaluation
```python
metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
]
```

## Integration with LangChain

```python
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Setup RAG system
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(your_documents, embeddings)
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=vectorstore.as_retriever()
)

# Evaluate
questions = ["What is...?", "How does...?"]
data = {
    "question": questions,
    "answer": [qa_chain.run(q) for q in questions],
    "contexts": [[doc.page_content for doc in vectorstore.similarity_search(q)] for q in questions],
    "ground_truth": your_ground_truths
}

dataset = Dataset.from_dict(data)
results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
```

## Best Practices

### 1. Start Simple
Begin with just `faithfulness` and `answer_relevancy`:
```python
results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
```

### 2. Add Ground Truth When Possible
Ground truth enables more metrics:
```python
# Without ground truth: 3 metrics
metrics = [faithfulness, answer_relevancy, context_relevancy]

# With ground truth: 5 metrics
metrics = [faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness]
```

### 3. Use Synthetic Data for Testing
Generate test cases automatically:
```python
testset = generator.generate_with_langchain_docs(documents, test_size=50)
```

### 4. Monitor Scores Over Time
Track improvements:
```python
# Version 1
results_v1 = evaluate(dataset_v1, metrics)

# Version 2 (after improvements)
results_v2 = evaluate(dataset_v2, metrics)

# Compare
print(f"Faithfulness improved: {results_v2['faithfulness'] - results_v1['faithfulness']}")
```

### 5. Set Thresholds for Production
Define minimum acceptable scores:
```python
THRESHOLDS = {
    'faithfulness': 0.8,
    'answer_relevancy': 0.75,
    'context_precision': 0.7
}

# Check if system meets requirements
for metric, threshold in THRESHOLDS.items():
    if results[metric] < threshold:
        print(f"WARNING: {metric} below threshold!")
```

## Score Interpretation

| Score Range | Meaning | Action |
|-------------|---------|--------|
| 0.9 - 1.0 | Excellent | System is performing very well |
| 0.7 - 0.9 | Good | Minor improvements possible |
| 0.5 - 0.7 | Acceptable | Needs attention |
| 0.3 - 0.5 | Poor | Significant issues |
| 0.0 - 0.3 | Very Poor | Major problems, requires immediate action |

## Troubleshooting

### Issue: Low Faithfulness Scores
**Cause**: AI is hallucinating or adding information not in context.
**Solution**: 
- Improve prompts to emphasize using only provided context
- Reduce temperature in LLM generation
- Add explicit instructions: "Only use information from the provided documents"

### Issue: Low Context Precision
**Cause**: Retrieval system is returning irrelevant documents.
**Solution**:
- Improve embedding model
- Adjust retrieval parameters (top_k, similarity threshold)
- Enhance document chunking strategy

### Issue: Low Answer Relevancy
**Cause**: AI is going off-topic or providing generic answers.
**Solution**:
- Improve prompts to focus on the question
- Use more specific retrieval queries
- Fine-tune the LLM for your domain

### Issue: Low Context Recall
**Cause**: Important information is being missed during retrieval.
**Solution**:
- Increase number of retrieved documents (top_k)
- Improve document indexing
- Use hybrid search (keyword + semantic)

## Installation and Setup

```bash
# Install Ragas
pip install ragas

# Install dependencies
pip install langchain langchain-openai datasets

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key"
```

## Running Evaluation

```python
# Basic evaluation
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

results = evaluate(your_dataset, metrics=[faithfulness, answer_relevancy])
print(results)

# Save results
results_df = results.to_pandas()
results_df.to_csv('evaluation_results.csv')
```

## Additional Resources

- [Ragas Documentation](https://docs.ragas.io/)
- [Ragas GitHub](https://github.com/explodinggradients/ragas)
- [RAG Evaluation Guide](https://docs.ragas.io/en/latest/concepts/metrics/)
- [LangChain Integration](https://docs.ragas.io/en/latest/howtos/integrations/langchain.html)

## Summary

**Ragas makes RAG evaluation simple:**

1. **Choose metrics** based on what you want to measure
2. **Prepare data** with questions, answers, contexts, and optionally ground truth
3. **Run evaluation** with `evaluate(dataset, metrics)`
4. **Analyze results** and improve your system
5. **Iterate** until scores meet your requirements

**Key Insight**: Different metrics measure different aspects of your RAG system. Use multiple metrics for comprehensive evaluation!

| Metric | Measures | Needs Ground Truth? |
|--------|----------|---------------------|
| Faithfulness | No hallucinations | ❌ No |
| Answer Relevancy | On-topic answers | ❌ No |
| Context Precision | Retrieval quality | ✅ Yes |
| Context Recall | Retrieval completeness | ✅ Yes |
| Answer Correctness | Factual accuracy | ✅ Yes |

Start with faithfulness and answer relevancy, then add more metrics as needed!
