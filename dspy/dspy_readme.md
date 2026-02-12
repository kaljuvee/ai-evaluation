# DSPy: A Simple Guide to Programmatic LLM Optimization

## What is DSPy?

**DSPy** (Declarative Self-improving Python) is a framework that **automatically optimizes your prompts and LLM programs**. Instead of manually tweaking prompts, DSPy uses algorithms to find the best prompts for you.

### The Big Idea

**Traditional Approach** (Manual):
```
You: "Summarize this text"
LLM: [mediocre output]
You: "Summarize this text concisely and focus on key points"
LLM: [better output]
You: "Summarize this text in 2-3 sentences highlighting main ideas"
LLM: [even better output]
... repeat 50 times ...
```

**DSPy Approach** (Automatic):
```
You: Define what "good" looks like (a metric)
DSPy: Automatically tests 100+ prompt variations
DSPy: Finds the best prompt for you
You: Use the optimized prompt
```

### Why Use DSPy?

- ✅ **Automatic optimization**: No more manual prompt engineering
- ✅ **Data-driven**: Uses your data to find what works
- ✅ **Systematic**: Tests many variations scientifically
- ✅ **Reproducible**: Same process works across different tasks
- ✅ **Flexible**: Works with or without ground truth

## Key Concepts (In Simple Terms)

### 1. **Signatures** 📝
**What it is**: A description of what your LLM should do.

**Think of it as**: A function signature, but for AI.

```python
# Traditional function
def summarize(text: str) -> str:
    pass

# DSPy signature (similar idea!)
signature = "text -> summary"
```

**Example:**
```python
import dspy

# Simple signature
class Summarize(dspy.Signature):
    """Summarize the input text."""
    text = dspy.InputField()
    summary = dspy.OutputField()

# Use it
summarizer = dspy.Predict(Summarize)
result = summarizer(text="Long article here...")
print(result.summary)
```

### 2. **Modules** 🧩
**What it is**: A reusable component that does a specific task.

**Think of it as**: A building block for your AI application.

```python
# Create a module
class QuestionAnswerer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.generate_answer(question=question)

# Use it
qa = QuestionAnswerer()
answer = qa(question="What is Python?")
```

### 3. **Teleprompters** (Optimizers) 🎯
**What it is**: An algorithm that automatically improves your prompts.

**Think of it as**: An auto-tuner for your AI.

**Common Teleprompters:**
- **BootstrapFewShot**: Uses examples (needs ground truth)
- **MIPRO**: Uses metrics only (no ground truth needed)
- **BootstrapFewShotWithRandomSearch**: Combines examples + search

## Two Optimization Approaches

### Approach 1: With Ground Truth (Labeled Examples)

**When to use**: You have example inputs and their ideal outputs.

**How it works**: DSPy learns from your examples to create better prompts.

**Example:**
```python
import dspy
from dspy.teleprompt import BootstrapFewShot

# Step 1: Configure DSPy
lm = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=lm)

# Step 2: Define your task
class Summarize(dspy.Signature):
    """Summarize the text concisely."""
    text = dspy.InputField()
    summary = dspy.OutputField()

class Summarizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_summary = dspy.ChainOfThought(Summarize)
    
    def forward(self, text):
        return self.generate_summary(text=text)

# Step 3: Create training examples (ground truth)
trainset = [
    dspy.Example(
        text="Long article about climate change...",
        summary="Climate change is caused by greenhouse gases."
    ).with_inputs("text"),
    dspy.Example(
        text="Article about AI advancements...",
        summary="AI has advanced rapidly in recent years."
    ).with_inputs("text"),
    # ... more examples
]

# Step 4: Define success metric
def validate_summary(example, pred, trace=None):
    # Check if summary is concise (< 100 chars)
    return len(pred.summary) < 100

# Step 5: Optimize!
optimizer = BootstrapFewShot(metric=validate_summary)
optimized_summarizer = optimizer.compile(
    Summarizer(),
    trainset=trainset
)

# Step 6: Use optimized version
result = optimized_summarizer(text="New article...")
print(result.summary)
```

**What happened?**
1. DSPy looked at your examples
2. It created prompts that produce similar outputs
3. It tested different prompt variations
4. It kept the best-performing prompts

---

### Approach 2: Without Ground Truth (Metric-Only)

**When to use**: You don't have labeled examples, but you know what "good" looks like.

**How it works**: DSPy uses your metric to judge quality and optimize accordingly.

**Example:**
```python
import dspy
from dspy.teleprompt import MIPRO

# Step 1: Configure DSPy
lm = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=lm)

# Step 2: Define your task (same as before)
class Summarize(dspy.Signature):
    """Summarize the text concisely."""
    text = dspy.InputField()
    summary = dspy.OutputField()

class Summarizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_summary = dspy.ChainOfThought(Summarize)
    
    def forward(self, text):
        return self.generate_summary(text=text)

# Step 3: Create examples WITHOUT ground truth
trainset = [
    dspy.Example(text="Long article about climate change...").with_inputs("text"),
    dspy.Example(text="Article about AI advancements...").with_inputs("text"),
    dspy.Example(text="Story about space exploration...").with_inputs("text"),
    # ... more examples (no summaries needed!)
]

# Step 4: Define success metric (no ground truth needed)
def quality_metric(example, pred, trace=None):
    summary = pred.summary
    
    # Check multiple quality criteria
    is_concise = len(summary) < 100
    has_content = len(summary) > 20
    no_repetition = len(set(summary.split())) / len(summary.split()) > 0.7
    
    # Return score (0.0 to 1.0)
    score = (is_concise + has_content + no_repetition) / 3
    return score

# Step 5: Optimize with MIPRO!
optimizer = MIPRO(
    metric=quality_metric,
    num_candidates=10,  # Try 10 different prompt variations
    init_temperature=1.0
)

optimized_summarizer = optimizer.compile(
    Summarizer(),
    trainset=trainset,
    num_trials=20  # Run 20 optimization trials
)

# Step 6: Use optimized version
result = optimized_summarizer(text="New article...")
print(result.summary)
```

**What happened?**
1. DSPy generated different prompt variations
2. It tested each variation on your examples
3. It scored each using your metric
4. It kept the best-performing prompts

---

## Side-by-Side Comparison

| Aspect | With Ground Truth | Without Ground Truth |
|--------|-------------------|----------------------|
| **Optimizer** | `BootstrapFewShot` | `MIPRO` |
| **Training Data** | Examples with ideal outputs | Examples with just inputs |
| **Metric** | Compares to ground truth | Evaluates quality criteria |
| **Best For** | Q&A, classification, translation | Summarization, creative tasks |
| **Accuracy** | Higher (has target to aim for) | Good (depends on metric quality) |
| **Setup Effort** | More (need labeled data) | Less (just define quality) |

### Example Comparison

**Task**: Summarize articles

**With Ground Truth:**
```python
trainset = [
    dspy.Example(
        text="Article...",
        summary="Ideal summary"  # ← You provide this
    ).with_inputs("text")
]

def metric(example, pred, trace=None):
    # Compare to ground truth
    return example.summary == pred.summary
```

**Without Ground Truth:**
```python
trainset = [
    dspy.Example(
        text="Article..."  # ← No summary needed!
    ).with_inputs("text")
]

def metric(example, pred, trace=None):
    # Evaluate quality without ground truth
    return len(pred.summary) < 100 and len(pred.summary) > 20
```

## Complete Working Examples

### Example 1: Question Answering (With Ground Truth)

```python
import dspy
from dspy.teleprompt import BootstrapFewShot

# Setup
lm = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=lm)

# Define task
class QA(dspy.Signature):
    """Answer the question based on context."""
    context = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.OutputField()

class QuestionAnswerer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(QA)
    
    def forward(self, context, question):
        return self.generate_answer(context=context, question=question)

# Training data with ground truth
trainset = [
    dspy.Example(
        context="Paris is the capital of France.",
        question="What is the capital of France?",
        answer="Paris"
    ).with_inputs("context", "question"),
    dspy.Example(
        context="Python was created by Guido van Rossum.",
        question="Who created Python?",
        answer="Guido van Rossum"
    ).with_inputs("context", "question"),
]

# Metric: Check if answer matches ground truth
def validate_answer(example, pred, trace=None):
    return example.answer.lower() in pred.answer.lower()

# Optimize
optimizer = BootstrapFewShot(metric=validate_answer, max_bootstrapped_demos=2)
optimized_qa = optimizer.compile(QuestionAnswerer(), trainset=trainset)

# Use
result = optimized_qa(
    context="The Eiffel Tower is in Paris.",
    question="Where is the Eiffel Tower?"
)
print(result.answer)  # "Paris" or "The Eiffel Tower is in Paris"
```

---

### Example 2: Text Classification (Without Ground Truth)

```python
import dspy
from dspy.teleprompt import MIPRO

# Setup
lm = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=lm)

# Define task
class Classify(dspy.Signature):
    """Classify the sentiment of the text."""
    text = dspy.InputField()
    sentiment = dspy.OutputField(desc="positive, negative, or neutral")

class SentimentClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(Classify)
    
    def forward(self, text):
        return self.classify(text=text)

# Training data WITHOUT ground truth
trainset = [
    dspy.Example(text="I love this product!").with_inputs("text"),
    dspy.Example(text="This is terrible.").with_inputs("text"),
    dspy.Example(text="It's okay, nothing special.").with_inputs("text"),
]

# Metric: Check if sentiment is valid (no ground truth needed)
def validate_sentiment(example, pred, trace=None):
    valid_sentiments = ["positive", "negative", "neutral"]
    sentiment = pred.sentiment.lower()
    
    # Check if output is valid
    is_valid = any(s in sentiment for s in valid_sentiments)
    
    # Check if output is concise
    is_concise = len(sentiment.split()) <= 3
    
    return is_valid and is_concise

# Optimize
optimizer = MIPRO(metric=validate_sentiment, num_candidates=5)
optimized_classifier = optimizer.compile(
    SentimentClassifier(),
    trainset=trainset,
    num_trials=10
)

# Use
result = optimized_classifier(text="This is amazing!")
print(result.sentiment)  # "positive"
```

---

## When to Use DSPy

### ✅ Use DSPy When:
- You're manually tweaking prompts repeatedly
- You have multiple similar tasks (e.g., classify 10 different categories)
- You want consistent, reproducible results
- You have training data (with or without labels)
- You need to optimize for specific metrics

### ❌ Don't Use DSPy When:
- You have a simple, one-off task
- Your prompt already works perfectly
- You don't have any training examples
- You need immediate results (optimization takes time)

## Common Patterns

### Pattern 1: Multi-Step Reasoning
```python
class ResearchAssistant(dspy.Module):
    def __init__(self):
        super().__init__()
        self.find_info = dspy.ChainOfThought("question -> search_query")
        self.synthesize = dspy.ChainOfThought("search_results -> answer")
    
    def forward(self, question):
        # Step 1: Generate search query
        query = self.find_info(question=question)
        
        # Step 2: Get search results (your retrieval logic)
        results = search_engine.search(query.search_query)
        
        # Step 3: Synthesize answer
        answer = self.synthesize(search_results=results)
        return answer
```

### Pattern 2: Iterative Refinement
```python
class IterativeWriter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.draft = dspy.ChainOfThought("topic -> draft")
        self.refine = dspy.ChainOfThought("draft, feedback -> improved_draft")
    
    def forward(self, topic, num_iterations=3):
        # Initial draft
        result = self.draft(topic=topic)
        
        # Iteratively refine
        for i in range(num_iterations):
            feedback = self.get_feedback(result.draft)
            result = self.refine(draft=result.draft, feedback=feedback)
        
        return result
```

### Pattern 3: Ensemble (Multiple Approaches)
```python
class EnsembleSummarizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.concise = dspy.ChainOfThought("text -> concise_summary")
        self.detailed = dspy.ChainOfThought("text -> detailed_summary")
        self.combine = dspy.ChainOfThought("summaries -> final_summary")
    
    def forward(self, text):
        # Get multiple summaries
        concise = self.concise(text=text)
        detailed = self.detailed(text=text)
        
        # Combine them
        final = self.combine(summaries=f"{concise.concise_summary}\n{detailed.detailed_summary}")
        return final
```

## Troubleshooting

### Issue: Optimization Takes Too Long
**Solution**: Reduce the number of trials and candidates
```python
optimizer = MIPRO(
    num_candidates=5,  # Instead of 20
    num_trials=10      # Instead of 50
)
```

### Issue: Optimized Model Performs Worse
**Solution**: Your metric might be too simple or wrong
```python
# ❌ Too simple
def metric(example, pred, trace=None):
    return len(pred.summary) < 100

# ✅ Better
def metric(example, pred, trace=None):
    is_concise = len(pred.summary) < 100
    has_content = len(pred.summary) > 20
    is_relevant = any(word in pred.summary for word in example.text.split()[:10])
    return (is_concise + has_content + is_relevant) / 3
```

### Issue: "No module named 'dspy'"
**Solution**: Install DSPy
```bash
pip install dspy-ai
```

### Issue: Optimization Fails with Error
**Solution**: Check your training data format
```python
# ✅ Correct format
example = dspy.Example(
    input_field="value",
    output_field="value"
).with_inputs("input_field")

# ❌ Wrong format
example = {"input_field": "value"}  # Not a dspy.Example
```

## Installation and Setup

```bash
# Install DSPy
pip install dspy-ai

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key"
```

## Running Optimization

```python
import dspy

# Configure
lm = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=lm)

# Create your module
module = YourModule()

# Optimize
optimizer = BootstrapFewShot(metric=your_metric)  # or MIPRO
optimized_module = optimizer.compile(module, trainset=your_data)

# Save optimized module
optimized_module.save("optimized_model.json")

# Load later
loaded_module = YourModule()
loaded_module.load("optimized_model.json")
```

## Additional Resources

- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [DSPy Paper](https://arxiv.org/abs/2310.03714)
- [MIPRO Paper](https://arxiv.org/abs/2406.11695)

## Summary

**DSPy makes prompt optimization automatic:**

1. **Define your task** with signatures and modules
2. **Create training data** (with or without ground truth)
3. **Define a metric** (what "good" looks like)
4. **Choose an optimizer**:
   - `BootstrapFewShot` if you have ground truth
   - `MIPRO` if you don't have ground truth
5. **Run optimization** and get improved prompts
6. **Use the optimized module** in production

### Key Decision Tree

```
Do you have labeled examples (ground truth)?
├─ YES → Use BootstrapFewShot
│         - Higher accuracy
│         - Learns from examples
│         - Best for Q&A, classification
│
└─ NO → Use MIPRO
          - Works without labels
          - Uses quality metrics
          - Best for summarization, generation
```

**The Magic**: DSPy automatically finds better prompts than you could manually, saving hours of trial and error!
