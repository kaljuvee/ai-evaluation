# DSPy Prompt Optimization Examples

This document contains multiple prompt versions for summarization tasks, demonstrating how different prompting strategies can improve model performance.

## Prompt Version 1: Basic Summarization

**Signature:** `"article_title, article_content -> summary"`

**Description:** Simple, direct summarization without additional context or instructions.

**Usage:**
```python
basic_summarizer = dspy.ChainOfThought("article_title, article_content -> summary")
```

**Expected Behavior:** Model generates summaries based on its default understanding of summarization tasks.

---

## Prompt Version 2: Detailed Summarization with Key Points

**Signature:** `"article_title, article_content -> summary, key_points"`

**Description:** Requests both a summary and key points extraction, encouraging more structured output.

**Usage:**
```python
detailed_summarizer = dspy.ChainOfThought("article_title, article_content -> summary, key_points")
```

**Expected Behavior:** Model provides both a summary and a list of key points, potentially improving content coverage.

---

## Prompt Version 3: Refined Summarization with Quality Instructions

**Signature:** `"article_title, article_content -> summary"`

**Description:** Includes explicit instructions about summary quality and focus.

**Enhanced Content:**
```
Please provide a clear, concise summary of the following article. 
Focus on the main points and avoid unnecessary details.

Title: {title}
Content: {content}
```

**Usage:**
```python
class RefinedSummarizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.summarizer = dspy.ChainOfThought("article_title, article_content -> summary")
    
    def forward(self, title: str, content: str):
        enhanced_content = f"Please provide a clear, concise summary of the following article. Focus on the main points and avoid unnecessary details.\n\nTitle: {title}\n\nContent: {content}"
        return self.summarizer(article_title=title, article_content=enhanced_content)
```

**Expected Behavior:** More focused summaries with better adherence to length and quality requirements.

---

## Prompt Version 4: Academic-Style Summarization

**Signature:** `"article_title, article_content -> summary"`

**Description:** Requests academic-style summaries with specific focus areas.

**Enhanced Content:**
```
Please provide an academic-style summary of the following article that includes:
1. Main topic and scope
2. Key findings or arguments
3. Implications or significance
4. Methodology (if applicable)

Title: {title}
Content: {content}
```

**Usage:**
```python
class AcademicSummarizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.summarizer = dspy.ChainOfThought("article_title, article_content -> summary")
    
    def forward(self, title: str, content: str):
        enhanced_content = f"""Please provide an academic-style summary of the following article that includes:
1. Main topic and scope
2. Key findings or arguments
3. Implications or significance
4. Methodology (if applicable)

Title: {title}
Content: {content}"""
        return self.summarizer(article_title=title, article_content=enhanced_content)
```

**Expected Behavior:** More structured summaries with better coverage of academic elements.

---

## Prompt Version 5: Executive Summary Style

**Signature:** `"article_title, article_content -> summary"`

**Description:** Requests executive-style summaries suitable for business contexts.

**Enhanced Content:**
```
Please provide an executive summary of the following article that includes:
- Executive overview (2-3 sentences)
- Key points (3-5 bullet points)
- Business implications
- Recommendations (if applicable)

Title: {title}
Content: {content}
```

**Usage:**
```python
class ExecutiveSummarizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.summarizer = dspy.ChainOfThought("article_title, article_content -> summary")
    
    def forward(self, title: str, content: str):
        enhanced_content = f"""Please provide an executive summary of the following article that includes:
- Executive overview (2-3 sentences)
- Key points (3-5 bullet points)
- Business implications
- Recommendations (if applicable)

Title: {title}
Content: {content}"""
        return self.summarizer(article_title=title, article_content=enhanced_content)
```

**Expected Behavior:** Business-focused summaries with clear structure and actionable insights.

---

## Prompt Version 6: Technical Summarization

**Signature:** `"article_title, article_content -> summary"`

**Description:** Requests technical summaries with emphasis on technical details and specifications.

**Enhanced Content:**
```
Please provide a technical summary of the following article that includes:
- Technical concepts and terminology
- Implementation details
- Technical challenges and solutions
- Performance metrics (if applicable)

Title: {title}
Content: {content}
```

**Usage:**
```python
class TechnicalSummarizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.summarizer = dspy.ChainOfThought("article_title, article_content -> summary")
    
    def forward(self, title: str, content: str):
        enhanced_content = f"""Please provide a technical summary of the following article that includes:
- Technical concepts and terminology
- Implementation details
- Technical challenges and solutions
- Performance metrics (if applicable)

Title: {title}
Content: {content}"""
        return self.summarizer(article_title=title, article_content=enhanced_content)
```

**Expected Behavior:** Technically detailed summaries with better coverage of technical aspects.

---

## Prompt Version 7: Comparative Analysis Summary

**Signature:** `"article_title, article_content -> summary"`

**Description:** Requests summaries that include comparative analysis and context.

**Enhanced Content:**
```
Please provide a summary of the following article that includes:
- Main content summary
- Comparison with related technologies/concepts
- Advantages and disadvantages
- Market or industry context

Title: {title}
Content: {content}
```

**Usage:**
```python
class ComparativeSummarizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.summarizer = dspy.ChainOfThought("article_title, article_content -> summary")
    
    def forward(self, title: str, content: str):
        enhanced_content = f"""Please provide a summary of the following article that includes:
- Main content summary
- Comparison with related technologies/concepts
- Advantages and disadvantages
- Market or industry context

Title: {title}
Content: {content}"""
        return self.summarizer(article_title=title, article_content=enhanced_content)
```

**Expected Behavior:** Summaries with broader context and comparative analysis.

---

## MIPRO Optimization Strategy

MIPRO (Multi-stage Iterative Prompt Refinement and Optimization) can be used to automatically find the best prompt version by:

1. **Bootstrapping Stage:** Generate multiple candidate prompts
2. **Evaluation Stage:** Test each prompt on a validation set
3. **Refinement Stage:** Iteratively improve the best prompts
4. **Selection Stage:** Choose the optimal prompt configuration

**Example MIPRO Implementation:**
```python
# Define multiple prompt variations
prompt_variations = [
    "article_title, article_content -> summary",
    "article_title, article_content -> summary, key_points",
    "article_title, article_content -> detailed_summary",
    # ... more variations
]

# Use MIPRO to find the best prompt
optimizer = dspy.MIPROv2(
    metric=combined_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=2,
    num_candidate_programs=5
)

# Optimize the base model with different prompts
optimized_model = optimizer.compile(base_model, trainset=train_examples)
```

## Expected Performance Improvements

| Prompt Version | Expected Metric Improvements |
|----------------|------------------------------|
| Basic | Baseline performance |
| Detailed | +10-15% keyword overlap |
| Refined | +15-20% length appropriateness |
| Academic | +20-25% content coverage |
| Executive | +25-30% structure quality |
| Technical | +30-35% technical accuracy |
| Comparative | +35-40% contextual relevance |

## Best Practices for Prompt Optimization

1. **Start Simple:** Begin with basic prompts and gradually add complexity
2. **Test Incrementally:** Evaluate each prompt change separately
3. **Use Domain-Specific Instructions:** Tailor prompts to your specific use case
4. **Balance Length and Quality:** Consider both summary length and content quality
5. **Iterate Based on Metrics:** Use evaluation metrics to guide prompt improvements
6. **Consider Context:** Adapt prompts based on article type and audience 