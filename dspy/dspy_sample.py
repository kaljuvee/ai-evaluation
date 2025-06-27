"""
DSPy Sample: Synthetic Data Generation, Summarization Tasks, and Iterative Optimization

This sample demonstrates:
1. Synthetic data generation for summarization tasks
2. Multiple DSPy module versions for comparison
3. Iterative optimization using different optimizers
4. Evaluation metrics and comparison
"""

import dspy
import random
import json
import os
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure DSPy with a language model
# Note: You'll need to set your API key as an environment variable
# export OPENAI_API_KEY="your-api-key-here"
# Or use a different model provider

def setup_dspy():
    """Setup DSPy with a language model"""
    try:
        # Check if OpenAI API key is available
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            print(f"✅ Found OpenAI API key: {api_key[:10]}...")
            lm = dspy.LM("openai/gpt-4o-mini")
            dspy.configure(lm=lm)
            print("✅ Configured DSPy with OpenAI GPT-4o-mini")
        else:
            print("⚠️  No OpenAI API key found in environment variables")
            print("Please set OPENAI_API_KEY in your .env file")
            print("Using mock configuration for demonstration...")
            dspy.configure(lm=None)
    except Exception as e:
        print(f"⚠️  Could not configure OpenAI: {e}")
        print("Please set OPENAI_API_KEY environment variable")
        print("Using mock configuration for demonstration...")
        # For demonstration, we'll use a mock configuration
        dspy.configure(lm=None)

@dataclass
class Article:
    """Represents an article with title, content, and expected summary"""
    title: str
    content: str
    expected_summary: str
    category: str
    difficulty: str  # easy, medium, hard

class SyntheticDataGenerator:
    """Generates synthetic articles and summaries for training and evaluation"""
    
    def __init__(self):
        self.categories = ["technology", "science", "business", "health", "environment"]
        self.difficulties = ["easy", "medium", "hard"]
        
        # Template articles for different categories
        self.article_templates = {
            "technology": [
                {
                    "title": "The Rise of Artificial Intelligence in Modern Computing",
                    "content": "Artificial Intelligence (AI) has transformed the landscape of modern computing in unprecedented ways. From machine learning algorithms that power recommendation systems to natural language processing models that enable human-like text generation, AI technologies are becoming increasingly sophisticated. The development of large language models like GPT-4 and Claude has demonstrated the potential for AI to understand and generate human language with remarkable accuracy. These advancements have applications across various industries, including healthcare, finance, education, and entertainment. However, the rapid pace of AI development also raises important questions about ethics, privacy, and the future of work. As AI systems become more capable, society must grapple with issues of bias, transparency, and accountability in AI decision-making processes.",
                    "expected_summary": "AI has revolutionized computing through machine learning and natural language processing, with applications across industries but raising ethical concerns about bias and accountability."
                },
                {
                    "title": "Blockchain Technology: Beyond Cryptocurrency",
                    "content": "While blockchain technology is most commonly associated with cryptocurrencies like Bitcoin and Ethereum, its applications extend far beyond digital currencies. Blockchain's core features of decentralization, immutability, and transparency make it suitable for various use cases including supply chain management, voting systems, digital identity verification, and smart contracts. In supply chain management, blockchain can provide end-to-end traceability of products from manufacturer to consumer, helping to prevent fraud and ensure product authenticity. Smart contracts, which are self-executing contracts with the terms directly written into code, can automate complex business processes and reduce the need for intermediaries. Despite its potential, blockchain technology faces challenges such as scalability issues, energy consumption concerns, and regulatory uncertainty.",
                    "expected_summary": "Blockchain technology extends beyond cryptocurrency to supply chain management, voting systems, and smart contracts, though it faces scalability and regulatory challenges."
                }
            ],
            "science": [
                {
                    "title": "Climate Change: The Impact on Global Ecosystems",
                    "content": "Climate change represents one of the most significant challenges facing our planet today. Rising global temperatures, caused primarily by human activities such as burning fossil fuels and deforestation, are having profound effects on ecosystems worldwide. Scientists have observed shifts in species distributions, changes in migration patterns, and alterations in the timing of seasonal events such as flowering and breeding. Ocean acidification, resulting from increased carbon dioxide absorption, threatens marine life including coral reefs and shellfish. The melting of polar ice caps and glaciers contributes to rising sea levels, which in turn affects coastal communities and habitats. Research indicates that many species may not be able to adapt quickly enough to these rapid environmental changes, potentially leading to increased extinction rates.",
                    "expected_summary": "Climate change affects global ecosystems through temperature shifts, ocean acidification, and habitat loss, threatening species adaptation and survival."
                }
            ],
            "business": [
                {
                    "title": "The Future of Remote Work in Post-Pandemic Era",
                    "content": "The COVID-19 pandemic accelerated the adoption of remote work practices across industries worldwide. What began as a temporary necessity has evolved into a fundamental shift in how organizations approach work arrangements. Companies have discovered that remote work can lead to increased productivity, reduced overhead costs, and improved employee satisfaction. However, this transition has also revealed challenges including difficulties in maintaining team cohesion, managing work-life boundaries, and ensuring equitable access to opportunities for all employees. Many organizations are now adopting hybrid work models that combine the benefits of remote work with the advantages of in-person collaboration. The long-term implications of this shift include potential changes in urban planning, real estate markets, and the structure of work itself.",
                    "expected_summary": "Remote work adoption accelerated by COVID-19 offers productivity benefits but challenges team cohesion, leading to hybrid models and broader societal impacts."
                }
            ]
        }
    
    def generate_synthetic_articles(self, num_articles: int = 20) -> List[Article]:
        """Generate synthetic articles for training and evaluation"""
        articles = []
        
        for i in range(num_articles):
            category = random.choice(self.categories)
            difficulty = random.choice(self.difficulties)
            
            # Use template articles or generate variations
            if category in self.article_templates:
                template = random.choice(self.article_templates[category])
                article = Article(
                    title=template["title"],
                    content=template["content"],
                    expected_summary=template["expected_summary"],
                    category=category,
                    difficulty=difficulty
                )
            else:
                # Generate a simple synthetic article
                article = self._generate_simple_article(category, difficulty)
            
            articles.append(article)
        
        return articles
    
    def _generate_simple_article(self, category: str, difficulty: str) -> Article:
        """Generate a simple synthetic article when templates aren't available"""
        titles = {
            "health": "Advances in Medical Technology",
            "environment": "Sustainable Energy Solutions",
            "technology": "Emerging Computing Paradigms"
        }
        
        title = titles.get(category, f"Important Developments in {category.title()}")
        content = f"This is a {difficulty} article about {category}. It discusses various aspects and implications of current developments in this field."
        summary = f"Overview of {category} developments and their significance."
        
        return Article(title=title, content=content, expected_summary=summary, category=category, difficulty=difficulty)

# Define DSPy signatures for summarization
class SummarizationSignature(dspy.Signature):
    """Signature for article summarization"""
    article_title = dspy.InputField(desc="The title of the article")
    article_content = dspy.InputField(desc="The full content of the article to summarize")
    summary = dspy.OutputField(desc="A concise summary of the article's main points")

class DetailedSummarizationSignature(dspy.Signature):
    """Signature for detailed article summarization with key points"""
    article_title = dspy.InputField(desc="The title of the article")
    article_content = dspy.InputField(desc="The full content of the article to summarize")
    summary = dspy.OutputField(desc="A detailed summary with key points and implications")
    key_points = dspy.OutputField(desc="List of 3-5 key points from the article")

# Define different DSPy modules for summarization
class BasicSummarizer(dspy.Module):
    """Basic summarization module"""
    
    def __init__(self):
        super().__init__()
        self.summarizer = dspy.ChainOfThought("article_title, article_content -> summary")
    
    def forward(self, title: str, content: str):
        return self.summarizer(article_title=title, article_content=content)

class DetailedSummarizer(dspy.Module):
    """Detailed summarization module with key points"""
    
    def __init__(self):
        super().__init__()
        self.summarizer = dspy.ChainOfThought("article_title, article_content -> summary, key_points")
    
    def forward(self, title: str, content: str):
        return self.summarizer(article_title=title, article_content=content)

class RefinedSummarizer(dspy.Module):
    """Refined summarization with better prompting"""
    
    def __init__(self):
        super().__init__()
        self.summarizer = dspy.ChainOfThought("article_title, article_content -> summary")
    
    def forward(self, title: str, content: str):
        # Add context about summarization quality
        enhanced_content = f"Please provide a clear, concise summary of the following article. Focus on the main points and avoid unnecessary details.\n\nTitle: {title}\n\nContent: {content}"
        return self.summarizer(article_title=title, article_content=enhanced_content)

# Evaluation metrics
def exact_match_metric(gold, pred, trace=None):
    """Simple exact match metric"""
    return gold.expected_summary.lower().strip() == pred.summary.lower().strip()

def length_metric(gold, pred, trace=None):
    """Check if summary length is appropriate (not too short, not too long)"""
    gold_length = len(gold.expected_summary.split())
    pred_length = len(pred.summary.split())
    
    # Summary should be within 50% of expected length
    min_length = gold_length * 0.5
    max_length = gold_length * 1.5
    
    return min_length <= pred_length <= max_length

def keyword_overlap_metric(gold, pred, trace=None):
    """Check for keyword overlap between expected and predicted summaries"""
    gold_words = set(gold.expected_summary.lower().split())
    pred_words = set(pred.summary.lower().split())
    
    if not gold_words:
        return 0.0
    
    overlap = len(gold_words.intersection(pred_words))
    return overlap / len(gold_words)

def combined_metric(gold, pred, trace=None):
    """Combined metric considering multiple factors"""
    exact = exact_match_metric(gold, pred, trace)
    length = length_metric(gold, pred, trace)
    keyword = keyword_overlap_metric(gold, pred, trace)
    
    # Weighted combination
    return (exact * 0.4 + length * 0.3 + keyword * 0.3)

class SummarizationEvaluator:
    """Evaluates summarization performance"""
    
    def __init__(self):
        self.metrics = {
            "exact_match": exact_match_metric,
            "length_appropriate": length_metric,
            "keyword_overlap": keyword_overlap_metric,
            "combined": combined_metric
        }
    
    def evaluate(self, model, test_data: List[Article]) -> Dict[str, float]:
        """Evaluate a model on test data"""
        results = {metric: [] for metric in self.metrics.keys()}
        
        for article in test_data:
            try:
                prediction = model(title=article.title, content=article.content)
                
                for metric_name, metric_func in self.metrics.items():
                    score = metric_func(article, prediction)
                    results[metric_name].append(score)
                    
            except Exception as e:
                print(f"Error evaluating article '{article.title}': {e}")
                # Add zero scores for failed evaluations
                for metric_name in self.metrics.keys():
                    results[metric_name].append(0.0)
        
        # Calculate averages
        avg_results = {}
        for metric_name, scores in results.items():
            avg_results[metric_name] = sum(scores) / len(scores) if scores else 0.0
        
        return avg_results

def run_summarization_experiment():
    """Run the complete summarization experiment"""
    print("🚀 Starting DSPy Summarization Experiment")
    print("=" * 50)
    
    # Setup DSPy
    setup_dspy()
    
    # Generate synthetic data
    print("\n📊 Generating synthetic data...")
    data_generator = SyntheticDataGenerator()
    articles = data_generator.generate_synthetic_articles(15)
    
    # Split into train and test sets
    random.shuffle(articles)
    train_articles = articles[:10]
    test_articles = articles[10:]
    
    print(f"Generated {len(articles)} articles:")
    print(f"  - Training set: {len(train_articles)} articles")
    print(f"  - Test set: {len(test_articles)} articles")
    
    # Create different summarizer versions
    summarizers = {
        "Basic": BasicSummarizer(),
        "Detailed": DetailedSummarizer(),
        "Refined": RefinedSummarizer()
    }
    
    # Initialize evaluator
    evaluator = SummarizationEvaluator()
    
    # Test each summarizer
    results = {}
    
    for name, summarizer in summarizers.items():
        print(f"\n Testing {name} Summarizer...")
        
        # Test on a few examples first
        print("Sample predictions:")
        for i, article in enumerate(test_articles[:2]):
            try:
                prediction = summarizer(title=article.title, content=article.content)
                print(f"  Article {i+1}: {article.title}")
                print(f"  Expected: {article.expected_summary}")
                print(f"  Predicted: {prediction.summary}")
                print()
            except Exception as e:
                print(f"  Error with article {i+1}: {e}")
        
        # Evaluate on full test set
        try:
            metrics = evaluator.evaluate(summarizer, test_articles)
            results[name] = metrics
            print(f"✅ {name} Summarizer Results:")
            for metric, score in metrics.items():
                print(f"  {metric}: {score:.3f}")
        except Exception as e:
            print(f"❌ Error evaluating {name} Summarizer: {e}")
            results[name] = {metric: 0.0 for metric in evaluator.metrics.keys()}
    
    # Compare results
    print("\n📈 Results Comparison:")
    print("-" * 40)
    
    # Print header
    metrics_list = list(evaluator.metrics.keys())
    header = f"{'Model':<12} " + " ".join(f"{metric:<15}" for metric in metrics_list)
    print(header)
    print("-" * len(header))
    
    # Print results
    for name, metrics in results.items():
        row = f"{name:<12} " + " ".join(f"{metrics[metric]:<15.3f}" for metric in metrics_list)
        print(row)
    
    # Find best model
    best_model = max(results.keys(), key=lambda x: results[x]['combined'])
    print(f"\n🏆 Best performing model: {best_model} (combined score: {results[best_model]['combined']:.3f})")
    
    return results, summarizers

def demonstrate_optimization():
    """Demonstrate DSPy optimization techniques"""
    print("\n🔄 Demonstrating DSPy Optimization")
    print("=" * 40)
    
    # Generate smaller dataset for optimization
    data_generator = SyntheticDataGenerator()
    train_articles = data_generator.generate_synthetic_articles(8)
    
    # Create training examples for DSPy
    train_examples = []
    for article in train_articles:
        example = dspy.Example(
            article_title=article.title,
            article_content=article.content,
            summary=article.expected_summary
        ).with_inputs("article_title", "article_content")
        train_examples.append(example)
    
    # Create a basic summarizer for optimization (use string signature)
    basic_summarizer = dspy.ChainOfThought("article_title, article_content -> summary")
    
    print("📚 Training examples created for optimization")
    print(f"Number of training examples: {len(train_examples)}")
    
    # Note: Full optimization would require API access
    print("\n💡 Optimization techniques available in DSPy:")
    print("  - BootstrapFewShot: Uses few-shot learning with examples")
    print("  - BootstrapFinetune: Fine-tunes the model")
    print("  - MIPROv2: Advanced prompt optimization")
    print("  - BetterTogether: Combines multiple optimization strategies")
    
    print("\n⚠️  Note: Full optimization requires API access and may incur costs")
    
    return basic_summarizer, train_examples

def save_results(results: Dict, filename: str = None):
    """Save experiment results to a file in test-data/dspy/*.json"""
    timestamp = datetime.now().isoformat()
    
    # Ensure directory exists
    out_dir = os.path.join("test-data", "dspy")
    os.makedirs(out_dir, exist_ok=True)
    if filename is None:
        filename = f"results_{timestamp.replace(':', '-')}.json"
    out_path = os.path.join(out_dir, filename)
    
    output_data = {
        "timestamp": timestamp,
        "results": results,
        "description": "DSPy summarization experiment results"
    }
    
    with open(out_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Results saved to {out_path}")

if __name__ == "__main__":
    print("🎯 DSPy Summarization Experiment")
    print("This sample demonstrates:")
    print("  - Synthetic data generation")
    print("  - Multiple summarization approaches")
    print("  - Evaluation metrics")
    print("  - Model comparison")
    print("  - Optimization techniques")
    print()
    
    try:
        # Run the main experiment
        results, models = run_summarization_experiment()
        
        # Demonstrate optimization
        basic_model, train_examples = demonstrate_optimization()
        
        # Save results
        save_results(results)
        
        print("\n✅ Experiment completed successfully!")
        print("\n📝 Next steps:")
        print("  1. Set up API keys for full functionality")
        print("  2. Try different DSPy optimizers")
        print("  3. Experiment with different model providers")
        print("  4. Add more sophisticated evaluation metrics")
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        print("This might be due to missing API keys or network issues.")
        print("The code structure and logic are still valid for demonstration purposes.")
