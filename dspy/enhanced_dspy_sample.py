"""
Enhanced DSPy Sample: Multi-Prompt Optimization with Detailed Metrics

This enhanced sample demonstrates:
1. Multiple prompt versions for summarization
2. MIPRO optimization techniques
3. Detailed evaluation metrics with explanations
4. Comprehensive results including data and summaries
5. Performance comparison across different prompting strategies
"""

import dspy
import random
import json
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
        print("Using mock configuration for demonstration...")
        dspy.configure(lm=None)

@dataclass
class Article:
    """Represents an article with title, content, and expected summary"""
    title: str
    content: str
    expected_summary: str
    category: str
    difficulty: str  # easy, medium, hard

@dataclass
class EvaluationResult:
    """Detailed evaluation result for a single article"""
    article_title: str
    article_content: str
    expected_summary: str
    predicted_summary: str
    exact_match: float
    length_appropriate: float
    keyword_overlap: float
    combined_score: float
    word_count_expected: int
    word_count_predicted: int
    common_keywords: List[str]
    missing_keywords: List[str]

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

# Define different DSPy modules for summarization with various prompt strategies
class BasicSummarizer(dspy.Module):
    """Basic summarization module - simple prompt"""
    
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
    """Refined summarization with quality instructions"""
    
    def __init__(self):
        super().__init__()
        self.summarizer = dspy.ChainOfThought("article_title, article_content -> summary")
    
    def forward(self, title: str, content: str):
        enhanced_content = f"Please provide a clear, concise summary of the following article. Focus on the main points and avoid unnecessary details.\n\nTitle: {title}\n\nContent: {content}"
        return self.summarizer(article_title=title, article_content=enhanced_content)

class AcademicSummarizer(dspy.Module):
    """Academic-style summarization"""
    
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

class ExecutiveSummarizer(dspy.Module):
    """Executive-style summarization"""
    
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

class TechnicalSummarizer(dspy.Module):
    """Technical summarization with technical focus"""
    
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

# Enhanced evaluation metrics with detailed explanations
def exact_match_metric(gold, pred, trace=None):
    """Exact match metric - checks if predicted summary exactly matches expected summary"""
    return gold.expected_summary.lower().strip() == pred.summary.lower().strip()

def length_metric(gold, pred, trace=None):
    """Length appropriateness metric - checks if summary length is suitable"""
    gold_length = len(gold.expected_summary.split())
    pred_length = len(pred.summary.split())
    
    # Summary should be within 50% of expected length
    min_length = gold_length * 0.5
    max_length = gold_length * 1.5
    
    return min_length <= pred_length <= max_length

def keyword_overlap_metric(gold, pred, trace=None):
    """Keyword overlap metric - measures semantic similarity through keyword matching"""
    gold_words = set(gold.expected_summary.lower().split())
    pred_words = set(pred.summary.lower().split())
    
    if not gold_words:
        return 0.0
    
    overlap = len(gold_words.intersection(pred_words))
    return overlap / len(gold_words)

def content_coverage_metric(gold, pred, trace=None):
    """Content coverage metric - measures how well the summary covers the main content"""
    # Simple implementation: check if key terms from title appear in summary
    title_words = set(gold.title.lower().split())
    summary_words = set(pred.summary.lower().split())
    
    if not title_words:
        return 0.0
    
    coverage = len(title_words.intersection(summary_words))
    return coverage / len(title_words)

def combined_metric(gold, pred, trace=None):
    """Combined metric considering multiple factors with weighted combination"""
    exact = exact_match_metric(gold, pred, trace)
    length = length_metric(gold, pred, trace)
    keyword = keyword_overlap_metric(gold, pred, trace)
    coverage = content_coverage_metric(gold, pred, trace)
    
    # Weighted combination - can be tuned based on importance
    return (exact * 0.3 + length * 0.2 + keyword * 0.3 + coverage * 0.2)

class EnhancedSummarizationEvaluator:
    """Enhanced evaluator with detailed metrics and explanations"""
    
    def __init__(self):
        self.metrics = {
            "exact_match": exact_match_metric,
            "length_appropriate": length_metric,
            "keyword_overlap": keyword_overlap_metric,
            "content_coverage": content_coverage_metric,
            "combined": combined_metric
        }
        
        self.metric_explanations = {
            "exact_match": "Measures perfect word-for-word matching between expected and predicted summaries. High scores indicate exact replication.",
            "length_appropriate": "Checks if the summary length is within 50% of the expected length. Ensures summaries are neither too short nor too long.",
            "keyword_overlap": "Measures the percentage of important words from the expected summary that appear in the predicted summary. Indicates semantic similarity.",
            "content_coverage": "Measures how well the summary covers key terms from the article title. Ensures main topics are addressed.",
            "combined": "Weighted combination of all metrics (30% exact match, 20% length, 30% keyword overlap, 20% coverage). Overall performance indicator."
        }
    
    def evaluate_single_article(self, model, article: Article) -> EvaluationResult:
        """Evaluate a single article and return detailed results"""
        try:
            prediction = model(title=article.title, content=article.content)
            
            # Calculate metrics
            exact_match = exact_match_metric(article, prediction)
            length_appropriate = length_metric(article, prediction)
            keyword_overlap = keyword_overlap_metric(article, prediction)
            content_coverage = content_coverage_metric(article, prediction)
            combined_score = combined_metric(article, prediction)
            
            # Calculate word counts
            word_count_expected = len(article.expected_summary.split())
            word_count_predicted = len(prediction.summary.split())
            
            # Find common and missing keywords
            expected_words = set(article.expected_summary.lower().split())
            predicted_words = set(prediction.summary.lower().split())
            common_keywords = list(expected_words.intersection(predicted_words))
            missing_keywords = list(expected_words - predicted_words)
            
            return EvaluationResult(
                article_title=article.title,
                article_content=article.content,
                expected_summary=article.expected_summary,
                predicted_summary=prediction.summary,
                exact_match=exact_match,
                length_appropriate=length_appropriate,
                keyword_overlap=keyword_overlap,
                combined_score=combined_score,
                word_count_expected=word_count_expected,
                word_count_predicted=word_count_predicted,
                common_keywords=common_keywords,
                missing_keywords=missing_keywords
            )
            
        except Exception as e:
            print(f"Error evaluating article '{article.title}': {e}")
            return None
    
    def evaluate(self, model, test_data: List[Article]) -> Dict[str, Any]:
        """Evaluate a model on test data with detailed results"""
        results = {metric: [] for metric in self.metrics.keys()}
        detailed_results = []
        
        for article in test_data:
            eval_result = self.evaluate_single_article(model, article)
            if eval_result:
                detailed_results.append(eval_result)
                
                for metric_name, metric_func in self.metrics.items():
                    score = metric_func(article, eval_result)
                    results[metric_name].append(score)
            else:
                # Add zero scores for failed evaluations
                for metric_name in self.metrics.keys():
                    results[metric_name].append(0.0)
        
        # Calculate averages
        avg_results = {}
        for metric_name, scores in results.items():
            avg_results[metric_name] = sum(scores) / len(scores) if scores else 0.0
        
        return {
            "average_metrics": avg_results,
            "detailed_results": [asdict(result) for result in detailed_results],
            "metric_explanations": self.metric_explanations
        }

def demonstrate_mipro_optimization():
    """Demonstrate MIPRO optimization techniques"""
    print("\n🔄 Demonstrating MIPRO Optimization")
    print("=" * 50)
    
    # Generate dataset for optimization
    data_generator = SyntheticDataGenerator()
    train_articles = data_generator.generate_synthetic_articles(10)
    
    # Create training examples for DSPy
    train_examples = []
    for article in train_articles:
        example = dspy.Example(
            article_title=article.title,
            article_content=article.content,
            summary=article.expected_summary
        ).with_inputs("article_title", "article_content")
        train_examples.append(example)
    
    print("📚 Created training examples for MIPRO optimization")
    print(f"Number of training examples: {len(train_examples)}")
    
    # Create base summarizer
    base_summarizer = dspy.ChainOfThought("article_title, article_content -> summary")
    
    print("\n💡 MIPRO Optimization Process:")
    print("1. Bootstrapping Stage: Generate multiple candidate prompts")
    print("2. Evaluation Stage: Test each prompt on validation set")
    print("3. Refinement Stage: Iteratively improve the best prompts")
    print("4. Selection Stage: Choose the optimal prompt configuration")
    
    try:
        # Create MIPRO optimizer
        optimizer = dspy.MIPROv2(
            metric=combined_metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=2,
            num_candidate_programs=3,
            num_threads=1
        )
        
        print("\n🔄 Running MIPRO optimization...")
        print("This will test multiple prompt variations and select the best performing one.")
        
        # Optimize the model
        optimized_model = optimizer.compile(base_summarizer, trainset=train_examples[:6])
        
        print("✅ MIPRO optimization completed!")
        print("Optimized model ready for use")
        
        return optimized_model, train_examples
        
    except Exception as e:
        print(f"❌ MIPRO optimization failed: {e}")
        print("This might be due to API limitations or network issues")
        return base_summarizer, train_examples

def run_enhanced_experiment():
    """Run the enhanced summarization experiment with multiple prompt versions"""
    print("🚀 Enhanced DSPy Summarization Experiment")
    print("=" * 60)
    
    # Setup DSPy
    setup_dspy()
    
    # Generate synthetic data
    print("\n📊 Generating synthetic data...")
    data_generator = SyntheticDataGenerator()
    articles = data_generator.generate_synthetic_articles(20)
    
    # Split into train and test sets
    random.shuffle(articles)
    train_articles = articles[:12]
    test_articles = articles[12:]
    
    print(f"Generated {len(articles)} articles:")
    print(f"  - Training set: {len(train_articles)} articles")
    print(f"  - Test set: {len(test_articles)} articles")
    
    # Create different summarizer versions
    summarizers = {
        "Basic": BasicSummarizer(),
        "Detailed": DetailedSummarizer(),
        "Refined": RefinedSummarizer(),
        "Academic": AcademicSummarizer(),
        "Executive": ExecutiveSummarizer(),
        "Technical": TechnicalSummarizer()
    }
    
    # Initialize evaluator
    evaluator = EnhancedSummarizationEvaluator()
    
    # Test each summarizer
    all_results = {}
    
    for name, summarizer in summarizers.items():
        print(f"\n🔍 Testing {name} Summarizer...")
        
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
            results = evaluator.evaluate(summarizer, test_articles)
            all_results[name] = results
            print(f"✅ {name} Summarizer Results:")
            for metric, score in results["average_metrics"].items():
                print(f"  {metric}: {score:.3f}")
        except Exception as e:
            print(f"❌ Error evaluating {name} Summarizer: {e}")
            all_results[name] = {
                "average_metrics": {metric: 0.0 for metric in evaluator.metrics.keys()},
                "detailed_results": [],
                "metric_explanations": evaluator.metric_explanations
            }
    
    # Compare results
    print("\n📈 Results Comparison:")
    print("-" * 80)
    
    # Print header
    metrics_list = list(evaluator.metrics.keys())
    header = f"{'Model':<12} " + " ".join(f"{metric:<15}" for metric in metrics_list)
    print(header)
    print("-" * len(header))
    
    # Print results
    for name, results in all_results.items():
        avg_metrics = results["average_metrics"]
        row = f"{name:<12} " + " ".join(f"{avg_metrics[metric]:<15.3f}" for metric in metrics_list)
        print(row)
    
    # Find best model
    best_model = max(all_results.keys(), key=lambda x: all_results[x]["average_metrics"]['combined'])
    print(f"\n🏆 Best performing model: {best_model} (combined score: {all_results[best_model]['average_metrics']['combined']:.3f})")
    
    return all_results, summarizers, test_articles

def save_enhanced_results(results: Dict, test_articles: List[Article], filename: str = None):
    """Save enhanced experiment results to a file"""
    timestamp = datetime.now().isoformat()
    
    # Ensure directory exists
    out_dir = os.path.join("test-data", "dspy")
    os.makedirs(out_dir, exist_ok=True)
    if filename is None:
        filename = f"enhanced_results_{timestamp.replace(':', '-')}.json"
    out_path = os.path.join(out_dir, filename)
    
    # Prepare comprehensive output data
    output_data = {
        "timestamp": timestamp,
        "experiment_info": {
            "description": "Enhanced DSPy summarization experiment with multiple prompt versions",
            "models_tested": list(results.keys()),
            "test_articles_count": len(test_articles),
            "metrics_used": list(results[list(results.keys())[0]]["metric_explanations"].keys())
        },
        "test_data": [asdict(article) for article in test_articles],
        "results": results,
        "metric_explanations": results[list(results.keys())[0]]["metric_explanations"],
        "summary": {
            "best_model": max(results.keys(), key=lambda x: results[x]["average_metrics"]['combined']),
            "best_score": max(results[x]["average_metrics"]['combined'] for x in results.keys()),
            "average_scores": {
                metric: sum(results[x]["average_metrics"][metric] for x in results.keys()) / len(results)
                for metric in results[list(results.keys())[0]]["average_metrics"].keys()
            }
        }
    }
    
    with open(out_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Enhanced results saved to {out_path}")
    return out_path

if __name__ == "__main__":
    print("🎯 Enhanced DSPy Summarization Experiment")
    print("This enhanced sample demonstrates:")
    print("  - Multiple prompt versions and strategies")
    print("  - Detailed evaluation metrics with explanations")
    print("  - MIPRO optimization techniques")
    print("  - Comprehensive results analysis")
    print("  - Performance comparison across different approaches")
    print()
    
    try:
        # Run the enhanced experiment
        results, models, test_articles = run_enhanced_experiment()
        
        # Demonstrate MIPRO optimization
        optimized_model, train_examples = demonstrate_mipro_optimization()
        
        # Save enhanced results
        output_file = save_enhanced_results(results, test_articles)
        
        print("\n✅ Enhanced experiment completed successfully!")
        print(f"\n📊 Results saved to: {output_file}")
        print("\n📝 Key Insights:")
        print("  - Different prompt strategies show varying performance")
        print("  - MIPRO can automatically find optimal prompt configurations")
        print("  - Detailed metrics provide insights into model behavior")
        print("  - Results include full test data and predictions for analysis")
        
    except Exception as e:
        print(f"\n❌ Enhanced experiment failed: {e}")
        print("This might be due to missing API keys or network issues.")
        print("The code structure and logic are still valid for demonstration purposes.") 