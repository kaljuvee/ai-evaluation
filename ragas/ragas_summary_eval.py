"""
Ragas Summary Evaluation Example

This script demonstrates how to evaluate text summarization using Ragas,
following the official documentation examples.
"""

import os
import asyncio
from typing import Dict, List, Any
import pandas as pd
import sys
import io
from datetime import datetime

# Ragas imports
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import BleuScore, AspectCritic

# LangChain imports for LLM wrapper
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper


class RagasSummaryEvaluator:
    """
    A class to evaluate text summarization using Ragas metrics.
    """
    
    def __init__(self, openai_api_key: str = None):
        """
        Initialize the evaluator with OpenAI API key.
        
        Args:
            openai_api_key: OpenAI API key. If None, will try to get from environment.
        """
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        elif not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
        # Initialize LLM for LLM-based metrics
        self.evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1
        ))
        
        # Initialize metrics
        self.bleu_metric = BleuScore()
        self.aspect_critic_metric = AspectCritic(
            name="summary_accuracy",
            llm=self.evaluator_llm,
            definition="Verify if the summary accurately captures all key details from the original text, including growth figures, market insights, and other essential information."
        )
    
    def evaluate_single_sample_non_llm(self, test_data: Dict[str, str]) -> float:
        """
        Evaluate a single sample using non-LLM metric (BleuScore).
        
        Args:
            test_data: Dictionary with 'user_input', 'response', and 'reference' keys
            
        Returns:
            BleuScore evaluation result
        """
        sample = SingleTurnSample(**test_data)
        return self.bleu_metric.single_turn_score(sample)
    
    async def evaluate_single_sample_llm(self, test_data: Dict[str, str]) -> int:
        """
        Evaluate a single sample using LLM-based metric (AspectCritic).
        
        Args:
            test_data: Dictionary with 'user_input' and 'response' keys
            
        Returns:
            AspectCritic evaluation result (1 for pass, 0 for fail)
        """
        sample = SingleTurnSample(**test_data)
        return await self.aspect_critic_metric.single_turn_ascore(sample)
    
    def create_sample_data(self) -> List[Dict[str, str]]:
        """
        Create sample test data for evaluation.
        
        Returns:
            List of test samples
        """
        return [
            {
                "user_input": "summarise given text\nThe company reported an 8% rise in Q3 2024, driven by strong performance in the Asian market. Sales in this region have significantly contributed to the overall growth. Analysts attribute this success to strategic marketing and product localization. The positive trend in the Asian market is expected to continue into the next quarter.",
                "response": "The company experienced an 8% increase in Q3 2024, largely due to effective marketing strategies and product adaptation, with expectations of continued growth in the coming quarter.",
                "reference": "The company reported an 8% growth in Q3 2024, primarily driven by strong sales in the Asian market, attributed to strategic marketing and localized products, with continued growth anticipated in the next quarter."
            },
            {
                "user_input": "summarise given text\nThe Q2 earnings report revealed a significant 15% increase in revenue, primarily driven by expansion into European markets. The company's strategic investments in technology infrastructure have resulted in improved operational efficiency. Customer satisfaction scores have reached an all-time high of 92%, reflecting the success of recent product improvements.",
                "response": "Q2 earnings showed 15% revenue growth through European expansion, with technology investments improving efficiency and customer satisfaction reaching 92%.",
                "reference": "The Q2 earnings report showed a 15% revenue increase driven by European market expansion, with technology investments improving efficiency and customer satisfaction reaching 92%."
            },
            {
                "user_input": "summarise given text\nIn 2023, North American sales experienced a 5% decline, attributed to increased competition and supply chain disruptions. However, the company's digital transformation initiatives have positioned it well for future growth. The launch of new AI-powered products has generated significant interest from enterprise clients.",
                "response": "North American sales declined 5% in 2023 due to competition and supply chain issues, but digital transformation and AI products show promise for future growth.",
                "reference": "North American sales declined 5% in 2023 due to competition and supply chain disruptions, but digital transformation and AI product launches show promise for future growth."
            }
        ]
    
    def load_dataset_from_hf(self, dataset_name: str = "explodinggradients/earning_report_summary", split: str = "train") -> EvaluationDataset:
        """
        Load a dataset from Hugging Face Hub.
        
        Args:
            dataset_name: Name of the dataset on Hugging Face
            split: Dataset split to load
            
        Returns:
            EvaluationDataset object
        """
        try:
            from datasets import load_dataset
            eval_dataset = load_dataset(dataset_name, split=split)
            eval_dataset = EvaluationDataset.from_hf_dataset(eval_dataset)
            return eval_dataset
        except Exception as e:
            print(f"Error loading dataset from Hugging Face: {e}")
            print("Falling back to sample data...")
            return self.create_evaluation_dataset_from_samples()
    
    def create_evaluation_dataset_from_samples(self) -> EvaluationDataset:
        """
        Create an EvaluationDataset from sample data.
        
        Returns:
            EvaluationDataset object
        """
        sample_data = self.create_sample_data()
        # Remove 'reference' key for LLM-based evaluation
        llm_data = [{k: v for k, v in sample.items() if k != 'reference'} for sample in sample_data]
        return EvaluationDataset.from_dict(llm_data)
    
    async def evaluate_dataset(self, dataset: EvaluationDataset = None) -> Dict[str, Any]:
        """
        Evaluate a dataset using LLM-based metrics.
        
        Args:
            dataset: EvaluationDataset to evaluate. If None, uses sample data.
            
        Returns:
            Dictionary containing evaluation results
        """
        if dataset is None:
            dataset = self.create_evaluation_dataset_from_samples()
        
        print(f"Evaluating dataset with {len(dataset)} samples...")
        print(f"Dataset features: {dataset.features()}")
        
        # Evaluate using LLM-based metric
        results = evaluate(dataset, metrics=[self.aspect_critic_metric])
        
        return results
    
    def export_results_to_pandas(self, results) -> pd.DataFrame:
        """
        Export evaluation results to pandas DataFrame.
        
        Args:
            results: Evaluation results from ragas.evaluate()
            
        Returns:
            Pandas DataFrame with sample-level results
        """
        return results.to_pandas()
    
    def save_results_to_csv(self, results, filename: str = "summary_evaluation_results.csv"):
        """
        Save evaluation results to CSV file.
        
        Args:
            results: Evaluation results from ragas.evaluate()
            filename: Name of the CSV file to save
        """
        df = self.export_results_to_pandas(results)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")


async def main():
    """
    Main function to demonstrate Ragas summary evaluation.
    """
    # Prepare to capture all console output
    output_buffer = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = output_buffer

    try:
        print("🚀 Ragas Summary Evaluation Demo")
        print("=" * 50)
        
        # Initialize evaluator
        evaluator = RagasSummaryEvaluator()
        
        # Example 1: Non-LLM metric evaluation (single sample)
        print("\n1. Non-LLM Metric Evaluation (BleuScore)")
        print("-" * 40)
        
        sample_data = evaluator.create_sample_data()[0]
        bleu_score = evaluator.evaluate_single_sample_non_llm(sample_data)
        print(f"BleuScore: {bleu_score:.3f}")
        print(f"Input: {sample_data['user_input'][:100]}...")
        print(f"Response: {sample_data['response']}")
        print(f"Reference: {sample_data['reference']}")
        
        # Example 2: LLM-based metric evaluation (single sample)
        print("\n2. LLM-based Metric Evaluation (AspectCritic)")
        print("-" * 40)
        
        llm_sample = {k: v for k, v in sample_data.items() if k != 'reference'}
        aspect_score = await evaluator.evaluate_single_sample_llm(llm_sample)
        print(f"AspectCritic Score: {aspect_score} ({'PASS' if aspect_score == 1 else 'FAIL'})")
        print(f"Input: {llm_sample['user_input'][:100]}...")
        print(f"Response: {llm_sample['response']}")
        
        # Example 3: Dataset evaluation
        print("\n3. Dataset Evaluation")
        print("-" * 40)
        
        try:
            # Try to load from Hugging Face
            dataset = evaluator.load_dataset_from_hf()
        except:
            # Fall back to sample data
            dataset = evaluator.create_evaluation_dataset_from_samples()
        
        results = await evaluator.evaluate_dataset(dataset)
        print(f"Overall Results: {results}")
        
        # Export results
        df = evaluator.export_results_to_pandas(results)
        print(f"\nSample-level results (first 3 rows):")
        print(df.head(3))
        
        # Save results
        evaluator.save_results_to_csv(results)
        
        print("\n✅ Evaluation completed successfully!")
    finally:
        # Restore stdout
        sys.stdout = sys_stdout
        # Write output to markdown file
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        output_path = f"../test-data/ragas/test_results_{timestamp}.md"
        with open(output_path, "w") as f:
            f.write("```markdown\n")
            f.write(output_buffer.getvalue())
            f.write("\n```")
        print(f"\n[Console output written to {output_path}]")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
