from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from langchain.evaluation import load_evaluator
from langchain.evaluation import load_dataset
from langchain.evaluation.criteria import Criteria
from langchain.evaluation.embedding_distance import EmbeddingDistance
from langchain.evaluation.string_distance import StringDistance
from langchain_openai import ChatOpenAI

# Initialize the language model
llm = ChatOpenAI(temperature=0)

def run_evaluation_examples():
    # 1. Basic QA Evaluation
    qa_evaluator = load_evaluator("qa")
    qa_result = qa_evaluator.evaluate_strings(
        prediction="The capital of France is Paris",
        input="What is the capital of France?",
        reference="Paris is the capital of France"
    )
    print("\nQA Evaluation Result:", qa_result)

    # 2. Criteria-based Evaluation
    criteria_evaluator = load_evaluator("criteria")
    criteria_result = criteria_evaluator.evaluate_strings(
        prediction="The weather is sunny today with a high of 75°F",
        input="What's the weather like?",
        criteria=[
            Criteria.CONCISENESS,
            Criteria.RELEVANCE,
            Criteria.CORRECTNESS
        ]
    )
    print("\nCriteria Evaluation Result:", criteria_result)

    # 3. Embedding Distance Evaluation
    embedding_evaluator = load_evaluator("embedding_distance")
    embedding_result = embedding_evaluator.evaluate_strings(
        prediction="The cat sat on the mat",
        reference="A feline was resting on the carpet",
        distance_metric=EmbeddingDistance.COSINE
    )
    print("\nEmbedding Distance Result:", embedding_result)

    # 4. String Distance Evaluation
    string_evaluator = load_evaluator("string_distance")
    string_result = string_evaluator.evaluate_strings(
        prediction="Hello World",
        reference="Hello World!",
        distance_metric=StringDistance.LEVENSHTEIN
    )
    print("\nString Distance Result:", string_result)

    # 5. Pairwise String Comparison
    pairwise_evaluator = load_evaluator("pairwise_string")
    pairwise_result = pairwise_evaluator.evaluate_string_pairs(
        prediction=(
            "The quick brown fox jumps over the lazy dog",
            "A fast brown fox leaps over a sleepy dog"
        )
    )
    print("\nPairwise Comparison Result:", pairwise_result)

    # 6. Load and use a dataset
    try:
        dataset = load_dataset("llm-math")
        print("\nDataset loaded successfully:", dataset)
    except Exception as e:
        print("\nError loading dataset:", str(e))

if __name__ == "__main__":
    run_evaluation_examples()
