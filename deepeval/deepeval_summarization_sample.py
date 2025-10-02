import os
from dotenv import load_dotenv
from deepeval import assert_test
from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Setup OpenAI model
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

def run_summarization_evaluation():
    """Runs a summarization evaluation example using DeepEval."""

    # Original document
    original_text = """
    The James Webb Space Telescope (JWST) is a space telescope designed primarily to conduct infrared
    astronomy. As the largest optical telescope in space, its high resolution and sensitivity allow it to
    view objects too old, distant, or faint for the Hubble Space Telescope. This enables investigations
    in many fields of astronomy and cosmology, such as observation of the first stars and the formation
    of the first galaxies, and detailed atmospheric characterization of potentially habitable exoplanets.
    """

    # Generated summary
    generated_summary = llm.predict(
        f"Summarize the following text in one sentence: \n\n{original_text}"
    )

    # Create a test case
    test_case = LLMTestCase(
        input=original_text,
        actual_output=generated_summary
    )

    # Create a summarization metric
    summarization_metric = SummarizationMetric(
        threshold=0.5,
        model="gpt-4",
        assessment_questions=[
            "Is the summary coherent and easy to understand?",
            "Does the summary accurately represent the main points of the original text?",
            "Is the summary concise and to the point?"
        ]
    )

    # Run the evaluation
    assert_test(test_case, [summarization_metric])

    print("--- Summarization Evaluation Results ---")
    print(f"Original Text: {original_text}")
    print(f"Generated Summary: {generated_summary}")
    print(f"Summarization Metric Score: {summarization_metric.score}")
    print(f"Summarization Metric Reason: {summarization_metric.reason}")

if __name__ == "__main__":
    run_summarization_evaluation()

