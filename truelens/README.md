# TruLens Evaluation Samples

This directory contains sample implementations for evaluating AI applications using the **TruLens** framework. TruLens provides a comprehensive suite of tools for evaluating and tracking LLM experiments, with a focus on providing detailed feedback on the quality and performance of AI agents and RAG systems.

## Overview

The primary example in this directory demonstrates how to evaluate a Retrieval-Augmented Generation (RAG) application using TruLens. The evaluation focuses on three key feedback functions:

- **Groundedness**: Measures whether the model's response is supported by the retrieved context, helping to identify hallucinations.
- **Context Relevance**: Assesses the relevance of the retrieved context to the input query.
- **Answer Relevance**: Evaluates how relevant the generated answer is to the original question.

## Setup and Installation

1.  **Install Dependencies**: Navigate to this directory and install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables**: Create a `.env` file in the root of the `ai-evaluation` repository and add your OpenAI API key:

    ```
    OPENAI_API_KEY="your-openai-api-key"
    ```

## Running the Evaluation

To run the RAG evaluation example, execute the following command from within the `truelens` directory:

```bash
python trulens_eval_sample.py
```

The script will perform the following actions:

1.  Load a document from a web URL.
2.  Create a RAG chain using LangChain.
3.  Wrap the RAG chain with TruLens for evaluation.
4.  Run the RAG chain with a sample question.
5.  Record the evaluation results, including groundedness, context relevance, and answer relevance.
6.  Print the evaluation records and feedback to the console.
7.  Launch the TruLens dashboard for detailed visualization of the evaluation results.

## TruLens Dashboard

After running the script, you can access the TruLens dashboard by navigating to the URL provided in the console output (usually `http://localhost:8501`). The dashboard provides an interactive interface to explore the evaluation results, including:

-   A summary of the feedback scores.
-   Detailed traces of the application's execution.
-   Side-by-side comparisons of different application versions.

