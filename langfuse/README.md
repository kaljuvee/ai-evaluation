# LangFuse Evaluation Samples

This directory contains sample implementations for evaluating AI applications using the **LangFuse** framework. LangFuse is an open-source observability and analytics platform for LLM applications, providing detailed tracing and evaluation capabilities.

## Overview

The example in this directory demonstrates how to trace and evaluate a simple question-answering interaction using LangFuse. The script performs the following steps:

1.  **Initializes LangFuse**: Sets up the LangFuse client with your API keys.
2.  **Creates a Callback Handler**: A LangFuse callback handler is created to automatically trace the execution of a LangChain model.
3.  **Runs a Model**: A LangChain `ChatOpenAI` model is invoked with a sample question, and the LangFuse callback handler captures the trace.
4.  **Scores the Interaction**: The script demonstrates how to programmatically add a score to the trace, in this case, a "user-satisfaction" score.
5.  **Prints Trace URL**: The URL to the detailed trace in the LangFuse UI is printed to the console.

## Setup and Installation

1.  **Install Dependencies**: Navigate to this directory and install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables**: Create a `.env` file in the root of the `ai-evaluation` repository and add your OpenAI and LangFuse API keys. You can get your LangFuse keys by signing up at [https://cloud.langfuse.com](https://cloud.langfuse.com).

    ```
    OPENAI_API_KEY="your-openai-api-key"
    LANGFUSE_PUBLIC_KEY="your-langfuse-public-key"
    LANGFUSE_SECRET_KEY="your-langfuse-secret-key"
    ```

## Running the Evaluation

To run the QA evaluation example, execute the following command from within the `langfuse` directory:

```bash
python langfuse_eval_sample.py
```

The script will print the model's response and a URL to the trace in the LangFuse UI.

## Viewing Results in the LangFuse UI

Open the trace URL in your browser to view the detailed interaction in the LangFuse UI. You will be able to see:

-   The input and output of the model.
-   The latency and token usage.
-   The score that was programmatically added to the trace.
-   A detailed timeline of the execution.

You can also manually add scores and comments to the trace directly in the UI, allowing for human-in-the-loop evaluation.

