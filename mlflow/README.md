# MLflow Evaluation Samples

This directory contains sample implementations for evaluating AI applications using the **MLflow** framework. MLflow is a popular open-source platform for managing the end-to-end machine learning lifecycle, including robust capabilities for model evaluation.

## Overview

The example in this directory demonstrates how to evaluate a question-answering (QA) model using MLflow's evaluation tools. The script performs the following steps:

1.  **Sets up an MLflow Experiment**: Creates a new experiment named "LLM Evaluation" to track the evaluation run.
2.  **Creates a Dataset**: A small pandas DataFrame is created with sample questions and their corresponding ground truth answers.
3.  **Defines a Model**: A LangChain `ChatOpenAI` model is used as the QA model.
4.  **Runs Evaluation**: The `mlflow.evaluate()` function is used to assess the model's performance against the provided dataset.
5.  **Logs Results**: The evaluation results, including metrics, are logged to the MLflow experiment.

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

To run the QA evaluation example, execute the following command from within the `mlflow` directory:

```bash
python mlflow_eval_sample.py
```

The script will print the evaluation metrics to the console and provide the run ID for the MLflow experiment.

## Viewing Results in the MLflow UI

To visualize the evaluation results, you need to launch the MLflow UI. Open a new terminal and run the following command from the root of the `ai-evaluation` repository:

```bash
mlflow ui
```

This will start a local web server (usually at `http://localhost:5000`). Open this URL in your browser, and you will see the MLflow UI. Navigate to the "LLM Evaluation" experiment to find your run and explore the detailed evaluation results, including metrics and artifacts.

