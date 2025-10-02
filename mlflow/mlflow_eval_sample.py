import os
import mlflow
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mlflow.models.signature import infer_signature

# Load environment variables
load_dotenv()

# Setup MLflow experiment
mlflow.set_experiment("LLM Evaluation")

def run_mlflow_evaluation():
    """Runs a question-answering evaluation example using MLflow."""

    # Create a simple dataset
    eval_data = pd.DataFrame(
        {
            "inputs": [
                "What is MLflow?",
                "What is the capital of France?",
            ],
            "ground_truth": [
                "MLflow is an open-source platform for managing the end-to-end machine learning lifecycle.",
                "The capital of France is Paris.",
            ],
        }
    )

    with mlflow.start_run() as run:
        # Log the dataset as an artifact
        mlflow.log_artifact(eval_data.to_csv(index=False), "eval_data.csv")

        # Define the model
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        # Define the prediction function
        def predict(inputs):
            messages = [{"role": "user", "content": q} for q in inputs["inputs"]]
            responses = [llm.predict(messages=m) for m in messages]
            return pd.DataFrame({"outputs": responses})

        # Run the evaluation
        results = mlflow.evaluate(
            data=eval_data,
            model_type="question-answering",
            targets="ground_truth",
            predictions=predict(eval_data),
        )

        # Print the results
        print("--- MLflow Evaluation Results ---")
        print(results.metrics)

        # You can also view the results in the MLflow UI
        print(f"\nTo view the results in the MLflow UI, run 'mlflow ui' and navigate to the experiment 'LLM Evaluation'.")
        print(f"Run ID: {run.info.run_id}")

if __name__ == "__main__":
    run_mlflow_evaluation()

