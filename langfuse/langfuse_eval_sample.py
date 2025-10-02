import os
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Initialize Langfuse
langfuse = Langfuse(
    public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
    host="https://cloud.langfuse.com"
)

# Create a Langfuse callback handler
langfuse_callback_handler = CallbackHandler()

def run_langfuse_evaluation():
    """Runs a question-answering evaluation example using Langfuse."""

    # Define the model and the chain
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Run the chain with the Langfuse callback handler
    response = llm.invoke(
        "What is Langfuse?",
        config={"callbacks": [langfuse_callback_handler]}
    )

    # Get the trace information
    trace_data = langfuse_callback_handler.get_trace()

    # Manually score the interaction
    trace_data.score(
        name="user-satisfaction",
        value=1,
        comment="The user was satisfied with the response."
    )

    # Print the response and trace URL
    print("--- Langfuse Evaluation Results ---")
    print(f"Response: {response.content}")
    print(f"Trace URL: {trace_data.get_trace_url()}")

    # Shutdown Langfuse to ensure all traces are sent
    langfuse.flush()

if __name__ == "__main__":
    run_langfuse_evaluation()

