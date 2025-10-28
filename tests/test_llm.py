import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def test_openai_basic():
    load_dotenv()
    api = os.getenv("OPENAI_API_KEY")
    if not api:
        print("OPENAI_API_KEY missing in environment/.env; skipping call")
        return

    # Prefer the widely available model for smoke
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    except TypeError:
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    answer = llm.invoke("What is the capital of France?")
    text = getattr(answer, "content", str(answer))
    print("LLM answer:", text)


