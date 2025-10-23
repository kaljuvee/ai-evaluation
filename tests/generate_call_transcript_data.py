import os
from datetime import datetime
from typing import List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_prompt_texts() -> List[str]:
    # Project root is one directory above this file's directory
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    system_path = os.path.join(root, "prompts", "sales_call_transcript_system.txt")
    user_examples_path = os.path.join(root, "prompts", "sales_call_transcript_user_examples.txt")
    with open(system_path, "r", encoding="utf-8") as sf:
        system_text = sf.read().strip()
    with open(user_examples_path, "r", encoding="utf-8") as uf:
        user_examples = uf.read().strip()
    return [system_text, user_examples]


def generate_transcripts(
    output_dir: str = "test-data/call_transcripts",
    num_files: int = 100,
    target_chars: int = 1000,
) -> None:
    load_dotenv()
    ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Build LLM with compatibility across langchain versions
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    except TypeError:
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

    system_text, user_examples = load_prompt_texts()

    for i in range(1, num_files + 1):
        user_prompt = (
            f"Industry: SaaS | Product: CRM | Persona: VP Sales. "
            f"Target length around {target_chars} characters.\n\n"
            f"{user_examples}"
        )
        try:
            ai_msg = llm.invoke([SystemMessage(content=system_text), HumanMessage(content=user_prompt)])
            content_text = getattr(ai_msg, "content", str(ai_msg))
        except Exception:
            # Fallback: send as a single string
            ai_msg = llm.invoke(system_text + "\n\n" + user_prompt)
            content_text = getattr(ai_msg, "content", str(ai_msg))

        file_name = f"call_{i:03d}_{timestamp}.txt"
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content_text)


if __name__ == "__main__":
    generate_transcripts()


