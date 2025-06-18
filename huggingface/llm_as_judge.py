import re
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset
from huggingface_hub import InferenceClient

# Initialize the LLM client
def init_llm_client(model_id="mistralai/Mixtral-8x7B-Instruct-v0.1"):
    return InferenceClient(
        model=model_id,
        timeout=120,
    )

# Basic judge prompt template
BASIC_JUDGE_PROMPT = """
You will be given a user_question and system_answer couple.
Your task is to provide a 'total rating' scoring how well the system_answer answers the user concerns expressed in the user_question.
Give your answer as a float on a scale of 0 to 10, where 0 means that the system_answer is not helpful at all, and 10 means that the answer completely and helpfully addresses the question.

Provide your feedback as follows:

Feedback:::
Total rating: (your rating, as a float between 0 and 10)

Now here are the question and answer.

Question: {question}
Answer: {answer}

Feedback:::
Total rating: """

# Improved judge prompt with better structure and scale
IMPROVED_JUDGE_PROMPT = """
You will be given a user_question and system_answer couple.
Your task is to provide a 'total rating' scoring how well the system_answer answers the user concerns expressed in the user_question.
Give your answer on a scale of 1 to 4, where 1 means that the system_answer is not helpful at all, and 4 means that the system_answer completely and helpfully addresses the user_question.

Here is the scale you should use to build your answer:
1: The system_answer is terrible: completely irrelevant to the question asked, or very partial
2: The system_answer is mostly not helpful: misses some key aspects of the question
3: The system_answer is mostly helpful: provides support, but still could be improved
4: The system_answer is excellent: relevant, direct, detailed, and addresses all the concerns raised in the question

Provide your feedback as follows:

Feedback:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 4)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and answer.

Question: {question}
Answer: {answer}

Provide your feedback. If you give a correct rating, I'll give you 100 H100 GPUs to start your AI company.
Feedback:::
Evaluation: """

def extract_judge_score(answer: str, split_str: str = "Total rating:") -> float:
    """Extract the numerical score from the LLM judge's response."""
    try:
        if split_str in answer:
            rating = answer.split(split_str)[1]
        else:
            rating = answer
        digit_groups = [el.strip() for el in re.findall(r"\d+(?:\.\d+)?", rating)]
        return float(digit_groups[0])
    except Exception as e:
        print(f"Error extracting score: {e}")
        return None

def evaluate_with_llm_judge(questions_answers, llm_client, prompt_template=IMPROVED_JUDGE_PROMPT):
    """Evaluate a list of question-answer pairs using the LLM judge."""
    results = []
    
    for qa_pair in tqdm(questions_answers):
        question = qa_pair["question"]
        answer = qa_pair["answer"]
        
        # Get LLM judge's evaluation
        response = llm_client.text_generation(
            prompt=prompt_template.format(question=question, answer=answer),
            max_new_tokens=500,
        )
        
        # Extract score
        score = extract_judge_score(response)
        
        results.append({
            "question": question,
            "answer": answer,
            "llm_judge_response": response,
            "llm_judge_score": score
        })
    
    return pd.DataFrame(results)

def main():
    # Initialize LLM client
    llm_client = init_llm_client()
    
    # Example question-answer pairs
    sample_qa_pairs = [
        {
            "question": "What is the capital of France?",
            "answer": "The capital of France is Paris. It is known for landmarks like the Eiffel Tower and the Louvre Museum."
        },
        {
            "question": "How do I make a chocolate cake?",
            "answer": "I'm not sure about that. Maybe you should check a recipe book."
        }
    ]
    
    # Evaluate using the improved prompt
    results_df = evaluate_with_llm_judge(sample_qa_pairs, llm_client)
    
    # Display results
    print("\nEvaluation Results:")
    print(results_df[["question", "llm_judge_score", "llm_judge_response"]])

if __name__ == "__main__":
    main()
