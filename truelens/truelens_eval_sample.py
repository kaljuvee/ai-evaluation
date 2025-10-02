import os
from dotenv import load_dotenv
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from trulens_eval import Feedback, Tru, TruChain, OpenAI as fOpenAI

# Load environment variables
load_dotenv()

# Setup TruLens
tru = Tru()
tru.reset_database()

# Setup OpenAI provider for TruLens
fopenai = fOpenAI()

# --- Feedback Functions ---
groundedness = Feedback(fopenai.groundedness_measure_with_cot_reasons).on_input_output()
context_relevance = Feedback(fopenai.context_relevance_with_cot_reasons).on_input().on_output()
answer_relevance = Feedback(fopenai.relevance_with_cot_reasons).on_input_output()

# --- RAG Example ---
def run_rag_evaluation():
    """Runs a RAG evaluation example using TruLens."""

    # Load, chunk and index the contents of the blog.
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Wrap the chain with TruLens
    tru_rag_chain = TruChain(
        rag_chain,
        app_id="RAG_v1",
        feedbacks=[groundedness, context_relevance, answer_relevance]
    )

    # Run evaluation
    with tru_rag_chain as recording:
        tru_rag_chain("What is Task Decomposition?")

    # Get TruLens records and feedback
    records, feedback = tru.get_records_and_feedback(app_ids=["RAG_v1"])
    print("--- RAG Evaluation Records ---")
    print(records.head())
    print("--- RAG Evaluation Feedback ---")
    print(feedback)

    # Launch the TruLens dashboard
    tru.run_dashboard()

if __name__ == "__main__":
    run_rag_evaluation()

