import argparse
import pickle
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import warnings
warnings.filterwarnings("ignore")

from get_embedding_function import get_embedding_function

FAISS_PATH = "faiss_index"

PROMPT_TEMPLATE = """
You are a tutor who helps me to study the subject "Distributed Systems" in the study program "Digital Business Engineering" at the Hermann Hollerith Zentrum (HHZ). I will provide you with slides for this subject. Your tasks include clarifying questions about the lecture content, creating exam-relevant questions based on the slide set, evaluating your own questions and evaluating my answers to these questions. Your answers and evaluations should be precise, contextualized and easy to understand. Make sure that each generated question has a reference to the corresponding slide.:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for the LLM (default: 0.7)")
    args = parser.parse_args()
    query_text = args.query_text
    temperature = args.temperature
    query_rag(query_text, temperature)

def query_rag(query_text: str, temperature: float):
    embedding_function = get_embedding_function()
    with open(FAISS_PATH, "rb") as f:
        db = pickle.load(f)
    db.embedding_function = embedding_function

    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="llama3", temperature=temperature)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

if __name__ == "__main__":
    main()