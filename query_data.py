import argparse
import pickle
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import warnings
import time
from get_embedding_function import get_embedding_function

warnings.filterwarnings("ignore")

FAISS_PATH = "faiss_index"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}

"""

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for the LLM (default: 0.3)")
    args = parser.parse_args()
    query_text = args.query_text
    temperature = args.temperature
    
    query_rag(query_text, temperature)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

def query_rag(query_text: str, temperature: float):
    # Load FAISS index
    load_start = time.time()
    embedding_function = get_embedding_function()
    with open(FAISS_PATH, "rb") as f:
        db = pickle.load(f)
    db.embedding_function = embedding_function
    load_end = time.time()
    print(f"Time to load FAISS index: {load_end - load_start:.2f} seconds")

    # Similarity search
    search_start = time.time()
    results = db.similarity_search_with_score(query_text, k=4)
    search_end = time.time()
    print(f"Time for similarity search: {search_end - search_start:.2f} seconds")

    # Prepare context and prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Generate response
    generate_start = time.time()
    model = Ollama(model="gemma", temperature=temperature)
    response_text = model.invoke(prompt)
    generate_end = time.time()
    print(f"Time to generate response: {generate_end - generate_start:.2f} seconds")

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

if __name__ == "__main__":
    main()