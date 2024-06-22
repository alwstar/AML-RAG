import argparse
import os
import pickle
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain.vectorstores import FAISS

FAISS_PATH = "faiss_index"
DATA_PATH = "data"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_faiss(chunks)

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_faiss(chunks: list[Document]):
    embedding_function = get_embedding_function()
    try:
        with open(FAISS_PATH, "rb") as f:
            db = pickle.load(f)
        db.embedding_function = embedding_function
        print("Loaded existing Faiss index")
    except FileNotFoundError:
        db = FAISS.from_documents(chunks, embedding_function)
        print("Created new Faiss index")

    chunks_with_ids = calculate_chunk_ids(chunks)

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in db.docstore._dict:
            new_chunks.append(chunk)

    if new_chunks:
        print(f"👉 Adding new documents: {len(new_chunks)}")
        db.add_documents(new_chunks)
    else:
        print("✅ No new documents to add")

    with open(FAISS_PATH, "wb") as f:
        pickle.dump(db, f)

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(FAISS_PATH):
        os.remove(FAISS_PATH)
        print("✨ Cleared Faiss index")

if __name__ == "__main__":
    main()