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

Question: What is a distributed system, and how is it characterized?
Answer: A distributed system is a collection of autonomous hosts connected through a computer network.
Each host executes computations and operates a distribution middleware, enabling components to coordinate activities via message-passing so that users perceive the system as a single, integrated computing facility.
Characterization includes verifying the correctness and efficiency of protocols, dealing with concurrency, consistency, and replication.

Question: Explain the concept of clock drift in distributed systems.
Answer: Clock drift refers to the phenomenon where clocks tick at different rates.
In distributed systems, clock drift causes computer clocks to diverge over time, even if they were initially synchronized.
The clock drift rate is the relative amount that a computer clock differs from a perfect clock. Typical quartz clocks have a drift rate of about 10^-6 seconds/second, while high-precision quartz clocks can have drift rates of about 10^-8 or 10^-9 seconds/second.

​Question: What is the purpose of replication in distributed systems?
Answer: Replication in distributed systems serves two main purposes: Performance enhancement:
It can distribute the load across multiple servers, improving response times and throughput.
Fault-tolerant service: It provides correct behavior and maintains availability even if some servers fail.

Question: Explain the Bully Algorithm for leader election in distributed systems.
Answer: The Bully Algorithm is used for leader election in distributed systems. Its key points are: Assumes processes can crash, message delivery is reliable and synchronous, and processes know the IDs of all peers.
When a process P detects the need for a leader, it sends an election message to all processes with higher IDs.
- If P receives no response from higher-ID processes, it declares itself the leader and broadcasts victory.
- If P receives a response from a higher-ID process, it waits for that process to declare leadership. If no declaration is received within a timeout, P restarts the election.
- If a process receives an election message from a lower-ID process, it sends an "I am alive" message and starts a new election.
- The algorithm gets its name because higher-ID processes can "bully" lower-ID processes out of the coordinator position.
- The Bully Algorithm ensures that the process with the highest ID becomes the leader, providing a consistent way to elect a coordinator in a distributed system.

Question: What is the main difference between structured and unstructured peer-to-peer systems?
Answer: Structured peer-to-peer systems use specific algorithms and data structures to organize nodes and route queries efficiently, typically using distributed hash tables (DHTs).
Unstructured peer-to-peer systems, on the other hand, have no predetermined organization and often rely on flooding or random walks for query routing.
Structured systems are more efficient for locating rare items, while unstructured systems are simpler to implement and maintain.

------------
Context: {context}
------------

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