import os
from typing import List

import chromadb
import vertexai

# import vertexai.preview.generative_models as generative_models
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from vertexai.generative_models import GenerativeModel

# Suppress logging warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Vertex AI
vertexai.init(project="dat-itowe-dev", location="europe-west4")

# Initialize Chroma DB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="pdf_collection")


# PDF processing functions
def extract_text_from_pdf(pdf_path: str) -> str:
    with open(pdf_path, "rb") as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def split_text(text: str) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(text)


# Chroma DB functions
def add_chunks_to_chroma(chunks: List[str], pdf_name: str):
    collection.add(
        documents=chunks,
        metadatas=[{"source": pdf_name} for _ in chunks],
        ids=[f"{pdf_name}_{i}" for i in range(len(chunks))],
    )


def query_chroma(query: str, n_results: int = 5) -> List[str]:
    results = collection.query(
        query_texts=[query], n_results=min(n_results, collection.count())
    )
    return results["documents"][0]


# Vertex AI functions
def get_vertex_ai_response(prompt: str) -> str:
    model = GenerativeModel("gemini-1.5-flash-001")
    response = model.generate_content(prompt)
    return response.text


# RAG implementation
def rag_query(user_query: str, pdf_name: str) -> str:
    relevant_chunks = query_chroma(user_query)
    context = "\n".join(relevant_chunks)

    prompt = f"""Based on the following context from the PDF '{pdf_name}', please answer the question.
    If the answer is not in the context,
      say 'I don't have enough information to answer that question.
    Please ask question from pdf only'

    Context: {context}

    Question: {user_query}

    Answer:"""

    return get_vertex_ai_response(prompt)


# Main application flow
def process_pdf(pdf_path: str):
    pdf_name = os.path.basename(pdf_path)
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text(text)
    add_chunks_to_chroma(chunks, pdf_name)
    print(f"Processed and added {pdf_name} to the database.")


def chat_with_pdf(pdf_name: str):
    while True:
        user_query = input("Ask a question about the PDF (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        response = rag_query(user_query, pdf_name)
        print(f"Response: {response}\n")


# Example usage
if __name__ == "__main__":
    pdf_path = "rtl_rag_chatbot_api/chatbot/gemini/file.pdf"
    process_pdf(pdf_path)
    chat_with_pdf(os.path.basename(pdf_path))
