import logging
import os
import re
from pathlib import Path
from typing import List

import chromadb
import pytesseract
import vertexai
from chromadb.config import Settings
from pdf2image import convert_from_path

# import PyPDF2
from pdfminer.high_level import extract_text
from vertexai.generative_models import GenerativeModel
from vertexai.preview.generative_models import (
    GenerationConfig,
    HarmBlockThreshold,
    HarmCategory,
)
from vertexai.preview.language_models import TextEmbeddingModel

logging.basicConfig(level=logging.INFO)


class GeminiHandler:
    """
    Handles interactions with Google's Gemini AI models for text processing and embedding generation.

    This class provides methods for initializing Gemini models, processing files, generating embeddings,
    and interacting with a Chroma vector database for efficient querying.

    Attributes:
        configs: Configuration object containing settings for Gemini and other services.
        gcs_handler: Google Cloud Storage handler for file operations.
        embedding_model: Text embedding model from Gemini.
        generative_model: Generative AI model from Gemini.
        chroma_client: Client for interacting with the Chroma vector database.
        max_tokens (int): Maximum number of tokens for text chunks.
        file_id (str): Identifier for the current file being processed.
        embedding_type (str): Type of embedding being used.
    """

    def __init__(self, configs, gcs_handler):
        """
        Initializes the GeminiHandler with configurations and GCS handler.

        Args:
            configs: Configuration object containing settings for Gemini and other services.
            gcs_handler: Google Cloud Storage handler for file operations.
        """
        self.configs = configs
        self.gcs_handler = gcs_handler
        vertexai.init(project=configs.gemini.project, location=configs.gemini.location)
        self.embedding_model = TextEmbeddingModel.from_pretrained(
            "textembedding-gecko@latest"
        )
        self.generative_model = None
        self.chroma_client = None
        self.max_tokens = 2000
        self.file_id = None
        self.embedding_type = None

    def initialize(self, model: str, file_id: str, embedding_type: str):
        # Initializes the Gemini model with specific configurations.
        generation_config = GenerationConfig(
            temperature=0.9,
            top_p=1,
            top_k=40,
            max_output_tokens=2048,
        )
        # Setting it to BLOCK_NONE because it is blocking and returning none in chat
        # for simple cases like if a girl is angry with a boy.
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }

        self.generative_model = GenerativeModel(
            model, generation_config=generation_config, safety_settings=safety_settings
        )
        self.file_id = file_id
        self.embedding_type = embedding_type

    def get_answer(self, query: str) -> str:
        # Generates an answer to a given query using relevant context.
        try:
            relevant_chunks = self.query_chroma(query, self.file_id, n_results=3)
            if not relevant_chunks:
                return (
                    "I couldn't find any relevant information to answer your question."
                )

            context = "\n".join(relevant_chunks)
            prompt = f"""Based on the following context, please answer the question.
            If the answer is not in the context, say 'I don't have enough information
              to answer that question from the uploaded document. Please rephrase or ask another question.'

            Context: {context}

            Question: {query}

            Answer:"""
            response = self.generative_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logging.error(f"Error in get_answer: {str(e)}")
            return "I'm sorry, but I encountered an error while trying to answer your question. Please try again."

    def process_file(
        self, file_id: str, decrypted_file_path: str, subfolder: str = "google"
    ):
        # Processes a file by extracting text, splitting it, and creating embeddings.
        text = self.extract_text_from_file(decrypted_file_path)
        chunks = self.split_text(text)
        self.create_and_store_embeddings(chunks, file_id, subfolder)

    def extract_text_from_file(self, file_path: str) -> str:
        #  Extracts text content from various file types.
        _, file_extension = os.path.splitext(file_path)

        if file_extension.lower() == ".pdf":
            return self.extract_text_from_pdf(file_path)
        else:
            # For other file types, attempt to read as text
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    return file.read()
            except UnicodeDecodeError:
                # If UTF-8 decoding fails, try reading as binary
                with open(file_path, "rb") as file:
                    return file.read().decode("utf-8", errors="ignore")

    def extract_text_from_pdf(self, file_path: str) -> str:
        # Extracts text from PDF files using pdfminer or OCR if necessary.
        try:
            logging.info(f"Attempting to extract text from PDF: {file_path}")

            # Try pdfminer first
            text = extract_text(file_path)
            word_count = len(text.split())

            # If pdfminer fails to extract text, use OCR
            if word_count == 0:
                logging.info("pdfminer failed to extract text. Attempting OCR...")

                # Convert PDF to images
                images = convert_from_path(file_path)

                # Perform OCR on each image
                text = ""
                for i, image in enumerate(images):
                    logging.info(f"Processing page {i+1}")
                    text += pytesseract.image_to_string(image)

                word_count = len(text.split())

            logging.info("Text extraction completed successfully")
            logging.info(f"Total words in the pdf are : {word_count}")
            return text
        except Exception as e:
            logging.error(f"Error in extract_text_from_pdf: {str(e)}")
            raise

    def extract_text_from_pdf2(self, file_path: str) -> str:
        """
        Extracts text from a PDF file using pdfminer or OCR if necessary.

        Args:
            file_path (str): The path to the PDF file.

        Returns:
            str: The extracted text content from the PDF.

        Raises:
            Exception: If an error occurs during the extraction process.
        """
        try:
            logging.info(f"Attempting to extract text from PDF: {file_path}")
            text = extract_text(file_path)
            logging.info("Text extraction completed successfully")
            word_count = len(text.split())
            logging.info(f"Total words in the pdf are : {word_count}")
            return text
        except Exception as e:
            logging.error(f"Error in extract_text_from_pdf: {str(e)}")
            raise

    def split_text(self, text: str) -> List[str]:
        """
        Splits the input text into chunks based on token limits.

        Args:
            text (str): The input text to be split into chunks.

        Returns:
            List[str]: A list of text chunks based on token limits.
        """
        chunks = []
        paragraphs = text.split("\n\n")
        current_chunk = []
        current_chunk_tokens = 0

        for paragraph in paragraphs:
            paragraph_tokens = len(self.simple_tokenize(paragraph))

            if paragraph_tokens > self.max_tokens:
                # If a single paragraph is too long, split it further
                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                for sentence in sentences:
                    sentence_tokens = len(self.simple_tokenize(sentence))
                    if current_chunk_tokens + sentence_tokens > self.max_tokens:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = []
                        current_chunk_tokens = 0
                    current_chunk.append(sentence)
                    current_chunk_tokens += sentence_tokens
            elif current_chunk_tokens + paragraph_tokens > self.max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = [paragraph]
                current_chunk_tokens = paragraph_tokens
            else:
                current_chunk.append(paragraph)
                current_chunk_tokens += paragraph_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def simple_tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    # deprecated
    def add_chunks_to_chroma(self, chunks: List[str], file_id: str):
        collection = self.chroma_client.get_or_create_collection(
            name=self.configs.chatbot.vector_db_collection_name
        )
        embeddings = self.get_embeddings(chunks)
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"source": file_id} for _ in chunks],
            ids=[f"{file_id}_{i}" for i in range(len(chunks))],
        )

    def query_chroma(self, query: str, file_id: str, n_results: int = 3) -> List[str]:
        """
        Queries the Chroma vector database for similar documents based on the input query.

        Args:
            query (str): The query string for similarity search.
            file_id (str): The identifier of the file associated with the query.
            n_results (int): The number of similar documents to retrieve (default is 3).

        Returns:
            List[str]: A list of text documents similar to the query.
        """
        chroma_db_path = f"./chroma_db/{file_id}/{self.embedding_type}"
        client = chromadb.PersistentClient(
            path=chroma_db_path, settings=Settings(allow_reset=True, is_persistent=True)
        )
        collection = client.get_collection(
            name=self.configs.chatbot.vector_db_collection_name
        )

        # Get the embedding for the query
        query_embedding = self.embedding_model.get_embeddings([query])[0].values

        # Query the collection
        results = collection.query(
            query_embeddings=[query_embedding], n_results=n_results
        )

        # Extract and return the text of the nearest neighbors
        documents = results["documents"][0] if results["documents"] else []
        return documents[:n_results]  # Return up to n_results documents

    # Retrieves the nearest neighbors for a given query.
    def get_n_nearest_neighbours(
        self, query: str, file_id: str, n_neighbors: int = 3
    ) -> List[str]:
        return self.query_chroma(query, file_id, n_results=n_neighbors)

    # Generates a response using the Google model.
    def get_gemini_response(self, prompt: str) -> str:
        model = GenerativeModel(self.configs.gemini.model_pro)
        response = model.generate_content(prompt)
        return response.text

    #  Generates embeddings for given texts.
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = self.embedding_model.get_embeddings(texts)
            return [embedding.values for embedding in embeddings]
        except Exception as e:
            logging.error(f"Error getting embeddings: {str(e)}")
            # Log the size of the problematic texts
            logging.error(
                f"Number of texts: {len(texts)}, Total tokens: {sum(len(self.simple_tokenize(text)) for text in texts)}"
            )
            raise

    #  Uploads generated embeddings to Google Cloud Storage.
    def upload_embeddings_to_gcs(self, file_id: str, subfolder: str = "google"):
        chroma_db_path = f"./chroma_db/{file_id}/{subfolder}"
        gcs_subfolder = f"file-embeddings/{file_id}/{subfolder}"

        files_to_upload = {}
        for file in Path(chroma_db_path).rglob("*"):
            if file.is_file():
                relative_path = file.relative_to(chroma_db_path)
                gcs_object_name = f"{gcs_subfolder}/{relative_path}"
                files_to_upload[str(relative_path)] = (str(file), gcs_object_name)

        self.gcs_handler.upload_to_gcs(
            self.configs.gcp_resource.bucket_name, files_to_upload
        )

    #  Creates embeddings for text chunks and stores them in the Chroma database.
    def create_and_store_embeddings(
        self, chunks: List[str], file_id: str, subfolder: str = "google"
    ):
        chroma_db_path = f"./chroma_db/{file_id}/{subfolder}"
        os.makedirs(chroma_db_path, exist_ok=True)

        client = chromadb.PersistentClient(
            path=chroma_db_path, settings=Settings(allow_reset=True, is_persistent=True)
        )
        collection = client.get_or_create_collection(
            name=self.configs.chatbot.vector_db_collection_name
        )

        max_tokens_per_request = (
            15000  # Set a safe limit below the 20,000 token maximum
        )
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            try:
                chunk_tokens = len(self.simple_tokenize(chunk))
                if chunk_tokens > max_tokens_per_request:
                    # If a single chunk is too large, split it further
                    sub_chunks = self.split_large_chunk(chunk, max_tokens_per_request)
                    for j, sub_chunk in enumerate(sub_chunks):
                        embedding = self.get_embeddings([sub_chunk])[0]
                        collection.add(
                            documents=[sub_chunk],
                            embeddings=[embedding],
                            metadatas=[{"source": file_id}],
                            ids=[f"{file_id}_{i}_{j}"],
                        )
                else:
                    embedding = self.get_embeddings([chunk])[0]
                    collection.add(
                        documents=[chunk],
                        embeddings=[embedding],
                        metadatas=[{"source": file_id}],
                        ids=[f"{file_id}_{i}"],
                    )

                if (i + 1) % 10 == 0:
                    logging.info(
                        f"Processed {i + 1} of {total_chunks} chunks for file_id: {file_id}"
                    )
            except Exception as e:
                logging.error(f"Error processing chunk {i + 1}: {str(e)}")
                logging.error(
                    f"Chunk size: {len(chunk)}, Tokens: {len(self.simple_tokenize(chunk))}"
                )

        logging.info(
            f"Completed processing {total_chunks} chunks for file_id: {file_id}"
        )

    # Splits a large chunk of text into smaller, manageable pieces.
    def split_large_chunk(self, chunk: str, max_tokens: int) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", chunk)
        sub_chunks = []
        current_sub_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(self.simple_tokenize(sentence))
            if current_tokens + sentence_tokens > max_tokens:
                sub_chunks.append(" ".join(current_sub_chunk))
                current_sub_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_sub_chunk.append(sentence)
                current_tokens += sentence_tokens

        if current_sub_chunk:
            sub_chunks.append(" ".join(current_sub_chunk))

        return sub_chunks
