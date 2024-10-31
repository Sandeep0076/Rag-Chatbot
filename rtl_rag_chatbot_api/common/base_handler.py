# base_handler.py

import logging
import os
import re
from pathlib import Path
from typing import List

import chromadb
import pytesseract
from chromadb.config import Settings
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text

logging.basicConfig(level=logging.INFO)


class BaseRAGHandler:
    """Base class for RAG handlers implementing common functionality."""

    def __init__(self, configs, gcs_handler):
        self.configs = configs
        self.gcs_handler = gcs_handler
        self.max_tokens = 2000
        self.file_id = None
        self.embedding_type = None
        self.collection_name = None
        self.embedding_model = None

    def query_chroma(self, query: str, file_id: str, n_results: int = 3) -> List[str]:
        """Query the Chroma vector database for similar documents."""
        try:
            query_embedding = self.get_embeddings([query])[0]
            collection = self._get_chroma_collection()
            results = collection.query(
                query_embeddings=[query_embedding], n_results=n_results
            )
            return results["documents"][0] if results["documents"] else []
        except Exception as e:
            logging.error(f"Error in query_chroma: {str(e)}")
            raise

    def _get_chroma_collection(self):
        """Helper method to get or create ChromaDB collection."""
        chroma_db_path = f"./chroma_db/{self.file_id}/{self.embedding_type}"
        client = chromadb.PersistentClient(
            path=chroma_db_path,
            settings=Settings(
                allow_reset=True, is_persistent=True, anonymized_telemetry=False
            ),
        )
        return client.get_collection(name=self.collection_name)

    def get_n_nearest_neighbours(self, query: str, n_neighbours: int = 3) -> List[str]:
        """Get nearest neighbors for a query."""
        return self.query_chroma(query, self.file_id, n_results=n_neighbours)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Abstract method to be implemented by child classes."""
        raise NotImplementedError("Subclasses must implement get_embeddings")

    def get_answer(self, query: str) -> str:
        """Abstract method to be implemented by child classes."""
        raise NotImplementedError("Subclasses must implement get_answer")

    def create_and_store_embeddings(
        self, chunks: List[str], file_id: str, subfolder: str
    ):
        """Create embeddings and store them in Chroma DB with token limit handling."""
        try:
            chroma_db_path = f"./chroma_db/{file_id}/{subfolder}"
            os.makedirs(chroma_db_path, exist_ok=True)

            client = chromadb.PersistentClient(
                path=chroma_db_path,
                settings=Settings(allow_reset=True, is_persistent=True),
            )

            collection = client.get_or_create_collection(
                name=self.collection_name, metadata={"file_id": file_id}
            )

            # Constants for token management
            MAX_TOKENS_PER_REQUEST = 15000  # Safe limit below Gemini's 20k limit
            BATCH_SIZE = 5  # Process chunks in small batches
            processed_count = 0
            total_chunks = len(chunks)

            # Process chunks in batches
            for i in range(0, total_chunks, BATCH_SIZE):
                batch_chunks = chunks[i : i + BATCH_SIZE]
                batch_to_process = []
                batch_ids = []

                # Prepare batch with token checking
                for chunk_idx, chunk in enumerate(batch_chunks):
                    chunk_tokens = len(self.simple_tokenize(chunk))

                    if chunk_tokens > MAX_TOKENS_PER_REQUEST:
                        # Split large chunks and process individually
                        sub_chunks = self.split_large_chunk(
                            chunk, MAX_TOKENS_PER_REQUEST
                        )
                        for sub_idx, sub_chunk in enumerate(sub_chunks):
                            batch_to_process.append(sub_chunk)
                            batch_ids.append(f"{file_id}_{i + chunk_idx}_sub{sub_idx}")
                    else:
                        batch_to_process.append(chunk)
                        batch_ids.append(f"{file_id}_{i + chunk_idx}")

                if batch_to_process:
                    try:
                        # Get embeddings for the batch
                        embeddings = self.get_embeddings(batch_to_process)

                        # Add to collection
                        collection.add(
                            documents=batch_to_process,
                            embeddings=embeddings,
                            metadatas=[{"source": file_id} for _ in batch_to_process],
                            ids=batch_ids,
                        )

                        processed_count += len(batch_to_process)
                        logging.info(
                            f"Processed {processed_count}/{total_chunks} chunks"
                        )

                    except Exception as e:
                        logging.error(
                            f"Error processing batch starting at chunk {i}: {str(e)}"
                        )
                        # Continue with next batch instead of failing completely
                        continue

            logging.info(f"Successfully created and stored embeddings for {file_id}")
            return "completed"

        except Exception as e:
            logging.error(f"Error in create_and_store_embeddings: {str(e)}")
            raise

    def process_file(
        self,
        file_id: str,
        decrypted_file_path: str,
        subfolder: str,
        collection_name: str = None,
    ):
        """Process a file by extracting text and creating embeddings."""
        text = self.extract_text_from_file(decrypted_file_path)
        self.collection_name = collection_name or f"rag_collection_{file_id}"
        chunks = self.split_text(text)
        self.create_and_store_embeddings(chunks, file_id, subfolder)
        logging.info(f"{self.collection_name} collection is being used")

    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file types."""
        _, file_extension = os.path.splitext(file_path)

        if file_extension.lower() == ".pdf":
            return self.extract_text_from_pdf(file_path)
        else:
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    return file.read()
            except UnicodeDecodeError:
                with open(file_path, "rb") as file:
                    return file.read().decode("utf-8", errors="ignore")

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using pdfminer or OCR if necessary."""
        try:
            logging.info(f"Attempting to extract text from PDF: {file_path}")
            text = extract_text(file_path)
            word_count = len(text.split())

            if word_count == 0:
                logging.info("pdfminer failed to extract text. Attempting OCR...")
                images = convert_from_path(file_path)
                text = ""
                for i, image in enumerate(images):
                    logging.info(f"Processing page {i+1}")
                    text += pytesseract.image_to_string(image)
                word_count = len(text.split())

            logging.info(
                f"Text extraction completed successfully. Word count: {word_count}"
            )
            return text
        except Exception as e:
            logging.error(f"Error in extract_text_from_pdf: {str(e)}")
            raise

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks based on token limits."""
        chunks = []
        paragraphs = text.split("\n\n")
        current_chunk = []
        current_chunk_tokens = 0

        for paragraph in paragraphs:
            paragraph_tokens = len(self.simple_tokenize(paragraph))

            if paragraph_tokens > self.max_tokens:
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
        """Simple word tokenization."""
        return re.findall(r"\b\w+\b", text.lower())

    def split_large_chunk(self, chunk: str, max_tokens: int) -> List[str]:
        """Split a large chunk into smaller pieces."""
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

    def upload_embeddings_to_gcs(self, file_id: str, subfolder: str):
        """Upload embeddings to Google Cloud Storage."""
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
