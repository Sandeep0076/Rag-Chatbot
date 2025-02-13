# base_handler.py
import logging
import os
import re
import uuid
from pathlib import Path
from typing import List

import pytesseract
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text

from rtl_rag_chatbot_api.common.chroma_manager import ChromaDBManager

logging.basicConfig(level=logging.INFO)


class BaseRAGHandler:
    """Base class for RAG handlers implementing common functionality."""

    def __init__(self, configs, gcs_handler):
        self.configs = configs
        self.gcs_handler = gcs_handler
        # Use chunk size from config instead of hardcoded value
        self.max_tokens = configs.chatbot.chunk_size_limit
        self.chunk_overlap = configs.chatbot.max_chunk_overlap
        self.file_id = None
        self.embedding_type = None
        self.user_id = None
        self.collection_name = None
        self.embedding_model = None
        self.chroma_manager = ChromaDBManager()

        # Model-specific token limits for response generation (not chunking)
        self.MODEL_TOKEN_LIMITS = {
            "gpt_3_5_turbo": 8000,
            "gpt_4": 32000,
            "gpt_4o_mini": 128000,
            "gpt_4_omni": 128000,
            "gemini-pro": 30500,
            "gemini-flash": 15000,
        }

        self.AZURE_MAX_TOKENS = self.MODEL_TOKEN_LIMITS.get(
            "gpt_3_5_turbo", 8000
        )  # Use GPT-3.5 limit for Azure
        self.GEMINI_MAX_TOKENS = self.MODEL_TOKEN_LIMITS.get(
            "gemini-flash", 15000
        )  # Use Gemini Flash limit
        self.BATCH_SIZE = 5

    def query_chroma(self, query: str, file_id: str, n_results: int = 3) -> List[str]:
        """Query the Chroma vector database for similar documents."""
        try:
            query_embedding = self.get_embeddings([query])[0]
            collection = self._get_chroma_collection()

            # Don't filter by user_id when querying - embeddings are shared
            results = collection.query(
                query_embeddings=[query_embedding], n_results=n_results
            )
            return results["documents"][0] if results["documents"] else []
        except Exception as e:
            logging.error(f"Error in query_chroma: {str(e)}")
            raise

    def _get_chroma_collection(self):
        """Helper method to get or create ChromaDB collection using the manager."""
        if not all([self.file_id, self.embedding_type, self.collection_name]):
            raise ValueError("file_id, embedding_type, and collection_name must be set")

        # Pass user_id but don't filter queries - for session tracking only
        return self.chroma_manager.get_collection(
            self.file_id,
            self.embedding_type,
            self.collection_name,
            self.user_id,
            is_embedding=False,
        )

    def get_n_nearest_neighbours(self, query: str, n_neighbours: int = 3) -> List[str]:
        """Get nearest neighbors for a query."""
        try:
            query_embedding = self.get_embeddings([query])[0]
            # Use base collection without user filtering for nearest neighbors
            collection = self.chroma_manager.get_collection(
                self.file_id,
                self.embedding_type,
                self.collection_name,
                user_id=None,  # No user filtering for nearest neighbors
                is_embedding=False,
            )

            results = collection.query(
                query_embeddings=[query_embedding], n_results=n_neighbours
            )
            return results["documents"][0] if results["documents"] else []
        except Exception as e:
            logging.error(f"Error getting nearest neighbors: {str(e)}")
            raise

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Abstract method to be implemented by child classes."""
        raise NotImplementedError("Subclasses must implement get_embeddings")

    def get_answer(self, query: str) -> str:
        """Abstract method to be implemented by child classes."""
        raise NotImplementedError("Subclasses must implement get_answer")

    def create_and_store_embeddings(
        self, chunks: List[str], file_id: str, subfolder: str, is_embedding: bool = True
    ) -> str:
        """
        Create embeddings for text chunks and store them in ChromaDB with batch processing.

        Processes text chunks into embeddings while handling token limits and batch sizes.
        Uses configured chunk size (chunk_size_limit) for optimal RAG performance.
        Implements automatic chunk splitting for oversized text segments.

        Args:
            chunks (List[str]): List of text chunks to be embedded
            file_id (str): Unique identifier for the file being processed
            subfolder (str): Storage subfolder ('azure' or 'google') determining model type

        Returns:
            str: Status string ('completed' on success)
        """
        try:
            # Get collection with explicit subfolder path
            collection = self.chroma_manager.get_collection(
                file_id=file_id,
                embedding_type=subfolder,  # 'azure' or 'google'
                collection_name=self.collection_name,
                user_id=None,  # No user filtering for embeddings creation
                is_embedding=is_embedding,
            )

            processed_count = 0
            total_chunks = len(chunks)
            batch_to_process = []
            batch_ids = []
            embedding_id = str(
                uuid.uuid4()
            )  # Generate a unique ID for embeddings folder

            # Use configured chunk size for optimal RAG performance
            max_chunk_size = self.max_tokens

            for i in range(0, total_chunks):
                chunk = chunks[i]
                chunk_tokens = len(self.simple_tokenize(chunk))

                if chunk_tokens > max_chunk_size:
                    sub_chunks = self.split_large_chunk(chunk, max_chunk_size)
                    for sub_idx, sub_chunk in enumerate(sub_chunks):
                        sub_chunk_tokens = len(self.simple_tokenize(sub_chunk))
                        if sub_chunk_tokens <= max_chunk_size:
                            batch_to_process.append(sub_chunk)
                            batch_ids.append(f"{embedding_id}_{i}_sub{sub_idx}")
                else:
                    batch_to_process.append(chunk)
                    batch_ids.append(f"{embedding_id}_{i}")

                if len(batch_to_process) >= self.BATCH_SIZE or i == total_chunks - 1:
                    if batch_to_process:
                        try:
                            embeddings = self.get_embeddings(batch_to_process)
                            collection.add(
                                documents=batch_to_process,
                                embeddings=embeddings,
                                metadatas=[
                                    {"source": file_id, "embedding_id": embedding_id}
                                    for _ in batch_to_process
                                ],
                                ids=batch_ids,
                            )
                            processed_count += len(batch_to_process)
                            logging.info(
                                f"Processed {processed_count}/{total_chunks} chunks"
                            )
                        except Exception as e:
                            logging.error(
                                f"Error processing batch at chunk {i}: {str(e)}"
                            )
                        finally:
                            batch_to_process = []
                            batch_ids = []

            # Store the embedding_id for reference
            self.embedding_id = embedding_id
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
        self.create_and_store_embeddings(chunks, file_id, subfolder, is_embedding=True)
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
                try:
                    images = convert_from_path(file_path)
                    text = ""
                    for i, image in enumerate(images):
                        logging.info(f"Processing page {i+1}")
                        text += pytesseract.image_to_string(image)
                    word_count = len(text.split())

                    if word_count == 0:
                        return (
                            "ERROR: Unable to extract text from this PDF. The file might be scanned,"
                            "corrupted, or password-protected. "
                            "Please try a different PDF file."
                        )

                except Exception as ocr_error:
                    logging.error(f"OCR extraction failed: {str(ocr_error)}")
                    return (
                        "ERROR: Unable to process this PDF. The file might be corrupted or in an unsupported format. "
                        "Please try a different PDF file."
                    )

            logging.info(
                f"Text extraction completed successfully. Word count: {word_count}"
            )
            return text
        except Exception as e:
            logging.error(f"Error in extract_text_from_pdf: {str(e)}")
            return "ERROR: Unable to read this PDF file. Please ensure it's a valid PDF and try again."

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks based on token limits."""
        chunks = []
        paragraphs = text.split("\n\n")
        current_chunk = []
        current_chunk_tokens = 0

        # Get appropriate token limit based on model type
        if self.embedding_type == "azure":
            max_tokens = self.AZURE_MAX_TOKENS
        elif self.embedding_type == "google":
            max_tokens = self.GEMINI_MAX_TOKENS
        else:
            max_tokens = self.max_tokens

        for paragraph in paragraphs:
            paragraph_tokens = len(self.simple_tokenize(paragraph))

            if paragraph_tokens > max_tokens:
                # Split large paragraphs into smaller chunks
                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                for sentence in sentences:
                    sentence_tokens = len(self.simple_tokenize(sentence))
                    if current_chunk_tokens + sentence_tokens > max_tokens:
                        if current_chunk:
                            chunks.append(" ".join(current_chunk))
                            current_chunk = []
                            current_chunk_tokens = 0
                        # If single sentence exceeds limit, split it further
                        if sentence_tokens > max_tokens:
                            sentence_chunks = self.split_large_chunk(
                                sentence, max_tokens
                            )
                            chunks.extend(sentence_chunks)
                        else:
                            current_chunk.append(sentence)
                            current_chunk_tokens = sentence_tokens
                    else:
                        current_chunk.append(sentence)
                        current_chunk_tokens += sentence_tokens
            elif current_chunk_tokens + paragraph_tokens > max_tokens:
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
        """Split a large chunk into smaller pieces based on token limit."""
        words = chunk.split()
        current_chunk = []
        chunks = []
        current_length = 0
        overlap_tokens = int(max_tokens * self.chunk_overlap)  # Use configured overlap

        for word in words:
            word_length = len(self.simple_tokenize(word))
            if current_length + word_length > max_tokens:
                if current_chunk:  # Only add if we have something
                    chunks.append(" ".join(current_chunk))
                    # Keep overlap_tokens worth of words for context
                    current_chunk = (
                        current_chunk[-overlap_tokens:] if overlap_tokens > 0 else []
                    )
                    current_length = sum(
                        len(self.simple_tokenize(w)) for w in current_chunk
                    )
            current_chunk.append(word)
            current_length += word_length

        if current_chunk:  # Add the last chunk if it exists
            chunks.append(" ".join(current_chunk))

        return chunks

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

    def get_model_token_limit(self, model_name: str) -> int:
        """Get the token limit for a specific model."""
        return self.MODEL_TOKEN_LIMITS.get(model_name.lower(), 2000)

    def ensure_token_limit(self, text: str, model_name: str) -> List[str]:
        """
        Ensure text chunks don't exceed model's token limit.
        Returns list of chunks that fit within the model's limit.
        """
        model_limit = self.get_model_token_limit(model_name)
        tokens = self.simple_tokenize(text)

        if len(tokens) <= model_limit:
            return [text]

        chunks = []
        current_chunk = []
        current_tokens = 0

        for token in tokens:
            if current_tokens + 1 > model_limit:
                chunks.append(" ".join(current_chunk))
                current_chunk = [token]
                current_tokens = 1
            else:
                current_chunk.append(token)
                current_tokens += 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
