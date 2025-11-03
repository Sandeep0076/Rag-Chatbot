# base_handler.py
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Lock, Semaphore
from typing import List

import pytesseract
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text

from rtl_rag_chatbot_api.common.chroma_manager import ChromaDBManager
from rtl_rag_chatbot_api.common.errors import (
    BaseAppError,
    DocTextTooShortError,
    DocTextValidationError,
    PdfTextExtractionError,
    TxtExtractionError,
)
from rtl_rag_chatbot_api.common.text_extractor import TextExtractor

logging.basicConfig(level=logging.INFO)


class RateLimiter:
    """Thread-safe rate limiter for API requests."""

    def __init__(self, max_requests_per_second: int):
        self.max_requests_per_second = max_requests_per_second
        self.semaphore = Semaphore(max_requests_per_second)
        self.request_times = Queue(maxsize=max_requests_per_second)
        self.lock = Lock()

    def acquire(self):
        """Acquire a permit to make a request."""
        with self.lock:
            now = time.time()

            # Clean up old request times
            while not self.request_times.empty():
                try:
                    oldest_time = self.request_times.queue[0]
                    if now - oldest_time > 1.0:
                        self.request_times.get_nowait()
                    else:
                        break
                except Exception as e:
                    logging.error(f"Error in RateLimiter: {str(e)}")
                    break

            # If we've hit our limit, wait until we can make another request
            if self.request_times.full():
                oldest_time = self.request_times.queue[0]
                wait_time = max(0, 1.0 - (now - oldest_time))
                if wait_time > 0:
                    time.sleep(wait_time)
                    now = time.time()

            # Acquire the semaphore and record the request time
            self.semaphore.acquire()
            if self.request_times.full():
                self.request_times.get_nowait()  # Make room if needed
            self.request_times.put(now)

    def release(self):
        """Release a permit after request completion."""
        self.semaphore.release()


class BaseRAGHandler:
    """Base class for RAG handlers implementing common functionality."""

    def __init__(self, configs, gcs_handler, chroma_manager=None):
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
        # Use the provided chroma_manager or create a new one
        self.chroma_manager = chroma_manager or ChromaDBManager()

        # Model-specific token limits for response generation (not chunking)
        self.model_token_limits = {
            "gpt_4o": 128000,
            "gpt_4o_mini": 128000,
            "gpt_4_omni": 128000,
            "gpt_4_1": 200000,  # GPT-4.1 token limit
            "gpt_4_1_nano": 50000,  # GPT-4.1-nano token limit
            "gpt_5": 200000,  # GPT-5 token limit
            "gpt_5_mini": 100000,  # GPT-5-mini token limit
            "o3": 100000,  # O3 token limit
            "o4_mini": 100000,  # O4-mini token limit
            "gemini-2.5-pro": 2097152,  # 2M tokens for Gemini 2.5 Pro
            "gemini-2.5-flash": 1048576,  # 1M tokens for Gemini 2.5 Flash
        }

        self.AZURE_MAX_TOKENS = self.model_token_limits.get(
            "gpt_4o_mini",
            128000,  # Default to gpt_4o_mini's limit for Azure if not specified elsewhere
        )
        self.GEMINI_MAX_TOKENS = self.model_token_limits.get(
            "gemini-2.5-flash", 15000
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

    def _process_chunk_batch(
        self, chunks: List[str], rate_limiter: RateLimiter
    ) -> List[List[float]]:
        """Process a batch of chunks to get their embeddings with rate limiting."""
        embeddings = []
        for chunk in chunks:
            rate_limiter.acquire()
            try:
                embedding = self.get_embeddings([chunk])[0]
                embeddings.append(embedding)
            finally:
                rate_limiter.release()
        return embeddings

    def get_answer(self, query: str) -> str:
        """Abstract method to be implemented by child classes."""
        raise NotImplementedError("Subclasses must implement get_answer")

    def create_and_store_embeddings(
        self, chunks: List[str], file_id: str, subfolder: str, is_embedding: bool = True
    ) -> str:
        """
        Create embeddings for text chunks and store them in ChromaDB with parallel processing.

        Uses thread-based parallelism with rate limiting:
        - Processes up to 5 batches in parallel
        - Each batch contains up to 5 chunks (configurable via self.BATCH_SIZE)
        - Respects Azure's 8000 token limit per request
        - Maintains rate limit of 10 requests per second
        - Uses thread pool for parallel processing
        - Maintains ChromaDB consistency with sequential updates

        Args:
            chunks (List[str]): List of text chunks to be embedded
            file_id (str): Unique identifier for the file being processed
            subfolder (str): Storage subfolder for embeddings storage
            is_embedding (bool): Flag for embedding creation mode

        Returns:
            str: Status string ('completed' on success)
        """
        try:
            # Get collection and initialize tracking
            collection = self.chroma_manager.get_collection(
                file_id=file_id,
                embedding_type=subfolder,
                collection_name=self.collection_name,
                user_id=None,
                is_embedding=is_embedding,
            )

            # Initialize resources and constants
            rate_limiter = RateLimiter(10)  # 10 requests per second
            AZURE_EMBEDDING_LIMIT = 8000  # Azure's token limit per request
            MAX_WORKERS = 3  # Number of parallel threads
            processed_count = 0
            embedding_id = str(uuid.uuid4())

            # Prepare chunks for processing
            prepared_chunks = []
            chunk_ids = []

            # Pre-process chunks to handle token limits
            for i, chunk in enumerate(chunks):
                chunk_tokens = len(self.simple_tokenize(chunk))
                if chunk_tokens > AZURE_EMBEDDING_LIMIT:
                    # Split large chunks into smaller ones
                    smaller_chunks = self.split_large_chunk(
                        chunk, AZURE_EMBEDDING_LIMIT - 100
                    )
                    for j, small_chunk in enumerate(smaller_chunks):
                        prepared_chunks.append(small_chunk)
                        chunk_ids.append(f"{embedding_id}_{i}_split_{j}")
                else:
                    prepared_chunks.append(chunk)
                    chunk_ids.append(f"{embedding_id}_{i}")

            logging.info(f"Prepared {len(prepared_chunks)} chunks for processing")

            # Process chunks in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = []

                # Submit chunks in batches
                for i in range(0, len(prepared_chunks), self.BATCH_SIZE):
                    batch_end = min(i + self.BATCH_SIZE, len(prepared_chunks))
                    batch = prepared_chunks[i:batch_end]
                    batch_ids = chunk_ids[i:batch_end]

                    # Submit the batch for processing
                    future = executor.submit(
                        self._process_chunk_batch, batch, rate_limiter
                    )
                    futures.append((future, batch, batch_ids))

                # Process results as they complete
                for future, batch, ids in futures:
                    try:
                        embeddings = future.result()

                        # Add to ChromaDB (this remains sequential for consistency)
                        collection.add(
                            documents=batch,
                            embeddings=embeddings,
                            metadatas=[
                                {"source": file_id, "embedding_id": embedding_id}
                                for _ in batch
                            ],
                            ids=ids,
                        )

                        processed_count += len(batch)
                        logging.info(
                            f"Processed {processed_count}/{len(prepared_chunks)} chunks"
                        )
                    except Exception as e:
                        logging.error(f"Error processing batch: {str(e)}")
                        raise

            self.embedding_id = embedding_id
            logging.info(f"Successfully created embeddings for {file_id}")
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
        """Extract text from various file types including PDF, TXT, DOC, and DOCX."""
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()

        logging.info(
            f"Extracting text from file: {file_path} with extension {file_extension}"
        )

        try:
            if file_extension == ".pdf":
                text = self.extract_text_from_pdf(file_path)
            elif file_extension in [".doc", ".docx"]:
                extractor = TextExtractor(self.configs)
                text = extractor.extract_text_from_doc(file_path)
            elif file_extension == ".txt":
                text = self.extract_text_from_txt(file_path)
            else:
                logging.warning(
                    f"Unsupported file type: {file_extension}. Attempting generic text extraction."
                )
                text = self.extract_text_from_txt(file_path)

            # Save extracted text for diagnosis
            self._save_extracted_text_for_diagnosis(file_path, text)

            # Check if extracted text is too short
            if len(text.strip()) < 100:
                raise DocTextTooShortError(
                    "Unable to extract sufficient text from this file (less than 100 characters). "
                    "Please try using the 'Chat with Image' feature instead.",
                    details={"file_path": file_path, "text_length": len(text.strip())},
                )

            return text

        except BaseAppError:
            # Preserve specific app error codes (e.g., 2010 for short text)
            raise
        except Exception as e:
            error_msg = f"Error extracting text from {file_path}: {str(e)}"
            logging.error(error_msg)
            raise DocTextValidationError(error_msg, details={"file_path": file_path})

    def _save_extracted_text_for_diagnosis(self, file_path: str, extracted_text: str):
        """Save extracted text to a diagnostic file for testing purposes."""
        # Check if diagnostic saving is enabled
        if not getattr(self.configs, "save_extracted_text_diagnostic", False):
            return

        try:
            # Create diagnostic directory if it doesn't exist
            diagnostic_dir = "diagnostic_extracted_texts"
            os.makedirs(diagnostic_dir, exist_ok=True)

            # Generate filename based on original file
            original_filename = os.path.basename(file_path)
            base_name = os.path.splitext(original_filename)[0]
            timestamp = int(time.time())
            diagnostic_filename = f"{base_name}_extracted_text_{timestamp}.txt"
            diagnostic_path = os.path.join(diagnostic_dir, diagnostic_filename)

            # Save the extracted text
            with open(diagnostic_path, "w", encoding="utf-8") as f:
                f.write("=== EXTRACTED TEXT DIAGNOSTIC FILE ===\n")
                f.write(f"Original file: {file_path}\n")
                f.write(f"Extraction timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Text length: {len(extracted_text)} characters\n")
                f.write(f"Word count: {len(extracted_text.split())} words\n")
                f.write(f"Line count: {len(extracted_text.splitlines())} lines\n")
                f.write(f"Contains markdown headers: {'#' in extracted_text}\n")
                f.write(
                    f"Contains markdown lists: {'* ' in extracted_text or '- ' in extracted_text}\n"
                )
                f.write(f"Contains markdown bold: {'**' in extracted_text}\n")
                f.write(f"Contains markdown italic: {'*' in extracted_text}\n")
                f.write(f"Contains markdown code blocks: {'```' in extracted_text}\n")
                f.write(
                    f"Contains markdown links: {'[' in extracted_text and '](' in extracted_text}\n"
                )
                f.write(f"Contains tables: {'|' in extracted_text}\n")
                newline_check = "\\n" in repr(extracted_text)
                tab_check = "\\t" in repr(extracted_text)
                f.write(f"Contains newlines: {newline_check}\n")
                f.write(f"Contains tabs: {tab_check}\n")
                f.write(f"First 200 characters: {repr(extracted_text[:200])}\n")
                f.write(f"Last 200 characters: {repr(extracted_text[-200:])}\n")
                f.write(f"\n{'=' * 50}\n")
                f.write("FULL EXTRACTED TEXT:\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(extracted_text)
                f.write(f"\n\n{'=' * 50}\n")
                f.write("END OF EXTRACTED TEXT\n")
                f.write(f"{'=' * 50}\n")

            logging.info(f"Saved extracted text diagnostic file: {diagnostic_path}")

        except Exception as e:
            logging.warning(f"Failed to save extracted text diagnostic file: {str(e)}")

    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT files with proper encoding handling."""
        logging.info(f"Extracting text from TXT file: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                logging.info(
                    f"Successfully extracted {len(text)} characters from TXT file"
                )
                return text
        except UnicodeDecodeError:
            logging.warning(
                "UTF-8 decoding failed, attempting with 'utf-8' and ignore errors"
            )
            try:
                with open(file_path, "rb") as file:
                    text = file.read().decode("utf-8", errors="ignore")
                    logging.info(
                        f"Successfully extracted {len(text)} characters from TXT file using fallback encoding"
                    )
                    return text
            except Exception as e:
                raise TxtExtractionError(
                    f"Failed to extract text from TXT file: {str(e)}",
                    details={"file_path": file_path},
                )
        except Exception as e:
            raise TxtExtractionError(
                f"Failed to read TXT file: {str(e)}", details={"file_path": file_path}
            )

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
                        logging.info(f"Processing page {i + 1}")
                        text += pytesseract.image_to_string(image)
                    word_count = len(text.split())

                    if word_count == 0:
                        error_msg = (
                            "Unable to extract text from this PDF. "
                            "The file might be scanned, corrupted, or password-protected."
                        )
                        logging.error(error_msg)
                        raise PdfTextExtractionError(
                            error_msg, details={"file_path": file_path, "method": "ocr"}
                        )

                except Exception as ocr_error:
                    error_msg = f"OCR extraction failed: {str(ocr_error)}"
                    logging.error(error_msg)
                    raise PdfTextExtractionError(
                        error_msg, details={"file_path": file_path, "method": "ocr"}
                    )

            logging.info(
                f"Text extraction completed successfully. Word count: {word_count}"
            )
            return text
        except Exception as e:
            logging.error(f"Error in extract_text_from_pdf: {str(e)}")
            raise PdfTextExtractionError(
                "Unable to read this PDF file. Please ensure it's a valid PDF and try again.",
                details={"file_path": file_path, "error": str(e)},
            )

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks based on token limits."""
        chunks = []
        paragraphs = text.split("\n\n")
        current_chunk = []
        current_chunk_tokens = 0

        # Use configured chunk size limit instead of Azure max tokens
        # This is the critical fix - use the configured chunk size (e.g., 2000 tokens)
        # instead of the Azure model limit (128,000 tokens)
        max_tokens = self.max_tokens  # Use configured chunk_size_limit

        logging.info(f"Splitting text with max_tokens limit: {max_tokens}")

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

        # Log the chunking results for debugging
        chunk_sizes = [len(self.simple_tokenize(chunk)) for chunk in chunks]
        logging.info(f"Text split into {len(chunks)} chunks with sizes: {chunk_sizes}")

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
        gcs_subfolder = (
            f"{self.configs.gcp_resource.gcp_embeddings_folder}/{file_id}/{subfolder}"
        )

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
        return self.model_token_limits.get(model_name.lower(), 2000)

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncates text to a maximum number of tokens, preserving whole words if possible."""
        if not text:  # Handle empty or None input text
            return ""

        tokens = self.simple_tokenize(text)  # Uses existing simple_tokenize
        if len(tokens) <= max_tokens:
            return text

        # Join the allowed number of tokens and return
        truncated_text = " ".join(tokens[:max_tokens])
        logging.debug(
            f"Truncated text from {len(tokens)} tokens to {max_tokens} tokens."
            f"Original length: {len(text)}, Truncated length: {len(truncated_text)}"
        )
        return truncated_text

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
