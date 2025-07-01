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
from rtl_rag_chatbot_api.common.text_extractor import TextExtractor

logging.basicConfig(level=logging.INFO)


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
        self.MODEL_TOKEN_LIMITS = {
            "gpt_4": 32000,
            "gpt_4o_mini": 128000,
            "gpt_4_omni": 128000,
            "gpt_4_1": 200000,  # GPT-4.1 token limit
            "gpt_4_1_nano": 50000,  # GPT-4.1-nano token limit
            "o3": 100000,  # O3 token limit
            "o3_mini": 100000,  # O3-mini token limit
            "o4_mini": 100000,  # O4-mini token limit
            "gemini-2.5-pro": 2097152,  # 2M tokens for Gemini 2.5 Pro
            "gemini-2.5-flash": 1048576,  # 1M tokens for Gemini 2.5 Flash
        }

        self.AZURE_MAX_TOKENS = self.MODEL_TOKEN_LIMITS.get(
            "gpt_4o_mini",
            128000,  # Default to gpt_4o_mini's limit for Azure if not specified elsewhere
        )
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
            subfolder (str): Storage subfolder (standardized on 'azure' for unified embedding approach)

        Returns:
            str: Status string ('completed' on success)
        """
        try:
            # Get collection with explicit subfolder path
            collection = self.chroma_manager.get_collection(
                file_id=file_id,
                embedding_type=subfolder,  # Always 'azure' in the unified embedding approach
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

            # Azure text-embedding-ada-002 has a maximum context length of 8192 tokens
            # For safety, we'll use a slightly lower limit
            AZURE_EMBEDDING_LIMIT = 8000

            for i in range(0, total_chunks):
                chunk = chunks[i]
                chunk_tokens = len(self.simple_tokenize(chunk))

                # Safety check: If chunk is still too large even after splitting,
                # log a warning and attempt to split it further
                if chunk_tokens > max_chunk_size:
                    logging.warning(
                        f"Chunk {i} has {chunk_tokens} tokens, exceeding limit of {max_chunk_size}. "
                        f"Attempting to split further."
                    )
                    sub_chunks = self.split_large_chunk(chunk, max_chunk_size)
                    for sub_idx, sub_chunk in enumerate(sub_chunks):
                        sub_chunk_tokens = len(self.simple_tokenize(sub_chunk))
                        if sub_chunk_tokens <= max_chunk_size:
                            batch_to_process.append(sub_chunk)
                            batch_ids.append(f"{embedding_id}_{i}_sub{sub_idx}")
                        else:
                            # If even after splitting, chunk is too large, truncate it as last resort
                            logging.error(
                                f"Sub-chunk {sub_idx} still has {sub_chunk_tokens} tokens. "
                                f"Truncating to {max_chunk_size} tokens."
                            )
                            truncated_chunk = self.truncate_text(
                                sub_chunk, max_chunk_size
                            )
                            batch_to_process.append(truncated_chunk)
                            batch_ids.append(
                                f"{embedding_id}_{i}_sub{sub_idx}_truncated"
                            )
                else:
                    batch_to_process.append(chunk)
                    batch_ids.append(f"{embedding_id}_{i}")

                # Calculate total tokens in current batch
                batch_total_tokens = sum(
                    len(self.simple_tokenize(b)) for b in batch_to_process
                )

                # Process batch if:
                # 1. We've reached BATCH_SIZE chunks, OR
                # 2. Adding the next chunk would exceed Azure's token limit, OR
                # 3. We're at the last chunk
                should_process_batch = (
                    len(batch_to_process) >= self.BATCH_SIZE
                    or batch_total_tokens > AZURE_EMBEDDING_LIMIT
                    or i == total_chunks - 1
                )

                if should_process_batch and batch_to_process:
                    try:
                        # Log batch details before processing
                        batch_token_counts = [
                            len(self.simple_tokenize(b)) for b in batch_to_process
                        ]
                        logging.info(
                            f"Processing batch {i // self.BATCH_SIZE + 1} with {len(batch_to_process)} chunks, "
                            f"token counts: {batch_token_counts}, total: {batch_total_tokens}"
                        )

                        # For large chunks that approach Azure's limit, process them individually
                        if (
                            batch_total_tokens > AZURE_EMBEDDING_LIMIT
                            and len(batch_to_process) > 1
                        ):
                            logging.info(
                                f"Batch total ({batch_total_tokens}) exceeds Azure limit. "
                                f"Processing chunks individually."
                            )
                            for idx, single_chunk in enumerate(batch_to_process):
                                single_chunk_tokens = len(
                                    self.simple_tokenize(single_chunk)
                                )
                                if single_chunk_tokens > AZURE_EMBEDDING_LIMIT:
                                    logging.error(
                                        f"Single chunk has {single_chunk_tokens} tokens, "
                                        f"exceeding Azure limit of {AZURE_EMBEDDING_LIMIT}"
                                    )
                                    # Split this chunk further
                                    smaller_chunks = self.split_large_chunk(
                                        single_chunk, AZURE_EMBEDDING_LIMIT - 100
                                    )
                                    for small_idx, small_chunk in enumerate(
                                        smaller_chunks
                                    ):
                                        embeddings = self.get_embeddings([small_chunk])
                                        collection.add(
                                            documents=[small_chunk],
                                            embeddings=embeddings,
                                            metadatas=[
                                                {
                                                    "source": file_id,
                                                    "embedding_id": embedding_id,
                                                }
                                            ],
                                            ids=[f"{batch_ids[idx]}_split_{small_idx}"],
                                        )
                                        processed_count += 1
                                else:
                                    embeddings = self.get_embeddings([single_chunk])
                                    collection.add(
                                        documents=[single_chunk],
                                        embeddings=embeddings,
                                        metadatas=[
                                            {
                                                "source": file_id,
                                                "embedding_id": embedding_id,
                                            }
                                        ],
                                        ids=[batch_ids[idx]],
                                    )
                                    processed_count += 1
                        else:
                            # Normal batch processing for smaller batches
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
                        logging.error(f"Error processing batch at chunk {i}: {str(e)}")
                        # Log the problematic batch details for debugging
                        problematic_counts = [
                            len(self.simple_tokenize(b)) for b in batch_to_process
                        ]
                        logging.error(
                            f"Problematic batch had {len(batch_to_process)} items "
                            f"with token counts: {problematic_counts}"
                        )
                    finally:
                        batch_to_process = []
                        batch_ids = []

            # Store the embedding_id for reference
            self.embedding_id = embedding_id
            logging.info(f"For {self.file_id}  embeddings are created successfully")
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
                return self.extract_text_from_pdf(file_path)
            elif file_extension in [".doc", ".docx"]:
                extractor = TextExtractor()
                return extractor.extract_text_from_doc(file_path)
            elif file_extension == ".txt":
                return self.extract_text_from_txt(file_path)
            else:
                logging.warning(
                    f"Unsupported file type: {file_extension}. Attempting generic text extraction."
                )
                return self.extract_text_from_txt(file_path)
        except Exception as e:
            error_msg = f"Error extracting text from {file_path}: {str(e)}"
            logging.error(error_msg)
            return f"ERROR: {error_msg}"

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
            with open(file_path, "rb") as file:
                text = file.read().decode("utf-8", errors="ignore")
                logging.info(
                    f"Successfully extracted {len(text)} characters from TXT file using fallback encoding"
                )
                return text

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
                        return f"ERROR: {error_msg}"

                except Exception as ocr_error:
                    error_msg = f"OCR extraction failed: {str(ocr_error)}"
                    logging.error(error_msg)
                    return f"ERROR: {error_msg}"

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
