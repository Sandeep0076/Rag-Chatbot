"""
Centralized error definitions and FastAPI exception handlers.

This module standardizes error codes, keys and response shapes across the API.
Use BaseAppError (or subclasses) to raise structured errors from business logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse


@dataclass(frozen=True)
class ErrorSpec:
    """Error specification containing code, key, and HTTP status."""

    code: int  # Numeric error code (e.g., 2005)
    key: str  # Stable error key for UI translation/messaging (e.g., ERROR_PDF_PARSING_FAILED)
    http_status: int


class ErrorRegistry:
    """Registry of all application error specifications."""

    # 1xxx General & Authentication
    ERROR_UNKNOWN = ErrorSpec(1000, "ERROR_UNKNOWN", 500)
    ERROR_API_KEY_MISSING = ErrorSpec(1001, "ERROR_API_KEY_MISSING", 401)
    ERROR_API_KEY_INVALID = ErrorSpec(1002, "ERROR_API_KEY_INVALID", 401)
    ERROR_RATE_LIMIT_EXCEEDED = ErrorSpec(1003, "ERROR_RATE_LIMIT_EXCEEDED", 429)
    ERROR_SERVER_UNAVAILABLE = ErrorSpec(1004, "ERROR_SERVER_UNAVAILABLE", 503)
    ERROR_INSUFFICIENT_PERMISSIONS = ErrorSpec(
        1005, "ERROR_INSUFFICIENT_PERMISSIONS", 403
    )
    ERROR_BAD_REQUEST = ErrorSpec(1010, "ERROR_BAD_REQUEST", 400)

    # 2xxx File & Document
    ERROR_FILE_UPLOAD_FAILED = ErrorSpec(2001, "ERROR_FILE_UPLOAD_FAILED", 500)
    ERROR_FILE_TYPE_UNSUPPORTED = ErrorSpec(2002, "ERROR_FILE_TYPE_UNSUPPORTED", 400)
    ERROR_FILE_SIZE_EXCEEDED = ErrorSpec(2003, "ERROR_FILE_SIZE_EXCEEDED", 413)
    ERROR_FILE_CORRUPTED = ErrorSpec(2004, "ERROR_FILE_CORRUPTED", 400)
    ERROR_PDF_PARSING_FAILED = ErrorSpec(2005, "ERROR_PDF_PARSING_FAILED", 400)
    ERROR_CSV_PARSING_FAILED = ErrorSpec(2006, "ERROR_CSV_PARSING_FAILED", 400)
    ERROR_DOCS_PARSING_FAILED = ErrorSpec(2007, "ERROR_DOCS_PARSING_FAILED", 400)
    ERROR_DOCUMENT_INDEXING_FAILED = ErrorSpec(
        2008, "ERROR_DOCUMENT_INDEXING_FAILED", 500
    )
    ERROR_FILE_NOT_FOUND = ErrorSpec(2009, "ERROR_FILE_NOT_FOUND", 404)
    ERROR_DOC_TEXT_TOO_SHORT = ErrorSpec(2010, "ERROR_DOC_TEXT_TOO_SHORT", 400)
    ERROR_URL_EXTRACTION_FAILED = ErrorSpec(2011, "ERROR_URL_EXTRACTION_FAILED", 400)
    ERROR_URL_CONTENT_TOO_SHORT = ErrorSpec(2012, "ERROR_URL_CONTENT_TOO_SHORT", 400)
    ERROR_TABULAR_INVALID_DATA = ErrorSpec(2013, "ERROR_TABULAR_INVALID_DATA", 422)
    ERROR_CSV_NO_TABLES = ErrorSpec(2014, "ERROR_CSV_NO_TABLES", 400)
    ERROR_CSV_ALL_TABLES_EMPTY = ErrorSpec(2015, "ERROR_CSV_ALL_TABLES_EMPTY", 400)
    ERROR_FILE_HASH_FAILED = ErrorSpec(2016, "ERROR_FILE_HASH_FAILED", 500)
    ERROR_FILE_SAVE_FAILED = ErrorSpec(2017, "ERROR_FILE_SAVE_FAILED", 500)
    ERROR_DOC_TEXT_VALIDATION_FAILED = ErrorSpec(
        2018, "ERROR_DOC_TEXT_VALIDATION_FAILED", 400
    )
    ERROR_TXT_EXTRACTION_FAILED = ErrorSpec(2019, "ERROR_TXT_EXTRACTION_FAILED", 400)

    # 3xxx Image
    ERROR_IMAGE_READER_FAILED = ErrorSpec(3001, "ERROR_IMAGE_READER_FAILED", 400)
    ERROR_IMAGE_FORMAT_UNSUPPORTED = ErrorSpec(
        3002, "ERROR_IMAGE_FORMAT_UNSUPPORTED", 415
    )
    ERROR_IMAGE_CREATION_FAILED = ErrorSpec(3003, "ERROR_IMAGE_CREATION_FAILED", 500)
    ERROR_IMAGE_CREATION_PROMPT_REJECTED = ErrorSpec(
        3004, "ERROR_IMAGE_CREATION_PROMPT_REJECTED", 422
    )
    ERROR_IMAGE_DIMENSIONS_INVALID = ErrorSpec(
        3005, "ERROR_IMAGE_DIMENSIONS_INVALID", 400
    )
    ERROR_IMAGE_ANALYSIS_FAILED = ErrorSpec(3006, "ERROR_IMAGE_ANALYSIS_FAILED", 500)

    # 4xxx Query & Chat
    ERROR_QUERY_INVALID = ErrorSpec(4001, "ERROR_QUERY_INVALID", 400)
    ERROR_CONTEXT_RETRIEVAL_FAILED = ErrorSpec(
        4002, "ERROR_CONTEXT_RETRIEVAL_FAILED", 404
    )
    ERROR_LLM_GENERATION_FAILED = ErrorSpec(4003, "ERROR_LLM_GENERATION_FAILED", 500)
    ERROR_QUERY_TOO_LONG = ErrorSpec(4004, "ERROR_QUERY_TOO_LONG", 413)
    ERROR_NO_SOURCE_SELECTED = ErrorSpec(4005, "ERROR_NO_SOURCE_SELECTED", 400)
    ERROR_CHART_JSON_INVALID = ErrorSpec(4006, "ERROR_CHART_JSON_INVALID", 400)
    ERROR_CHART_GENERATION_FAILED = ErrorSpec(
        4007, "ERROR_CHART_GENERATION_FAILED", 500
    )
    ERROR_SAFETY_FILTER_BLOCKED = ErrorSpec(4008, "ERROR_SAFETY_FILTER_BLOCKED", 422)
    ERROR_MODEL_INITIALIZATION_FAILED = ErrorSpec(
        4009, "ERROR_MODEL_INITIALIZATION_FAILED", 500
    )
    ERROR_EMBEDDING_CREATION_FAILED = ErrorSpec(
        4010, "ERROR_EMBEDDING_CREATION_FAILED", 500
    )
    ERROR_EMBEDDINGS_NOT_FOUND = ErrorSpec(4011, "ERROR_EMBEDDINGS_NOT_FOUND", 404)
    ERROR_AGENT_EXECUTION_FAILED = ErrorSpec(4012, "ERROR_AGENT_EXECUTION_FAILED", 500)
    ERROR_TITLE_GENERATION_FAILED = ErrorSpec(
        4013, "ERROR_TITLE_GENERATION_FAILED", 500
    )
    ERROR_API_OVERLOADED = ErrorSpec(4014, "ERROR_API_OVERLOADED", 503)
    ERROR_API_RETRY_EXHAUSTED = ErrorSpec(4015, "ERROR_API_RETRY_EXHAUSTED", 503)

    # 5xxx Database & Storage
    ERROR_DATABASE_CONNECTION_FAILED = ErrorSpec(
        5001, "ERROR_DATABASE_CONNECTION_FAILED", 500
    )
    ERROR_DATABASE_QUERY_FAILED = ErrorSpec(5002, "ERROR_DATABASE_QUERY_FAILED", 500)
    ERROR_GCS_UPLOAD_FAILED = ErrorSpec(5003, "ERROR_GCS_UPLOAD_FAILED", 500)
    ERROR_GCS_DOWNLOAD_FAILED = ErrorSpec(5004, "ERROR_GCS_DOWNLOAD_FAILED", 500)
    ERROR_GCS_DELETE_FAILED = ErrorSpec(5005, "ERROR_GCS_DELETE_FAILED", 500)
    ERROR_ENCRYPTION_FAILED = ErrorSpec(5006, "ERROR_ENCRYPTION_FAILED", 500)
    ERROR_DECRYPTION_FAILED = ErrorSpec(5007, "ERROR_DECRYPTION_FAILED", 500)


class BaseAppError(Exception):
    """Base exception class for all application errors."""

    def __init__(
        self, spec: ErrorSpec, message: str, details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.spec = spec
        self.message = message
        self.details = details or {}

    def to_response(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Convert error to standardized response format."""
        payload: Dict[str, Any] = {
            "status": "error",
            "code": self.spec.code,
            "key": self.spec.key,
            "http_status": self.spec.http_status,
            # Backward compatibility fields
            "error_code": self.spec.code,
            "error_key": self.spec.key,
            "message": self.message,
        }
        if self.details:
            payload["details"] = self.details
        if extra:
            payload.update(extra)
        return payload


# Specific error types
class PdfTextExtractionError(BaseAppError):
    """Error raised when PDF text extraction fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorRegistry.ERROR_PDF_PARSING_FAILED, message, details)


class DocTextTooShortError(BaseAppError):
    """Error raised when document text is too short."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorRegistry.ERROR_DOC_TEXT_TOO_SHORT, message, details)


class DocTextValidationError(BaseAppError):
    """Error raised when document text validation fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            ErrorRegistry.ERROR_DOC_TEXT_VALIDATION_FAILED, message, details
        )


class TxtExtractionError(BaseAppError):
    """Error raised when TXT file extraction fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorRegistry.ERROR_TXT_EXTRACTION_FAILED, message, details)


class CsvInvalidOrEmptyError(BaseAppError):
    """Error raised when CSV is invalid or empty."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorRegistry.ERROR_CSV_PARSING_FAILED, message, details)


class CsvNoTablesError(BaseAppError):
    """Error raised when CSV has no tables."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorRegistry.ERROR_CSV_NO_TABLES, message, details)


class CsvAllTablesEmptyError(BaseAppError):
    """Error raised when all CSV tables are empty."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorRegistry.ERROR_CSV_ALL_TABLES_EMPTY, message, details)


class TabularInvalidDataError(BaseAppError):
    """Error raised when tabular data is invalid."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorRegistry.ERROR_TABULAR_INVALID_DATA, message, details)


class UrlExtractionError(BaseAppError):
    """Error raised when URL content extraction fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorRegistry.ERROR_URL_EXTRACTION_FAILED, message, details)


class UrlContentTooShortError(BaseAppError):
    """Error raised when URL content is too short."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorRegistry.ERROR_URL_CONTENT_TOO_SHORT, message, details)


class ImageCreationError(BaseAppError):
    """Error raised when image creation fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorRegistry.ERROR_IMAGE_CREATION_FAILED, message, details)


class ImagePromptRejectedError(BaseAppError):
    """Error raised when image creation prompt is rejected."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            ErrorRegistry.ERROR_IMAGE_CREATION_PROMPT_REJECTED, message, details
        )


class ImageAnalysisError(BaseAppError):
    """Error raised when image analysis fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorRegistry.ERROR_IMAGE_ANALYSIS_FAILED, message, details)


class SafetyFilterError(BaseAppError):
    """Error raised when content is blocked by safety filters."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorRegistry.ERROR_SAFETY_FILTER_BLOCKED, message, details)


class EmbeddingCreationError(BaseAppError):
    """Error raised when embedding creation fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            ErrorRegistry.ERROR_EMBEDDING_CREATION_FAILED, message, details
        )


class EmbeddingsNotFoundError(BaseAppError):
    """Error raised when embeddings are not found."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorRegistry.ERROR_EMBEDDINGS_NOT_FOUND, message, details)


class ModelInitializationError(BaseAppError):
    """Error raised when model initialization fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            ErrorRegistry.ERROR_MODEL_INITIALIZATION_FAILED, message, details
        )


class ChartGenerationError(BaseAppError):
    """Error raised when chart generation fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorRegistry.ERROR_CHART_GENERATION_FAILED, message, details)


class FileUploadError(BaseAppError):
    """Error raised when file upload fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorRegistry.ERROR_FILE_UPLOAD_FAILED, message, details)


class GcsUploadError(BaseAppError):
    """Error raised when GCS upload fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorRegistry.ERROR_GCS_UPLOAD_FAILED, message, details)


class GcsDownloadError(BaseAppError):
    """Error raised when GCS download fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorRegistry.ERROR_GCS_DOWNLOAD_FAILED, message, details)


def register_exception_handlers(app: FastAPI) -> None:
    """Register centralized exception handlers for the FastAPI app."""

    @app.exception_handler(BaseAppError)
    async def handle_base_app_error(_, exc: BaseAppError):
        return JSONResponse(status_code=exc.spec.http_status, content=exc.to_response())

    from fastapi import HTTPException

    @app.exception_handler(HTTPException)
    async def handle_http_exception(_, exc: HTTPException):
        # If detail already contains structured payload, pass through
        if isinstance(exc.detail, dict) and (
            "error_key" in exc.detail or "error_code" in exc.detail
        ):
            content = exc.detail
        else:
            # Fallback mapping based on status code
            status = exc.status_code or 500
            spec = (
                ErrorRegistry.ERROR_BAD_REQUEST
                if status == 400
                else ErrorRegistry.ERROR_API_KEY_INVALID
                if status == 401
                else ErrorRegistry.ERROR_INSUFFICIENT_PERMISSIONS
                if status == 403
                else ErrorRegistry.ERROR_FILE_NOT_FOUND
                if status == 404
                else ErrorRegistry.ERROR_RATE_LIMIT_EXCEEDED
                if status == 429
                else ErrorRegistry.ERROR_UNKNOWN
            )
            content = BaseAppError(spec, str(exc.detail)).to_response()
        return JSONResponse(status_code=exc.status_code, content=content)

    # Optional: generic catch-all to avoid leaking internals
    @app.exception_handler(Exception)
    async def handle_unexpected_error(_, exc: Exception):
        # Avoid exposing internal exception details
        import logging

        logging.exception("Unexpected error occurred")
        error = BaseAppError(
            ErrorRegistry.ERROR_UNKNOWN,
            "An unexpected error occurred. Please try again later.",
        )
        return JSONResponse(status_code=500, content=error.to_response())


def map_exception_to_app_error(exc: Exception) -> BaseAppError:
    """Best-effort mapping from raised exceptions/messages to structured app errors."""
    msg = str(exc) if exc else ""
    low = msg.lower()

    if "no tables" in low:
        return CsvNoTablesError(msg)
    if "all tables are empty" in low or "no data" in low:
        return CsvAllTablesEmptyError(msg)
    if (
        "failed to process csv" in low
        or "failed to read csv" in low
        or "csv file" in low
    ):
        return CsvInvalidOrEmptyError(msg)
    if "tabular" in low and ("invalid" in low or "corrupted" in low):
        return TabularInvalidDataError(msg)
    if "pdf" in low and (
        "unable to extract" in low or "unable to read" in low or "ocr" in low
    ):
        return PdfTextExtractionError(msg)
    if "less than 100 characters" in low or "insufficient text" in low:
        return DocTextTooShortError(msg)
    if "embedding" in low and "fail" in low:
        return EmbeddingCreationError(msg)
    if "maximum context length" in low or (
        "token" in low and ("limit" in low or "overflow" in low or "exceed" in low)
    ):
        return EmbeddingCreationError(msg)
    if "safety" in low and "block" in low:
        return SafetyFilterError(msg)
    if "chart" in low and ("json" in low or "parse" in low or "invalid" in low):
        return ChartGenerationError(msg)
    if "url" in low and ("extract" in low or "content" in low):
        return UrlExtractionError(msg)
    if "image" in low and ("creation" in low or "generation" in low):
        return ImageCreationError(msg)

    return BaseAppError(ErrorRegistry.ERROR_BAD_REQUEST, msg or "Bad request")


def build_error_result(
    error: BaseAppError,
    file_id: Optional[str] = None,
    is_image: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build error result with optional file context."""
    extra: Dict[str, Any] = {}
    if file_id is not None:
        extra["file_id"] = file_id
    if is_image is not None:
        extra["is_image"] = is_image
    return error.to_response(extra=extra)
