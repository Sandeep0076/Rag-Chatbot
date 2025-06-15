# RAG PDF API - Postman Usage Guide

This guide provides detailed instructions on how to use each endpoint of the RAG PDF API with Postman. The API supports processing and querying various document types including PDFs, images, and tabular data (CSV/Excel) using Retrieval-Augmented Generation (RAG) and SQL querying capabilities.

## Table of Contents

- [Authentication](#authentication)
- [Basic Workflow](#basic-workflow)
- [Endpoints](#endpoints)
  - [File Upload](#file-upload)
  - [Check Embeddings](#check-embeddings)
  - [Chat with File](#chat-with-file)
  - [Get Available Models](#get-available-models)
  - [Chat with Gemini](#chat-with-gemini)
  - [Delete Resources](#delete-resources)
  - [Delete All Resources](#delete-all-resources)
  - [Find File by Name](#find-file-by-name)
  - [Manual Cleanup](#manual-cleanup)
- [Additional Information](#additional-information)

## Authentication

All endpoints require authentication. The API uses OAuth authentication, which is handled by the `get_current_user` dependency.

In Postman, you need to add an Authorization header to your requests. The specific authentication method depends on your deployment configuration.

## Basic Workflow

The typical workflow for using the API is:

1. Upload a file (PDF, image, CSV, or Excel) using the `/file/upload` endpoint
4. Chat with the content using the `/file/chat` endpoint

## Endpoints

### File Upload

Upload a file (PDF, image, CSV, or Excel) to the API.

- **Endpoint URL**: `/file/upload`
- **HTTP Method**: POST
- **Request Headers**:
  - `Authorization`: Your auth token
- **Request Body**:
  - Form data:
    - `file`: The file to upload (File)
    - `is_image`: Whether the file is an image (Boolean)
    - `username`: Username for tracking file ownership (String)
- **Response Format**:
  ```json
  {
    "file_id": "uuid-string",
    "message": "File uploaded successfully",
    "status": "success",
    "original_filename": "example.pdf",
    "is_image": false
  }
  ```
- **Usage Example**:
  1. Open Postman and create a new request
  2. Set the request method to POST
  3. Enter the URL: `http://your-api-domain/file/upload`
  4. Go to the "Body" tab and select "form-data"
  5. Add the following key-value pairs:
     - Key: `file`, Value: Select a file from your computer
     - Key: `is_image`, Value: `false` (or `true` if uploading an image)
     - Key: `username`, Value: Enter a username
  6. Click "Send" to upload the file


### Check Embeddings

Check if embeddings exist for a specific file and model.

- **Endpoint URL**: `/embeddings/check`
- **HTTP Method**: POST
- **Request Headers**:
  - `Authorization`: Your auth token
  - `Content-Type`: `application/json`
- **Request Body**:
  ```json
  {
    "file_id": "uuid-string",
    "model_choice": "gpt_4o_mini"
  }
  ```
- **Response Format**:
  ```json
  {
    "embeddings_exist": true,
    "model_type": "azure",
    "file_id": "uuid-string",
    "status": "ready_for_chat"
  }
  ```
- **Usage Example**:
  1. Create a new request in Postman
  2. Set the request method to POST
  3. Enter the URL: `http://your-api-domain/embeddings/check`
  4. Go to the "Body" tab and select "raw" and "JSON"
  5. Enter the JSON request body with the file_id and model_choice
  6. Click "Send" to check if embeddings exist

### Check Embedding Status

Get the current status of embeddings for a specific file. The frontend can poll this endpoint to know when embeddings are ready for chat.

- **Endpoint URL**: `/embeddings/status/{file_id}`
- **HTTP Method**: GET
- **Request Headers**:
  - `Authorization`: Your auth token
- **Path Parameters**:
  - `file_id`: The ID of the file to check
- **Response Format**:
  ```json
  {
    "status": "ready_for_chat",
    "can_chat": true,
    "file_id": "uuid-string",
    "message": "Embeddings are ready for chat. Background upload still in progress."
  }
  ```
- **Possible Status Values**:
  - `not_started`: Embeddings generation has not started
  - `in_progress`: Embeddings are being generated
  - `ready_for_chat`: Local embeddings are ready for chat, but cloud upload may still be in progress
  - `completed`: Embeddings are fully generated and uploaded to cloud storage
- **Usage Example**:
  1. Create a new request in Postman
  2. Set the request method to GET
  3. Enter the URL: `http://your-api-domain/embeddings/status/{your-file-id}`
  4. Click "Send" to check the embedding status

### Chat with File

Process chat queries against document content using specified language models.

- **Endpoint URL**: `/file/chat`
- **HTTP Method**: POST
- **Request Headers**:
  - `Authorization`: Your auth token
  - `Content-Type`: `application/json`
- **Request Body**:
  ```json
  {
    "text": ["What is this document about?"],
    "file_id": "uuid-string",
    "model_choice": "gpt_4o_mini",
    "user_id": "user123",
    "generate_visualization": false
  }
  ```
- **Response Format**:
  ```json
  {
    "response": "This document is about...",
    "is_table": false
  }
  ```
  For tabular data, the response may include formatted table data:
  ```json
  {
    "response": "| Column1 | Column2 |\n|---------|---------|...",
    "is_table": true,
    "headers": ["Column1", "Column2", ...],
    "rows": [["value1", "value2"], ...]
  }
  ```
- **Usage Example**:
  1. Create a new request in Postman
  2. Set the request method to POST
  3. Enter the URL: `http://your-api-domain/file/chat`
  4. Go to the "Body" tab and select "raw" and "JSON"
  5. Enter the JSON request body with your query
  6. Click "Send" to chat with the document

### Get Available Models

Retrieve a list of available models including Azure LLM models and Gemini models.

- **Endpoint URL**: `/available-models`
- **HTTP Method**: GET
- **Request Headers**:
  - `Authorization`: Your auth token
- **Response Format**:
  ```json
  {
    "models": ["gpt_4o_mini", "gpt_4o", "gpt_3_5_turbo", "gemini-flash", "gemini-pro"]
  }
  ```
- **Usage Example**:
  1. Create a new request in Postman
  2. Set the request method to GET
  3. Enter the URL: `http://your-api-domain/available-models`
  4. Click "Send" to get the list of available models

### Chat with Gemini

Chat with Gemini models (Flash or Pro) without RAG or file context.

- **Endpoint URL**: `/chat/gemini`
- **HTTP Method**: POST
- **Request Headers**:
  - `Authorization`: Your auth token
  - `Content-Type`: `application/json`
- **Request Body**:
  ```json
  {
    "model": "gemini-pro",
    "message": "Tell me about artificial intelligence"
  }
  ```
- **Response Format**: Streaming text response
- **Usage Example**:
  1. Create a new request in Postman
  2. Set the request method to POST
  3. Enter the URL: `http://your-api-domain/chat/gemini`
  4. Go to the "Body" tab and select "raw" and "JSON"
  5. Enter the JSON request body with your message
  6. Click "Send" to chat with Gemini

### Delete Resources

Delete ChromaDB embeddings and associated resources for one or multiple files based on username.

- **Endpoint URL**: `/delete`
- **HTTP Method**: DELETE
- **Request Headers**:
  - `Authorization`: Your auth token
  - `Content-Type`: `application/json`
- **Request Body**:
  ```json
  {
    "file_ids": "uuid-string",
    "include_gcs": false,
    "username": "user123"
  }
  ```
  or for multiple files:
  ```json
  {
    "file_ids": ["uuid-string1", "uuid-string2"],
    "include_gcs": false,
    "username": "user123"
  }
  ```
- **Response Format**:
  For a single file:
  ```json
  {
    "message": "Username removed from file info"
  }
  ```
  For multiple files:
  ```json
  {
    "results": {
      "uuid-string1": "Success: Username removed from file info",
      "uuid-string2": "Success: Embeddings deleted"
    }
  }
  ```
- **Usage Example**:
  1. Create a new request in Postman
  2. Set the request method to DELETE
  3. Enter the URL: `http://your-api-domain/delete`
  4. Go to the "Body" tab and select "raw" and "JSON"
  5. Enter the JSON request body with the file_ids
  6. Click "Send" to delete resources

### Delete All Resources

Delete ChromaDB embeddings and associated resources for one or multiple files (General method).

- **Endpoint URL**: `/delete_all_embeddings`
- **HTTP Method**: DELETE
- **Request Headers**:
  - `Authorization`: Your auth token
  - `Content-Type`: `application/json`
- **Request Body**:
  ```json
  {
    "file_ids": "uuid-string",
    "include_gcs": false
  }
  ```
  or for multiple files:
  ```json
  {
    "file_ids": ["uuid-string1", "uuid-string2"],
    "include_gcs": false
  }
  ```
- **Response Format**:
  For a single file:
  ```json
  {
    "message": "ChromaDB embeddings for file_id uuid-string have been deleted successfully"
  }
  ```
  For multiple files:
  ```json
  {
    "results": {
      "uuid-string1": "Success",
      "uuid-string2": "Success"
    }
  }
  ```
- **Usage Example**:
  1. Create a new request in Postman
  2. Set the request method to DELETE
  3. Enter the URL: `http://your-api-domain/delete_all_embeddings`
  4. Go to the "Body" tab and select "raw" and "JSON"
  5. Enter the JSON request body with the file_ids
  6. Click "Send" to delete all resources

### Find File by Name

Find a file_id by searching through all file_info.json files for a matching original_filename.

- **Endpoint URL**: `/find-file-by-name`
- **HTTP Method**: GET
- **Request Headers**:
  - `Authorization`: Your auth token
- **Request Parameters**:
  - `filename`: The original filename to search for
- **Response Format**:
  ```json
  {
    "file_id": "uuid-string",
    "found": true
  }
  ```
- **Usage Example**:
  1. Create a new request in Postman
  2. Set the request method to GET
  3. Enter the URL: `http://your-api-domain/find-file-by-name?filename=example.pdf`
  4. Click "Send" to find the file by name

### Manual Cleanup

Manually trigger cleanup of resources.

- **Endpoint URL**: `/file/cleanup`
- **HTTP Method**: POST
- **Request Headers**:
  - `Authorization`: Your auth token
  - `Content-Type`: `application/json`
- **Request Body**:
  ```json
  {
    "is_manual": true
  }
  ```
- **Response Format**:
  ```json
  {
    "status": "Cleanup completed successfully"
  }
  ```
- **Usage Example**:
  1. Create a new request in Postman
  2. Set the request method to POST
  3. Enter the URL: `http://your-api-domain/file/cleanup`
  4. Go to the "Body" tab and select "raw" and "JSON"
  5. Enter the JSON request body
  6. Click "Send" to trigger manual cleanup

## Additional Information

### Setting Up Environment Variables in Postman

To make your requests more manageable, you can set up environment variables in Postman:

1. Click on the "Environment" dropdown in the top right corner
2. Click "New" to create a new environment
3. Add variables such as:
   - `base_url`: Your API base URL
   - `auth_token`: Your authentication token
4. Use these variables in your requests like `{{base_url}}/file/upload` and `{{auth_token}}` in the Authorization header

### Error Handling

The API returns appropriate HTTP status codes for different error scenarios:
- 400: Bad Request (invalid input)
- 404: Not Found (resource not found)
- 500: Internal Server Error (server-side error)

Error responses typically include a `detail` field with more information about the error.
