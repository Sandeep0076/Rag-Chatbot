import json
import os

os.environ["azure_llm"] = json.dumps(
    {
        "azure_llm_api_key": os.getenv("AZURE_OPENAI_LLM_API_KEY"),
        "azure_llm_endpoint": os.getenv("AZURE_OPENAI_LLM_ENDPOINT"),
        "azure_llm_api_version": os.getenv("AZURE_LLM_API_VERSION"),
        "azure_llm_deployment": os.getenv("AZURE_LLM_DEPLOYMENT"),
        "azure_llm_model_name": os.getenv("AZURE_LLM_MODEL_NAME"),
    }
)

os.environ["azure_embedding"] = json.dumps(
    {
        "azure_embedding_api_key": os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
        "azure_embedding_endpoint": os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
        "azure_embedding_api_version": os.getenv("AZURE_EMBEDDING_API_VERSION"),
        "azure_embedding_deployment": os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
        "azure_embedding_model_name": os.getenv("AZURE_EMBEDDING_MODEL_NAME"),
    }
)

os.environ["chatbot"] = json.dumps(
    {
        "system_prompt_plain_llm": os.getenv("SYSTEM_PROMPT_PLAIN_LLM"),
        "system_prompt_rag_llm": os.getenv("SYSTEM_PROMPT_RAG_LLM"),
        "vector_db_collection_name": os.getenv("VECTOR_DB_COLLECTION_NAME"),
        "image_file_path": os.getenv("IMAGE_FILE_PATH"),
        "title": os.getenv("TITLE"),
        "description": os.getenv("DESCRIPTION"),
        "info_text": os.getenv("INFO_TEXT"),
        "max_input_size": os.getenv("MAX_INPUT_SIZE", "4096"),
        "num_outputs": os.getenv("NUM_OUTPUTS", "2000"),
        "max_chunk_overlap": os.getenv("MAX_CHUNK_OVERLAP", "0.2"),
        "chunk_size_limit": os.getenv("CHUNK_SIZE_LIMIT", "400"),
        "n_neighbours": os.getenv("N_NEIGHBOURS", "5"),
    }
)


os.environ["llm_hyperparams"] = json.dumps(
    {
        "temperature": os.getenv("TEMPERATURE", "0.1"),
        "max_tokens": os.getenv("MAX_TOKENS", "2000"),
        "top_p": os.getenv("TOP_P", "0.8"),
        "frequency_penalty": os.getenv("FREQUENCY_PENALTY", "0.0"),
        "presence_penalty": os.getenv("PRESENCE_PENALTY", "0.0"),
        "stop": os.getenv("STOP_WORDS_LIST", "").split(",")
        if os.getenv("STOP_WORDS_LIST")
        else [],
    }
)

os.environ["gcp_resource"] = json.dumps(
    {
        "gcp_project": os.getenv("GCP_PROJECT"),
        "bucket_name": os.getenv("BUCKET_NAME"),
        "embeddings_folder": os.getenv("EMBEDDINGS_FOLDER", "embeddings_folder/"),
    }
)
