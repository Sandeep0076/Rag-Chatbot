import logging
from typing import Any

from configs.app_config import Config
from rtl_rag_chatbot_api.common.vector_db_creator import VectorDbWrapper


def run_preprocessor(
    configs: Config,
    text_data_folder_path: str,
    file_id: str,
    chroma_db_path: str,
    chroma_collection: Any,
    is_image: bool,
    gcs_handler,
    username: str,
    collection_name=None,
):
    """
    Runs the data preprocessor which reads PDF data, converts it into a vector database,
    and uploads the database files to Google Cloud Storage.

    Args:
    configs (Config): Configuration object containing necessary settings.
    text_data_folder_path (str): The path to the folder containing the text data to be processed.
    file_id (str): Unique identifier for the file being processed.
    chroma_db_path (str): Path to store the Chroma database.

    The function performs the following steps:
    1. Initializes the VectorDbWrapper with Azure and GCP configurations.
    2. Logs the start of the index creation and storage process.
    3. Creates and stores the index using specified chunk size and overlap.
    4. Uploads the created database files to GCS.
    """
    # Create and store index in the specified storage folder
    # Initialize VectorDbWrapper with environment variables and text data path
    my_wrapper = VectorDbWrapper(
        azure_api_key=configs.azure_embedding.azure_embedding_api_key,
        azure_endpoint=configs.azure_embedding.azure_embedding_endpoint,
        text_data_folder_path=text_data_folder_path,
        gcp_project=configs.gcp_resource.gcp_project,
        bucket_name=configs.gcp_resource.bucket_name,
        gcs_subfolder="file-embeddings",
        file_id=file_id,
        chroma_collection=chroma_collection,
        is_image=is_image,
        gcs_handler=gcs_handler,
        username=username,
    )

    logging.info("Now creating and storing index")
    logging.info(
        f"Now creating index with these parameters \n chunk_size: "
        f"{configs.chatbot.chunk_size_limit} \n chunk_overlap: "
        "{configs.chatbot.max_chunk_overlap}"
    )

    # Create and store index in the specified storage folder
    my_wrapper.create_and_store_index(
        storage_folder=chroma_db_path,
        collection_name=collection_name,
        chunk_size=configs.chatbot.chunk_size_limit,
        chunk_overlap=configs.chatbot.max_chunk_overlap,
    )

    logging.info("Now uploading files to GCS")
    logging.info(f"{collection_name} collection is being used")
    # Upload database files to Google Cloud Storage
    my_wrapper.upload_db_files_to_gcs()
    """
    Command-line usage examples:
    - python src/main.py ./upload
    """
