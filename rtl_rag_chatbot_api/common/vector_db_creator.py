import os
from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from llama_index.core import ServiceContext, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore


class VectorDbWrapper:
    """
    Class to handle creation of Chroma DB Index

    This class is meant to handle the creation and uploading of a Chroma
    Vector DB. Specifically, it:
    - Creates Chroma DB artifacts from any kind of text files in a folder
    - Uploads those artifacts into a GCP bucket

    Attributes:
    azure_api_key (str): API key to access a private Azure endpoint
    azure_endpoint (str): Azure endpoint where an LLM and embeddings model are deployed
    text_data_folder_path (str): Relative folder name where text data to be embedded is stored
    gcp_project (str): GCP project where bucket is into which files should be uploaded
    bucket_name (str): GCP bucket to which files should be uploaded
    llm_model (AzureOpenAI): LLM model instance
    embedding_model (AzureOpenAIEmbedding): Embedding model instance
    gcs_subfolder (str): Subfolder in GCS bucket for storing embeddings
    file_id (str): Unique identifier for the file being processed
    """

    def __init__(
        self,
        azure_api_key,
        azure_endpoint,
        gcs_handler,
        gcp_project,
        bucket_name,
        text_data_folder_path,
        gcs_subfolder="file-embeddings",
        file_id=None,
        chroma_db=None,
        is_image=False,
        username=None,
    ):
        self.azure_api_key = azure_api_key
        self.azure_endpoint = azure_endpoint
        self.text_data_folder_path = text_data_folder_path
        self.gcp_project = gcp_project
        self.bucket_name = bucket_name
        self.llm_model = self._init_llm_model()
        self.embedding_model = self._init_embedding_model()
        self.gcs_subfolder = gcs_subfolder
        self.file_id = file_id
        self.chroma_db = chroma_db
        self.documents = self._create_list_of_documents()
        self.is_image = is_image
        self.gcs_handler = gcs_handler
        self.username = username

    def _create_list_of_documents(self) -> List:
        """Create a list of Llama_index Document classes

        This method takes all files in self.text_data_folder_path and loads them
        as a list of llama_index.Documents classes.

        Uses the SimpleDirectoryReader which can handle all standard file types
        (e.g. .json, .txt, ...)

        Return:
            List of llama_index Documents
        """
        # Load all files from folder
        documents = SimpleDirectoryReader(
            # input_dir=os.getcwd() + "\\text_data",
            input_dir=self.text_data_folder_path,
            recursive=False,
        ).load_data()

        doc_text = "\n\n".join([d.get_content() for d in documents])

        print(f"The total number of words in the docs is {len(doc_text.split())}")
        self.n_total_words = len(doc_text.split())

        # Create a list of documents
        # docs = [Document(text=doc_text)] # TODO if new code works, delete this
        # Unclear wht the logic behind this step was
        print(f"The number of docs in the list is {len(documents)}")

        return documents

    def _init_llm_model(self) -> AzureOpenAI:
        """Instantiate an AzureChatOpenAI class to be used as LLM"""
        llm_model = AzureOpenAI(
            model="gpt-35-turbo",
            azure_deployment="mgr-openai-text-gpt35-turbo",
            api_key=self.azure_api_key,
            azure_endpoint=self.azure_endpoint,
            api_version="2023-07-01-preview",
        )

        return llm_model

    def _init_embedding_model(self) -> AzureOpenAIEmbedding:
        """Instantiate an AzureOpenAIEmbedding class

        Used to create embeddings of the files in self.text_data_folder_path
        """
        embedding_model = AzureOpenAIEmbedding(
            model="text-embedding-ada-002",
            deployment_name="mgr-openai-embedding-ada-002",
            api_key=self.azure_api_key,
            azure_endpoint=self.azure_endpoint,
            api_version="2023-07-01-preview",
        )

        return embedding_model

    def create_and_store_index(
        self,
        storage_folder: str = "./chroma_db",
        collection_name: str = "RAG_CHATBOT",
        chunk_size: int = 400,
        chunk_overlap: int = 40,
    ) -> None:
        """
        Create and store a vector index for a RAG (Retrieval-Augmented Generation) chatbot using Chroma DB.

        This method processes the documents stored in the instance, chunks them into nodes,
        creates a vector index, and stores it in a persistent Chroma DB. The index can be
        used later for efficient similarity searches in the RAG chatbot.

        Args:
            storage_folder (str, optional): The path where the Chroma DB will be stored.
                Defaults to "./chroma_db".
            collection_name (str, optional): The name of the collection in Chroma DB.
                Defaults to "RAG_CHATBOT".
            chunk_size (int, optional): The size of each text chunk when parsing documents.
                Defaults to 400.
            chunk_overlap (int, optional): The number of overlapping tokens between chunks.
                Defaults to 40.

        Returns:
            None
        """
        if self.chroma_db:
            db = self.chroma_db
        else:
            db = chromadb.PersistentClient(
                path=storage_folder,
                settings=Settings(allow_reset=True, is_persistent=True),
            )

        chroma_collection = db.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Always create the service_context
        service_context = ServiceContext.from_defaults(
            llm=self.llm_model, embed_model=self.embedding_model
        )

        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=chunk_size,
            separator=" ",
            chunk_overlap=chunk_overlap,
            tokenizer=None,
            paragraph_separator="\n\n\n",
            chunking_tokenizer_fn=None,
            secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?",
            callback_manager=None,
            include_metadata=True,
            include_prev_next_rel=True,
        )

        base_nodes = node_parser.get_nodes_from_documents(self.documents)

        print(
            "Done creating base_nodes from documents.",
            f"The text was transformed into a total of {len(base_nodes)} nodes, i.e. chunks.",
            "Now creating and storing VectorStoreIndex. This might take a while",
        )
        self.n_base_nodes = len(base_nodes)  # mlflow logging

        # This will create all Chroma DB blobs and store them in /chroma_db
        my_vector_index = VectorStoreIndex(
            base_nodes,
            service_context=service_context,
            storage_context=storage_context,
            show_progress=True,
        )

        # This will store Chroma DB config files (.jsons), required to
        # load it again later
        my_vector_index.storage_context.persist(storage_folder)
        print("Done creating and storing Chroma DB artifacts")

    def upload_all_files_in_folder(
        self,
        bucket,
        folder_name,
        current_ts,
        gcp_subfolder="file-embeddings",
        hash_folder=None,
    ) -> None:
        """Upload all files in a folder, excluding subfolder

        Args:
            bucket (storage_client.bucket): Bucket to which files are uploaded
            folder_name (pathlib.Path): pathlib.Path Object to storage_folder
            current_ts (str): Timestamp string, will be used to name a subfolder
            gcp_subfolder (str): GCS subfolder in which new timestamp folder
                should be created
            hash_folder (pathlib.Path): Path object to the hash-folder that
                Chroma DB creates inside the chroma_db upon creating the vector DB

        Return:
            None, will upload all files in folder_name
        """
        for file in folder_name.rglob("*"):
            if file.is_file():
                # Calculate the relative path within the local folder
                relative_path = file.relative_to(folder_name)

                # Construct the destination GCS object name
                if hash_folder:
                    gcs_object_name = f"{gcp_subfolder}/{current_ts}/{hash_folder.name}/{relative_path}"
                else:
                    gcs_object_name = f"{gcp_subfolder}/{current_ts}/{relative_path}"

                # Ensure forward slashes and remove any double slashes
                gcs_object_name = gcs_object_name.replace("\\", "/").replace("//", "/")

                # Upload the file to GCS
                blob = bucket.blob(gcs_object_name)
                blob.upload_from_filename(str(file))
                print(f"Uploaded {file} to gs://{self.bucket_name}/{gcs_object_name}")

    def delete_db_artifacts(self, folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)

            if os.path.isfile(item_path):
                os.remove(item_path)

            elif os.path.isdir(item_path):
                self.delete_db_artifacts(item_path)
                os.rmdir(item_path)

    def upload_db_files_to_gcs(self) -> None:
        chroma_folder = Path(f"./chroma_db/{self.file_id}")

        # Upload metadata
        metadata = {"is_image": self.is_image, "username": self.username}
        self.gcs_handler.upload_to_gcs(
            self.bucket_name,
            {
                "metadata": (
                    metadata,
                    f"{self.gcs_subfolder}/{self.file_id}/metadata.json",
                )
            },
        )

        # Upload Chroma DB files
        files_to_upload = {}
        for file in chroma_folder.rglob("*"):
            if file.is_file():
                relative_path = file.relative_to(chroma_folder)
                gcs_object_name = f"{self.gcs_subfolder}/{self.file_id}/{relative_path}"
                files_to_upload[str(relative_path)] = (str(file), gcs_object_name)

        self.gcs_handler.upload_to_gcs(self.bucket_name, files_to_upload)

        print(
            f"Successfully uploaded all Chroma DB files and metadata to bucket "
            f"{self.bucket_name}/{self.gcs_subfolder}/{self.file_id}"
        )
