import os
from datetime import datetime
from pathlib import Path
from typing import List
from chromadb.config import Settings
import chromadb
from google.cloud import storage
from llama_index.core import (ServiceContext, SimpleDirectoryReader,
                              VectorStoreIndex)
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
        gcp_project,
        bucket_name,
        text_data_folder_path,
        gcs_subfolder="pdf-embeddings",
        file_id=None
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
        self.documents = self._create_list_of_documents()

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

    def _create_timestamp_folder_string(self):
        """Create current time stamp"""
        # Get current date and time
        current_datetime = datetime.now()

        # Format the datetime object into yyyymmdd_hhmmss format
        timestamp_string = current_datetime.strftime("%Y%m%d_%H%M%S")

        return timestamp_string

    def create_and_store_index(
        self,
        storage_folder: str = "./chroma_db",
        collection_name: str = "RAG_CHATBOT",
        chunk_size: int = 400,
        chunk_overlap: int = 40,
    ) -> None:
        """Create a Chroma Vector DB and store artifacts on local

        This function creates all artifacts that make up a Chroma Vector DB
        and stores them in a local folder called storage_folder.

        Args:
            storage_folder (str): Folder where Chroma DB artifacts are stored
                Note that the following file types and folder structure inside
                of storage_folder will be created by the code:
                - storage_folder
                    - hash_folder
                        - file1.bin
                        - file2.bin
                        -...
                    - chroma.sqlite3
                    - doctore.json
                    - ... (other .json files)
            collection_name (str): Name for Chroma DB internals, how collection
                should be called. Will be required when loading the blobs to
                re-instantiate the Chroma DB again
            chunk_size (int):
            chunk_overlap (int):

        Return:
            None, will store Chroma DB artifact in storage_folder
        """
        db = chromadb.PersistentClient(path=storage_folder,settings=Settings(
                allow_reset=True,
                is_persistent=True
            ))
        chroma_collection = db.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
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
        gcp_subfolder="pdf-embeddings",
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
        # Iterate over all items in the folder
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            # If it's a file, delete it
            if os.path.isfile(item_path):
                os.remove(item_path)
            # If it's a directory, recursively delete its contents
            elif os.path.isdir(item_path):
                self.delete_db_artifacts(item_path)
                # Once the contents are deleted, remove the empty directory
                os.rmdir(item_path)

    def upload_db_files_to_gcs(self) -> None:
        storage_client = storage.Client(self.gcp_project)
        bucket = storage_client.bucket(self.bucket_name)
        chroma_folder = Path(f"./chroma_db/{self.file_id}")

        self.upload_all_files_in_folder(
            bucket=bucket,
            folder_name=chroma_folder,
            current_ts=self.file_id,
            gcp_subfolder=self.gcs_subfolder
        )

        hash_folder = [item for item in chroma_folder.iterdir() if item.is_dir()][0]

        self.upload_all_files_in_folder(
            bucket=bucket,
            folder_name=hash_folder,
            current_ts=self.file_id,
            gcp_subfolder=self.gcs_subfolder,
            hash_folder=hash_folder,
        )

        print(f"Successfully uploaded all Chroma DB files to bucket {self.bucket_name}/{self.gcs_subfolder}/{self.file_id}")