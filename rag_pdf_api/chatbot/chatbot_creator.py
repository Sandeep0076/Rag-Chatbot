import os

import chromadb
import openai
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

from rag_pdf_api.chatbot.gcs_handler import GCSHandler

# Set up Azure OpenAI API keys and endpoints
os.environ["AZURE_OPENAI_API_KEY"] = os.environ.get("AZURE_OPENAI_LLM_API_KEY", "")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.environ.get("AZURE_OPENAI_LLM_ENDPOINT", "")

os.environ["AZURE_OPENAI_API_KEY"] = os.environ.get(
    "AZURE_OPENAI_EMBEDDING_API_KEY", ""
)
os.environ["AZURE_OPENAI_ENDPOINT"] = os.environ.get(
    "AZURE_OPENAI_EMBEDDING_ENDPOINT", ""
)


class Chatbot:
    """
    Class to set up an in-memory vector database for chatbot functionality.

    Attributes:
    configs: Configuration object containing necessary settings.
    _index: Index object created from the vector store.
    _vanilla_llm: Plain LLM instance for generating answers.
    retriever: Retriever instance for fetching similar documents.
    query_engine: Query engine instance for processing queries.
    chat_engine: ChatGPT instance for generating chat responses.
    """

    def __init__(self, configs):
        """
        Initializes the Chatbot class.

        Parameters:
        configs (object): Configuration object containing necessary settings.
        """
        self.configs = configs
        self._index = self._create_index()
        self._vanilla_llm = self._create_llm_instance_only()
        self.retriever = self._create_retriever()
        self.query_engine = self._create_query_engine()
        self.chat_engine = self._create_chat_gpt_instance()

    def _create_index(self, chroma_folder_path: str = "chroma_db"):
        """
        Creates a vector store index from stored documents.

        Parameters:
        chroma_folder_path (str): Path to the folder containing the vector database files.

        Returns:
        VectorStoreIndex: Index object created from the vector store.
        """
        llm_llama = AzureOpenAI(
            api_key=self.configs.azure_llm.azure_llm_api_key,
            azure_endpoint=self.configs.azure_llm.azure_llm_endpoint,
            azure_deployment=self.configs.azure_llm.azure_llm_deployment,
            api_version=self.configs.azure_llm.azure_llm_api_version,
            system_prompt=self.configs.chatbot.system_prompt_rag_llm,
        )

        embedding_function_llama = AzureOpenAIEmbedding(
            api_key=self.configs.azure_embedding.azure_embedding_api_key,
            azure_endpoint=self.configs.azure_embedding.azure_embedding_endpoint,
            model=self.configs.azure_embedding.azure_embedding_model_name,
            deployment_name=self.configs.azure_embedding.azure_embedding_deployment,
            api_version=self.configs.azure_embedding.azure_embedding_api_version,
        )
        # functionalities previously handled by PromptHelper have been integrated into ServiceContext.
        db = chromadb.PersistentClient(path=chroma_folder_path)
        chroma_collection = db.get_collection(
            self.configs.chatbot.vector_db_collection_name
        )
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        service_context = ServiceContext.from_defaults(
            llm=llm_llama,
            embed_model=embedding_function_llama,
            chunk_size=self.configs.chatbot.chunk_size_limit,
            chunk_overlap=self.configs.chatbot.max_chunk_overlap,
        )

        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            service_context=service_context,
            storage_context=storage_context,
        )

        return index

    def _create_llm_instance_only(self):
        """
        Creates an LLM instance for generating answers without retrieval-augmented generation.

        Returns:
        AzureOpenAI: Plain LLM instance.
        """
        llm_llama = openai.AzureOpenAI(
            api_key=self.configs.azure_llm.azure_llm_api_key,
            azure_endpoint=self.configs.azure_llm.azure_llm_endpoint,
            azure_deployment=self.configs.azure_llm.azure_llm_deployment,
            api_version=self.configs.azure_llm.azure_llm_api_version,
        )

        return llm_llama

    def _create_retriever(self):
        """
        Creates a retriever instance for fetching similar documents.

        Returns:
        Retriever: Retriever instance.
        """
        retriever = self._index.as_retriever(
            similarity_top_k=self.configs.chatbot.n_neighbours
        )
        return retriever

    def _create_query_engine(self):
        """
        Creates a query engine instance for processing queries.

        Returns:
        QueryEngine: Query engine instance.
        """
        query_engine = self._index.as_query_engine(
            vector_store_query_mode="mmr", chat_mode="CONTEXT"
        )
        return query_engine

    def _create_chat_gpt_instance(self):
        """
        Creates a ChatGPT instance for generating chat responses.

        Returns:
        AzureOpenAI: ChatGPT instance.
        """
        client = openai.AzureOpenAI(
            api_key=self.configs.azure_llm.azure_llm_api_key,
            azure_endpoint=self.configs.azure_llm.azure_llm_endpoint,
            api_version=self.configs.azure_llm.azure_llm_api_version,
        )
        return client

    def get_llm_answer(self, query: str) -> str:
        """
        Generates an answer for a given query using a plain LLM instance.

        Parameters:
        query (str): User query.

        Returns:
        str: LLM-generated response.
        """
        messages_prompt = [
            {
                "role": "system",
                "content": self.configs.chatbot.system_prompt_plain_llm,
            },
            {"role": "user", "content": query},
        ]
        completion = self._vanilla_llm.chat.completions.create(
            messages=messages_prompt,
            model=self.configs.azure_llm.azure_llm_deployment,
            temperature=self.configs.llm_hyperparams.temperature,
            max_tokens=self.configs.llm_hyperparams.max_tokens,
        )

        return completion.choices[0].message.content

    def get_answer(self, query: str) -> str:
        """
        Generates an answer for a given query using the query engine.

        Parameters:
        query (str): User query.

        Returns:
        str: Query engine-generated response.
        """
        response = self.query_engine.query(query)
        return response.response

    def get_n_nearest_neighbours(
        self, query: str, n_neighbours: int, unpack_response=False
    ) -> str:
        """
        Retrieves the n nearest neighbors for a given query based on similarity.

        Parameters:
        query (str): User query.
        n_neighbours (int): Number of nearest neighbors to retrieve.
        unpack_response (bool): Whether to unpack the response.

        Returns:
        str: Nearest neighbors.
        """
        retriever = self._index.as_retriever(similarity_top_k=n_neighbours)
        response = retriever.retrieve(query)

        return response

    def _check_message(self, current_prompt: str, history_prompt: list = []) -> str:
        """
        Checks if a given prompt appears in the history and returns an appropriate response.

        Parameters:
        current_prompt (str): Current user prompt.
        history_prompt (list): List of historical prompts.

        Returns:
        str: "TRUE" if the prompt appears in history, otherwise "FALSE".
        """
        temp_history_list = history_prompt.copy()

        if current_prompt in temp_history_list:
            temp_history_list.remove(current_prompt)

        message_text = [
            {
                "role": "user",
                "content": f"Check the following statement."
                "Does this statement refer to a statement from the attached list? "
                "If this is the case, then answer TRUE. "
                "If the statement appears in the list, also return the index of the matching element."
                f"Otherwise answer FALSE. Statement: {current_prompt} "
                f"List: {' '.join(temp_history_list)}",
            }
        ]

        completion = self.chat_engine.chat.completions.create(
            model=self.configs.azure_llm.azure_llm_deployment,
            messages=message_text,
            temperature=self.configs.llm_hyperparams.temperature,
            max_tokens=self.configs.llm_hyperparams.max_tokens,
            top_p=self.configs.llm_hyperparams.top_p,
            frequency_penalty=self.configs.llm_hyperparams.frequency_penalty,
            presence_penalty=self.configs.llm_hyperparams.presence_penalty,
            stop=self.configs.llm_hyperparams.stop,
        )

        if "TRUE" in completion.choices[0].message.content:
            return completion.choices[0].message.content
        else:
            return "FALSE"


def setup_chatbot(configs):
    """
    Sets up the Chatbot instance and retrieves the latest timestamp folder.

    Parameters:
    configs (Config): Configuration object containing necessary settings.

    Returns:
    tuple: A tuple containing the Chatbot instance and the latest timestamp folder.

    The function logs the start of the download process, creates a GCSHandler instance,
    downloads the latest timestamp files, retrieves the latest timestamp folder,
    and returns the Chatbot instance and timestamp folder.
    """
    print("Now downloading latest timestamp files.")
    gcs_handler = GCSHandler(configs)
    gcs_handler.download_latest_timestamp_files()
    timestamp = gcs_handler.get_latest_time_stamp_folder()

    return Chatbot(configs), timestamp
