import json
import os

import cv2
import fitz  # PyMuPDF
import numpy as np
import torch
from llama_index.core import (
    Document,
    Node,
    ServiceContext,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from transformers import CLIPModel, CLIPProcessor


class MultimodalProcessor:
    def __init__(self, configs):
        self.configs = configs
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.text_chunk_size = self.configs.chatbot.chunk_size_limit
        self.text_chunk_overlap = self.configs.chatbot.max_chunk_overlap
        self.image_embedding_size = 512  # CLIP's default embedding size
        self.llm_model = self._init_llm_model()
        self.embedding_model = self._init_embedding_model()

    def _init_llm_model(self) -> AzureOpenAI:
        return AzureOpenAI(
            model=self.configs.azure_llm.azure_llm_model_name,
            deployment_name=self.configs.azure_llm.azure_llm_deployment,
            api_key=self.configs.azure_llm.azure_llm_api_key,
            azure_endpoint=self.configs.azure_llm.azure_llm_endpoint,
            api_version=self.configs.azure_llm.azure_llm_api_version,
        )

    def _init_embedding_model(self) -> AzureOpenAIEmbedding:
        return AzureOpenAIEmbedding(
            model=self.configs.azure_embedding.azure_embedding_model_name,
            deployment_name=self.configs.azure_embedding.azure_embedding_deployment,
            api_key=self.configs.azure_embedding.azure_embedding_api_key,
            azure_endpoint=self.configs.azure_embedding.azure_embedding_endpoint,
            api_version=self.configs.azure_embedding.azure_embedding_api_version,
        )

    def process_and_embed(self, file_id, directory_path, chroma_db, contain_multimedia):
        documents = (
            self.process_pdf_with_multimedia(directory_path)
            if contain_multimedia
            else SimpleDirectoryReader(directory_path).load_data()
        )

        text_nodes = self.process_text_documents(documents)
        image_nodes = (
            self.process_image_documents(documents) if contain_multimedia else []
        )

        all_nodes = text_nodes + image_nodes

        chroma_collection = chroma_db.get_or_create_collection(
            self.configs.chatbot.vector_db_collection_name
        )
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        service_context = ServiceContext.from_defaults(
            llm=self.llm_model, embed_model=self.embedding_model
        )

        index = VectorStoreIndex(
            all_nodes,
            storage_context=storage_context,
            service_context=service_context,
        )

        index.storage_context.persist(f"./chroma_db/{file_id}")

        metadata = {"contain_multimedia": contain_multimedia}
        with open(os.path.join(f"./chroma_db/{file_id}", "metadata.json"), "w") as f:
            json.dump(metadata, f)

    def process_text_documents(self, documents):
        text_documents = [
            doc for doc in documents if doc.metadata.get("type") == "text"
        ]
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=self.text_chunk_size,
            chunk_overlap=self.text_chunk_overlap,
        )
        return node_parser.get_nodes_from_documents(text_documents)

    def process_image_documents(self, documents):
        image_documents = [
            doc for doc in documents if doc.metadata.get("type") == "image"
        ]
        image_nodes = []
        for doc in image_documents:
            embedding = np.array(doc.metadata["embedding_chunk"])
            image_nodes.append(
                Node(
                    text=doc.text,
                    embedding=embedding,
                    metadata={
                        "page": doc.metadata["page"],
                        "type": "image",
                        "image_index": doc.metadata["image_index"],
                    },
                )
            )
        return image_nodes

    def process_pdf_with_multimedia(self, directory_path):
        documents = []
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(directory_path, filename)
                pdf_document = fitz.open(file_path)

                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    text = page.get_text()

                    if text.strip():
                        documents.append(
                            Document(
                                text=text,
                                metadata={"page": page_num + 1, "type": "text"},
                            )
                        )

                    images = page.get_images(full=True)
                    for img_index, img in enumerate(images):
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]

                        nparr = np.frombuffer(image_bytes, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        inputs = self.clip_processor(
                            images=image,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                        )
                        with torch.no_grad():
                            image_features = self.clip_model.get_image_features(
                                **inputs
                            )

                        img_doc = Document(
                            text=f"Image on page {page_num + 1}",
                            metadata={
                                "page": page_num + 1,
                                "type": "image",
                                "image_index": img_index,
                                "embedding_chunk": image_features.numpy()
                                .flatten()
                                .tolist(),
                            },
                        )
                        documents.append(img_doc)

                pdf_document.close()
        return documents
