import json
import os

import pytesseract
import torch
from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class MultimodalProcessor:
    def __init__(self, configs):
        self.configs = configs
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    def process_and_embed(self, file_id, file_path, chroma_db, contain_multimedia):
        # Process text
        text_documents = SimpleDirectoryReader(file_path).load_data()

        # Process images
        image_documents = self.process_images(file_path)

        # Combine text and image documents
        all_documents = text_documents + image_documents

        # Create nodes
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=self.configs.chatbot.chunk_size_limit,
            chunk_overlap=self.configs.chatbot.max_chunk_overlap,
        )
        nodes = node_parser.get_nodes_from_documents(all_documents)

        # Create and store index
        chroma_collection = chroma_db.get_or_create_collection(
            self.configs.chatbot.vector_db_collection_name
        )
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            service_context=self.configs.service_context,
        )

        index.storage_context.persist(f"./chroma_db/{file_id}")

        # Save metadata
        metadata = {"contain_multimedia": contain_multimedia}
        with open(os.path.join(f"./chroma_db/{file_id}", "metadata.json"), "w") as f:
            json.dump(metadata, f)

    def process_images(self, file_path):
        image_documents = []
        for filename in os.listdir(file_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
                img_path = os.path.join(file_path, filename)
                image = Image.open(img_path)

                # Extract text from image using OCR
                ocr_text = pytesseract.image_to_string(image)

                # Generate CLIP embeddings
                inputs = self.clip_processor(
                    images=image, return_tensors="pt", padding=True, truncation=True
                )
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)

                # Create a document with image metadata, OCR text, and CLIP embeddings
                doc = Document(
                    text=ocr_text,
                    metadata={
                        "file_name": filename,
                        "clip_embedding": image_features.numpy().tolist(),
                    },
                )
                image_documents.append(doc)

        return image_documents
