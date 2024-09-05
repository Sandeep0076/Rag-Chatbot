import hashlib
import os

from fastapi import UploadFile

from rtl_rag_chatbot_api.common.encryption_utils import encrypt_file


class FileHandler:
    def __init__(self, configs, gcs_handler):
        self.configs = configs
        self.gcs_handler = gcs_handler

    def calculate_file_hash(self, file_content):
        return hashlib.md5(file_content).hexdigest()

    async def process_file(self, file: UploadFile, file_id: str, is_image: bool):
        original_filename = file.filename
        file_content = await file.read()
        file_hash = self.calculate_file_hash(file_content)

        existing_file_id = self.gcs_handler.find_existing_file_by_hash(file_hash)

        if existing_file_id:
            return {
                "file_id": existing_file_id,
                "is_image": is_image,
                "message": "File with identical content already exists. Embeddings downloaded.",
                "status": "existing",
            }

        temp_file_path = f"temp_{file_id}_{os.path.splitext(original_filename)[1]}"

        with open(temp_file_path, "wb") as buffer:
            buffer.write(file_content)

        encrypted_file_path = encrypt_file(temp_file_path)

        destination_blob_name = f"files-raw/{file_id}/{original_filename}.encrypted"
        self.gcs_handler.upload_to_gcs(
            self.configs.gcp_resource.bucket_name,
            {
                "file": (encrypted_file_path, destination_blob_name),
                "metadata": (
                    {"is_image": is_image, "file_hash": file_hash},
                    f"files-raw/{file_id}/metadata.json",
                ),
            },
        )

        os.remove(temp_file_path)
        os.remove(encrypted_file_path)

        return {
            "file_id": file_id,
            "is_image": is_image,
            "message": "File uploaded, encrypted, and processed successfully",
            "status": "new",
        }

    def download_existing_file(self, file_id: str):
        chroma_db_path = f"./chroma_db/{file_id}"
        os.makedirs(chroma_db_path, exist_ok=True)

        try:
            self.gcs_handler.download_files_from_folder_by_id(file_id)
            return True
        except Exception as e:
            print(f"Error downloading embeddings: {str(e)}")
            return False
