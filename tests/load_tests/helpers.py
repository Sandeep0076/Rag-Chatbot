import os
import random


def get_random_file_id_from_file(file_path: str):
    """"""
    if not os.path.exists(file_path):
        return []

    # the file is read every time the function is called, because
    # the tests might be invoked in parallel and the file contents
    # is subject to change
    with open(file_path, "r", encoding="utf-8") as file:
        # read each line, strip newlines, and return as a list
        file_ids = [line.strip() for line in file.readlines()]

    if not file_ids:
        return []

    return random.choice(file_ids)


def get_random_file_id_from_chromadb_folder(chromadb_folder: str = "./chroma_db"):
    """"""
    # read the list of folder names inside the "chromadb" folder
    folder_names = [
        f
        for f in os.listdir(chromadb_folder)
        if os.path.isdir(os.path.join(chromadb_folder, f))
    ]

    if not folder_names:
        raise ValueError("No folders found in the chromadb directory.")

    # randomly select a folder name to use as the file_id
    return random.choice(folder_names)


def get_random_pdf_file(source_dir: str = "tests/resources/"):
    """"""
    # Read the list of files inside the source directory and filter by .pdf extension
    pdf_files = [
        os.path.join(source_dir, f)
        for f in os.listdir(source_dir)
        if os.path.isfile(os.path.join(source_dir, f)) and f.endswith(".pdf")
    ]

    if not pdf_files:
        raise ValueError(f"No PDF files found in the directory {source_dir}.")

    # Randomly select a PDF file from the list
    return random.choice(pdf_files)
