import logging
import os

from cryptography.fernet import Fernet


def get_encryption_key():
    """
    Retrieve the encryption key from environment variables.
    """
    key = os.environ.get("ENCRYPTION_KEY")
    if not key:
        raise ValueError("Encryption key not found in environment variables")
    return key.encode()


def encrypt_file(file_path):
    """
    This function reads the content of the file specified by file_path,
    encrypts it using a key obtained from get_encryption_key(), and saves
    the encrypted data to a new file with '.encrypted' appended to the original filename.

    Args:
        file_path (str): The path to the file to be encrypted.

    Returns:
        str: The path to the encrypted file.
    """
    key = get_encryption_key()
    fernet = Fernet(key)

    with open(file_path, "rb") as file:
        file_data = file.read()

    encrypted_data = fernet.encrypt(file_data)

    original_filename = os.path.basename(file_path)
    encrypted_file_path = os.path.join(
        os.path.dirname(file_path), f"{original_filename}.encrypted"
    )
    with open(encrypted_file_path, "wb") as encrypted_file:
        encrypted_file.write(encrypted_data)

    return encrypted_file_path


def decrypt_file(encrypted_file_path):
    """
    Decrypt a file that was encrypted using the encrypt_file function.

    Args:
        encrypted_file_path (str): The path to the encrypted file.

    Returns:
        str: The path to the decrypted file.
    """
    key = get_encryption_key()
    fernet = Fernet(key)

    with open(encrypted_file_path, "rb") as enc_file:
        encrypted_data = enc_file.read()

    decrypted_data = fernet.decrypt(encrypted_data)

    # Remove '.encrypted' from the end of the filename
    decrypted_file_path = (
        encrypted_file_path[:-10]
        if encrypted_file_path.endswith(".encrypted")
        else encrypted_file_path
    )
    with open(decrypted_file_path, "wb") as dec_file:
        dec_file.write(decrypted_data)
    logging.info(f"Decrypted file to {decrypted_file_path}")
    return decrypted_file_path
