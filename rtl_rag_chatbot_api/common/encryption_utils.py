import os
from cryptography.fernet import Fernet

# Generate a key
#key = Fernet.generate_key()
# Store the key as an environment variable
#os.environ['ENCRYPTION_KEY'] = key.decode()

def get_encryption_key():
    key = os.environ.get('ENCRYPTION_KEY')
    if not key:
        raise ValueError("Encryption key not found in environment variables")
    return key.encode()

def encrypt_file(file_path):
    key = get_encryption_key()
    fernet = Fernet(key)
    
    with open(file_path, 'rb') as file:
        file_data = file.read()
    
    encrypted_data = fernet.encrypt(file_data)
    
    encrypted_file_path = file_path + '.encrypted'
    with open(encrypted_file_path, 'wb') as encrypted_file:
        encrypted_file.write(encrypted_data)
    
    return encrypted_file_path

def decrypt_file(encrypted_file_path):
    key = get_encryption_key()
    fernet = Fernet(key)
    
    with open(encrypted_file_path, 'rb') as enc_file:
        encrypted_data = enc_file.read()
    
    decrypted_data = fernet.decrypt(encrypted_data)
    
    decrypted_file_path = encrypted_file_path.replace('.encrypted', '')
    with open(decrypted_file_path, 'wb') as dec_file:
        dec_file.write(decrypted_data)
    
    return decrypted_file_path