from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicNumbers
from cryptography.hazmat.primitives import hashes
import base64
import json

def construct_rsa_key(jwk):
    exponent = base64.urlsafe_b64decode(jwk['e'] + '==')
    modulus = base64.urlsafe_b64decode(jwk['n'] + '==')
    public_numbers = RSAPublicNumbers(
        int.from_bytes(exponent, byteorder='big'),
        int.from_bytes(modulus, byteorder='big')
    )
    public_key = public_numbers.public_key(default_backend())
    pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return pem