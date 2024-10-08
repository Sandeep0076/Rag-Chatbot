import os

import requests

from rtl_rag_chatbot_api.oauth.construct_rsa_key import construct_rsa_key


def get_public_key(kid):
    # keys_url
    jwks_url = os.getenv("JWKS_URL")
    response = requests.get(jwks_url)
    jwks = response.json()
    for jwk in jwks["keys"]:
        if jwk["kid"] == kid:
            return construct_rsa_key(jwk)
