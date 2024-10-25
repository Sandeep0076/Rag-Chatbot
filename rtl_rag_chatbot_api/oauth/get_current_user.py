import os
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from rtl_rag_chatbot_api.oauth.get_public_key import get_public_key

# token_url
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl=os.getenv("TOKEN_URL"))

dev_env = os.getenv("DEV")


def get_predefined_user():
    if dev_env == "true":
        return return_fake_email
    return OAuth2PasswordBearer(tokenUrl=os.getenv("TOKEN_URL"))


def return_fake_email():
    return "rag-api-test@netrtl.com"


oauth2_scheme = get_predefined_user()


async def get_current_user(token: str = Depends(oauth2_scheme)) -> Optional[str]:
    if dev_env == "false" or dev_env is None:
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            header = jwt.get_unverified_header(token)
            public_key = get_public_key(header["kid"])
            payload = jwt.decode(
                token, public_key, algorithms=["RS256"], audience=os.getenv("CLIENT_ID")
            )
            username: str = payload.get("sub")
            if username is None:
                raise credentials_exception
            return username
        except JWTError:
            raise credentials_exception
    return token
