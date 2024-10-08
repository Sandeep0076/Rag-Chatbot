import os
from typing import Optional

from fastapi import Depends, HTTPException, status
from jose import JWTError, jwt

from rtl_rag_chatbot_api.oauth.get_public_key import get_public_key

# token_url
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl=os.getenv("TOKEN_URL"))

oauth2_scheme = "test"


async def get_current_user(token: str = Depends(oauth2_scheme)) -> Optional[str]:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        header = jwt.get_unverified_header(token)
        public_key = get_public_key(header["kid"])
        # AUDIENCE = CLIENT_ID in single tenant apps
        payload = jwt.decode(
            token, public_key, algorithms=["RS256"], audience=os.getenv("CLIENT_ID")
        )
        print(payload)
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        return username
    except JWTError as error:
        print(error)
        raise credentials_exception
