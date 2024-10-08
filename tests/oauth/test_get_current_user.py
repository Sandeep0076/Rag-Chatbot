from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException
from jose import JWTError

from rtl_rag_chatbot_api.oauth.get_current_user import get_current_user

# SEE __init__.py for setting required environment variables

# this is a real token, but artificially constructed to mock the payload for testing
valid_token = (
    "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJzdWIiOiJ0ZXN0dXNlciIsImF1ZCI6InlvdXJfY2xpZW50X2lkIn0."
    "c2lnbmF0dXJl"  # "signature" in base64-encoded format.
)
invalid_token = "your_invalid_token"
payload = {"sub": "testuser", "aud": "your_client_id"}
public_key = "your_public_key"
mock_header = {"alg": "RS256", "kid": "mock_key_id"}  # Include a 'kid'

# mock response for the `requests.get` call
mock_jwks_response = {"keys": [{"kid": "mock_kid", "e": "AQAB", "n": "mock_modulus"}]}


@pytest.mark.asyncio
@patch("requests.get")
@patch(
    "rtl_rag_chatbot_api.oauth.get_public_key.get_public_key", return_value=public_key
)
@patch("jose.jwt.decode", return_value=payload)
@patch("jose.jwt.get_unverified_header", return_value=mock_header)  # Mocking header
@patch(
    "rtl_rag_chatbot_api.oauth.get_current_user.oauth2_scheme", return_value=valid_token
)
async def test_get_current_user_valid_token(
    mock_oauth2_scheme,
    mock_jwt_get_unverified_header,
    mock_jwt_decode,
    mock_get_public_key,
    mock_requests_get,
):
    # Mock the response from `requests.get` for JWKS
    mock_requests_get.return_value = Mock(status_code=200)
    mock_requests_get.return_value.json.return_value = mock_jwks_response

    # Call the function under test
    result = await get_current_user(token=valid_token)

    # Check the result
    assert result == "testuser"
    mock_requests_get.assert_called_once()  # Ensure `requests.get` was called


@pytest.mark.asyncio
@patch("requests.get")
@patch(
    "rtl_rag_chatbot_api.oauth.get_public_key.get_public_key", return_value=public_key
)
@patch("jose.jwt.decode", side_effect=JWTError("Token is invalid"))
@patch("jose.jwt.get_unverified_header", return_value=mock_header)  # Mocking header
@patch(
    "rtl_rag_chatbot_api.oauth.get_current_user.oauth2_scheme",
    return_value=invalid_token,
)
async def test_get_current_user_missing_sub(
    mock_oauth2_scheme,
    mock_jwt_get_unverified_header,
    mock_jwt_decode,
    mock_get_public_key,
    mock_requests_get,
):
    """"""
    # Mock the response from `requests.get` for JWKS
    mock_requests_get.return_value = Mock(status_code=200)
    mock_requests_get.return_value.json.return_value = mock_jwks_response

    invalid_payload = payload.copy()
    # simulate the error
    invalid_payload.pop("sub")

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(token=valid_token)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Could not validate credentials"


@patch(
    "rtl_rag_chatbot_api.oauth.get_public_key.get_public_key", return_value=public_key
)
@patch("jose.jwt.decode", side_effect=JWTError("Token is invalid"))
@patch(
    "rtl_rag_chatbot_api.oauth.get_current_user.oauth2_scheme",
    return_value=invalid_token,
)
async def test_get_current_user_jwt_error(
    mock_oauth2_scheme, mock_jwt_decode, mock_get_public_key
):
    """"""
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(token=invalid_token)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Could not validate credentials"
