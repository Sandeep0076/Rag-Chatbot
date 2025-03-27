from unittest.mock import MagicMock, patch

import pytest

from workflows.msgraph import GRAPH_API_ENDPOINT, is_user_account_enabled


@pytest.fixture
def mock_get_access_token():
    with patch("workflows.msgraph.get_access_token") as mock_token:
        mock_token.return_value = "mocked_token"
        yield mock_token


@pytest.fixture
def mock_requests_get():
    with patch("requests.get") as mock_get:
        yield mock_get


@pytest.fixture
def mock_log():
    with patch("workflows.msgraph.log") as mock_log:
        yield mock_log


def test_is_user_account_enabled_success(mock_get_access_token, mock_requests_get):
    """"""
    # mock a successful API response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "accountEnabled": True,
        "displayName": "Test User",
        "mail": "testuser@example.com",
    }

    mock_requests_get.return_value = mock_response

    # call the function under test
    result = is_user_account_enabled("testuser@example.com")

    # Assertions
    assert result is True
    mock_requests_get.assert_called_once_with(
        f"{GRAPH_API_ENDPOINT}/users?$filter=mail eq 'testuser@example.com'&$select=accountEnabled,displayName,mail",
        headers={
            "Authorization": "Bearer mocked_token",
            "Content-Type": "application/json",
        },
    )


def test_is_user_account_enabled_disabled_account(
    mock_get_access_token, mock_requests_get
):
    """"""
    # mock an API response where accountEnabled is False
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "accountEnabled": False,
        "displayName": "Test User",
        "mail": "testuser@example.com",
    }

    mock_requests_get.return_value = mock_response

    result = is_user_account_enabled("testuser@example.com")
    assert result is False


def test_is_user_account_enabled_failure(
    mock_get_access_token, mock_requests_get, mock_log
):
    """"""
    # mock an API failure response
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = "Not Found"

    mock_requests_get.return_value = mock_response

    # Call the function under test
    result = is_user_account_enabled("nonexistentuser@example.com")

    # Assertions
    assert result is None
    mock_log.error.assert_any_call(
        "Failed to fetch user details for nonexistentuser@example.com: 404, Not Found"
    )
    mock_requests_get.assert_called_once()
