import logging
import os
import sys

import msal
import requests

TENANT_ID = os.getenv("RTL_TENANT_ID")
CLIENT_ID = os.getenv("RTL_CLIENT_ID")
CLIENT_SECRET = os.getenv("WF_CLIENT_SECRET")
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPE = ["https://graph.microsoft.com/.default"]
GRAPH_API_ENDPOINT = "https://graph.microsoft.com/v1.0"
BATCH_URL = f"{GRAPH_API_ENDPOINT}/$batch"

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


def get_access_token():
    """"""
    app = msal.ConfidentialClientApplication(
        CLIENT_ID, authority=AUTHORITY, client_credential=CLIENT_SECRET
    )
    log.info("Requesting token for client")
    token_response = app.acquire_token_for_client(scopes=SCOPE)
    return token_response.get("access_token")


def is_user_account_enabled(user_email):
    """
    Accesses the Microsoft Graph API to check if the user account is enabled.
    """
    token = get_access_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # graph API query to get user details
    log.info("Sending request to get user details")
    user_response = requests.get(
        # filter follows a particular syntax, https://learn.microsoft.com/en-us/graph/filter-query-parameter?tabs=http
        f"{GRAPH_API_ENDPOINT}/users?$filter=mail eq '{user_email}'&$select=accountEnabled,displayName,mail",
        headers=headers,
    )

    if user_response.status_code == 200:
        log.info(f"Successfully obtained user details for {user_email}")
        user_data = user_response.json()
        if "accountEnabled" in user_data:
            log.info(
                f"User account {user_email} is enabled: {user_data['accountEnabled']}"
            )
            return user_data["accountEnabled"]
    else:
        log.error(
            f"Failed to fetch user details for {user_email}: {user_response.status_code}, {user_response.text}"
        )
        # return None here in case of error, since False would mean that the requested user is not enabled,
        # and thus marked as a deletion candidate
        return None
