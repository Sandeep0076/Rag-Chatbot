from datetime import datetime, timedelta

from rtl_rag_chatbot_api.common.scheduled_tasks import is_stale_conversation

# Assuming TIME_THRESHOLD is defined somewhere, such as 2 hours
TIME_THRESHOLD = timedelta(hours=2)


# Mock conversation model
class Conversation:
    def __init__(self, updated_at):
        self.updatedAt = updated_at


# Import the method to be tested


def test_is_stale_conversation():
    # Case 1: Conversation updated within the threshold (not stale)
    recent_conversation = Conversation(
        updated_at=(datetime.now() - timedelta(minutes=30))
    )
    assert not is_stale_conversation(
        recent_conversation
    ), "Conversation should not be stale."

    # Case 2: Conversation updated beyond the threshold (stale)
    old_conversation = Conversation(updated_at=(datetime.now() - timedelta(hours=3)))
    assert is_stale_conversation(old_conversation), "Conversation should be stale."
