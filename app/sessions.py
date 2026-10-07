"""Session store: a session is a persisted conversation (see app/conversations.py).
`SessionStore()` with no arguments is an in-memory database, which keeps tests isolated."""
from app.conversations import ConversationStore


class SessionStore(ConversationStore):
    def create(self) -> str:
        return super().create()
