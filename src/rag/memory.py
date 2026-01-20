"""
Conversation Memory for follow-up questions.

Maintains conversation history and context for multi-turn interactions.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Message:
    """A single message in the conversation."""
    role: str  # 'user', 'assistant', or 'system'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dict for Ollama API."""
        return {"role": self.role, "content": self.content}


class ConversationMemory:
    """
    Manages conversation history for multi-turn RAG interactions.

    Features:
    - Configurable history length
    - Context summarization for long conversations
    - Model/topic tracking for contextual retrieval
    """

    def __init__(
        self,
        max_messages: int = 20,
        max_context_tokens: int = 2000,
    ):
        self.max_messages = max_messages
        self.max_context_tokens = max_context_tokens
        self.messages: list[Message] = []
        self.current_models: list[str] = []  # Models being discussed
        self.current_vendor: Optional[str] = None

    def add_user_message(
        self,
        content: str,
        models: Optional[list[str]] = None,
        vendor: Optional[str] = None,
    ) -> None:
        """
        Add a user message to history.

        Args:
            content: The user's message.
            models: Camera models mentioned in the message.
            vendor: Vendor context.
        """
        self.messages.append(Message(
            role="user",
            content=content,
            metadata={"models": models, "vendor": vendor},
        ))

        # Update context tracking
        if models:
            for model in models:
                if model not in self.current_models:
                    self.current_models.append(model)

        if vendor:
            self.current_vendor = vendor

        self._trim_history()

    def add_assistant_message(
        self,
        content: str,
        sources: Optional[list[str]] = None,
    ) -> None:
        """
        Add an assistant response to history.

        Args:
            content: The assistant's response.
            sources: Documents cited in the response.
        """
        self.messages.append(Message(
            role="assistant",
            content=content,
            metadata={"sources": sources},
        ))

        self._trim_history()

    def add_system_message(self, content: str) -> None:
        """Add a system message."""
        self.messages.append(Message(
            role="system",
            content=content,
        ))

    def get_messages(self, include_system: bool = True) -> list[dict]:
        """
        Get messages formatted for Ollama API.

        Args:
            include_system: Whether to include system messages.

        Returns:
            List of message dicts with 'role' and 'content'.
        """
        messages = []
        for msg in self.messages:
            if msg.role == "system" and not include_system:
                continue
            messages.append(msg.to_dict())
        return messages

    def get_history_string(self, last_n: int = 5) -> str:
        """
        Get recent history as a formatted string.

        Args:
            last_n: Number of recent exchanges to include.

        Returns:
            Formatted conversation history.
        """
        recent = self.messages[-(last_n * 2):]  # Approximate exchanges

        lines = []
        for msg in recent:
            if msg.role == "user":
                lines.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                # Truncate long responses
                content = msg.content
                if len(content) > 300:
                    content = content[:300] + "..."
                lines.append(f"Assistant: {content}")

        return "\n\n".join(lines)

    def get_context_models(self) -> list[str]:
        """
        Get models currently being discussed.

        Returns:
            List of model numbers from recent conversation.
        """
        return self.current_models.copy()

    def get_context_vendor(self) -> Optional[str]:
        """Get the current vendor context."""
        return self.current_vendor

    def set_vendor_context(self, vendor: str) -> None:
        """Set the vendor filter context."""
        self.current_vendor = vendor

    def clear_vendor_context(self) -> None:
        """Clear the vendor filter."""
        self.current_vendor = None

    def clear_model_context(self) -> None:
        """Clear tracked models."""
        self.current_models = []

    def clear(self) -> None:
        """Clear all conversation history."""
        self.messages = []
        self.current_models = []
        self.current_vendor = None

    def _trim_history(self) -> None:
        """Trim history to max_messages."""
        if len(self.messages) > self.max_messages:
            # Keep system messages and recent history
            system_msgs = [m for m in self.messages if m.role == "system"]
            other_msgs = [m for m in self.messages if m.role != "system"]

            # Keep most recent
            keep_count = self.max_messages - len(system_msgs)
            other_msgs = other_msgs[-keep_count:]

            self.messages = system_msgs + other_msgs

    def get_summary(self) -> dict:
        """
        Get a summary of the conversation state.

        Returns:
            Dict with conversation metadata.
        """
        return {
            "message_count": len(self.messages),
            "current_models": self.current_models,
            "current_vendor": self.current_vendor,
            "user_messages": sum(1 for m in self.messages if m.role == "user"),
            "assistant_messages": sum(1 for m in self.messages if m.role == "assistant"),
        }

    def is_followup(self, query: str) -> bool:
        """
        Detect if a query is likely a follow-up question.

        Args:
            query: The user's query.

        Returns:
            True if this appears to be a follow-up.
        """
        if not self.messages:
            return False

        # Check for pronouns/references that suggest follow-up
        followup_indicators = [
            "it", "its", "they", "them", "this", "that", "these", "those",
            "the same", "also", "too", "as well", "what about", "how about",
            "and the", "another", "other", "more", "else",
        ]

        query_lower = query.lower()
        return any(indicator in query_lower for indicator in followup_indicators)


class SessionManager:
    """
    Manages multiple conversation sessions.

    Useful for Streamlit or multi-user scenarios.
    """

    def __init__(self):
        self.sessions: dict[str, ConversationMemory] = {}

    def get_session(self, session_id: str) -> ConversationMemory:
        """
        Get or create a conversation session.

        Args:
            session_id: Unique session identifier.

        Returns:
            ConversationMemory instance for this session.
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationMemory()
        return self.sessions[session_id]

    def clear_session(self, session_id: str) -> None:
        """Clear a specific session."""
        if session_id in self.sessions:
            self.sessions[session_id].clear()

    def delete_session(self, session_id: str) -> None:
        """Delete a session entirely."""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        return list(self.sessions.keys())
