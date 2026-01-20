"""
Tests for RAG components.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.prompts import classify_query, format_context, format_poe_data
from src.rag.memory import ConversationMemory, SessionManager


class TestQueryClassification:
    """Tests for query type classification."""

    def test_poe_query(self):
        assert classify_query("What is the power consumption?") == "poe"
        assert classify_query("POE wattage for XNV-8080R") == "poe"
        assert classify_query("Calculate power budget") == "poe"

    def test_accessory_query(self):
        assert classify_query("What mount fits this camera?") == "accessory"
        assert classify_query("Compatible brackets for XNV-8080R") == "accessory"
        assert classify_query("Pendant mount options") == "accessory"

    def test_comparison_query(self):
        assert classify_query("Compare XNV-8080R vs P3265-LVE") == "comparison"
        assert classify_query("Difference between these models") == "comparison"
        assert classify_query("Which camera is better?") == "comparison"

    def test_specification_query(self):
        assert classify_query("What is the resolution?") == "specification"
        assert classify_query("Camera specs") == "specification"
        assert classify_query("Sensor size and lens info") == "specification"

    def test_general_query(self):
        assert classify_query("Tell me about this camera") == "general"
        assert classify_query("How do I configure it?") == "general"


class TestFormatContext:
    """Tests for context formatting."""

    def test_format_single_result(self):
        results = [{
            "content": "XNV-8080R specifications",
            "metadata": {
                "source_file": "datasheet.pdf",
                "page_num": 1,
                "vendor": "hanwha",
                "model_num": "XNV-8080R",
            },
        }]

        context = format_context(results)

        assert "datasheet.pdf" in context
        assert "Page 1" in context
        assert "HANWHA" in context
        assert "XNV-8080R" in context

    def test_format_respects_max_length(self):
        # Create long content
        results = [{
            "content": "A" * 5000,
            "metadata": {"source_file": "test.pdf", "page_num": 1},
        }]

        context = format_context(results, max_length=100)

        # Should truncate
        assert len(context) <= 200  # Some overhead for formatting

    def test_format_multiple_results(self):
        results = [
            {"content": "First result", "metadata": {"source_file": "a.pdf", "page_num": 1}},
            {"content": "Second result", "metadata": {"source_file": "b.pdf", "page_num": 2}},
        ]

        context = format_context(results)

        assert "First result" in context
        assert "Second result" in context
        assert "---" in context  # Separator


class TestFormatPOEData:
    """Tests for POE data formatting."""

    def test_format_single_model(self):
        poe_info = {
            "total_watts": 25.5,
            "by_model": {"XNV-8080R": 25.5},
            "missing": [],
        }

        result = format_poe_data(poe_info)

        assert "XNV-8080R" in result
        assert "25.5W" in result

    def test_format_multiple_models(self):
        poe_info = {
            "total_watts": 38.4,
            "by_model": {"XNV-8080R": 25.5, "P3265-LVE": 12.9},
            "missing": [],
        }

        result = format_poe_data(poe_info)

        assert "38.4W" in result or "TOTAL" in result
        assert "XNV-8080R" in result
        assert "P3265-LVE" in result

    def test_format_with_missing(self):
        poe_info = {
            "total_watts": 25.5,
            "by_model": {"XNV-8080R": 25.5},
            "missing": ["UNKNOWN-MODEL"],
        }

        result = format_poe_data(poe_info)

        assert "Missing" in result or "UNKNOWN-MODEL" in result


class TestConversationMemory:
    """Tests for conversation memory."""

    @pytest.fixture
    def memory(self):
        return ConversationMemory()

    def test_add_messages(self, memory):
        memory.add_user_message("Hello")
        memory.add_assistant_message("Hi there!")

        assert len(memory.messages) == 2
        assert memory.messages[0].role == "user"
        assert memory.messages[1].role == "assistant"

    def test_get_messages(self, memory):
        memory.add_user_message("Test question")
        memory.add_assistant_message("Test answer")

        messages = memory.get_messages()

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Test question"

    def test_model_tracking(self, memory):
        memory.add_user_message("Tell me about XNV-8080R", models=["XNV-8080R"])

        assert "XNV-8080R" in memory.get_context_models()

    def test_vendor_context(self, memory):
        memory.set_vendor_context("hanwha")

        assert memory.get_context_vendor() == "hanwha"

        memory.clear_vendor_context()
        assert memory.get_context_vendor() is None

    def test_is_followup_detection(self, memory):
        # Empty memory - not a followup
        assert not memory.is_followup("What is the resolution?")

        # Add some history
        memory.add_user_message("Tell me about XNV-8080R")
        memory.add_assistant_message("The XNV-8080R is a camera...")

        # Now check followup detection
        assert memory.is_followup("What about its power consumption?")
        assert memory.is_followup("And the mount options?")
        assert not memory.is_followup("Tell me about P3265-LVE")  # Different topic

    def test_history_trimming(self):
        memory = ConversationMemory(max_messages=5)

        # Add more messages than max
        for i in range(10):
            memory.add_user_message(f"Message {i}")

        assert len(memory.messages) <= 5

    def test_clear(self, memory):
        memory.add_user_message("Test")
        memory.set_vendor_context("hanwha")

        memory.clear()

        assert len(memory.messages) == 0
        assert memory.get_context_vendor() is None

    def test_get_summary(self, memory):
        memory.add_user_message("Q1")
        memory.add_assistant_message("A1")
        memory.add_user_message("Q2", models=["XNV-8080R"])

        summary = memory.get_summary()

        assert summary["message_count"] == 3
        assert summary["user_messages"] == 2
        assert summary["assistant_messages"] == 1
        assert "XNV-8080R" in summary["current_models"]


class TestSessionManager:
    """Tests for session management."""

    @pytest.fixture
    def manager(self):
        return SessionManager()

    def test_get_session_creates_new(self, manager):
        session = manager.get_session("user-123")

        assert isinstance(session, ConversationMemory)
        assert "user-123" in manager.list_sessions()

    def test_get_same_session(self, manager):
        session1 = manager.get_session("user-123")
        session1.add_user_message("Hello")

        session2 = manager.get_session("user-123")

        assert session1 is session2
        assert len(session2.messages) == 1

    def test_clear_session(self, manager):
        session = manager.get_session("user-123")
        session.add_user_message("Hello")

        manager.clear_session("user-123")

        assert len(session.messages) == 0

    def test_delete_session(self, manager):
        manager.get_session("user-123")

        manager.delete_session("user-123")

        assert "user-123" not in manager.list_sessions()


# Integration tests (require Ollama)
def ollama_available():
    try:
        from src.rag.llm_client import OllamaLLM
        llm = OllamaLLM()
        return llm.check_model_available()
    except Exception:
        return False


@pytest.mark.skipif(not ollama_available(), reason="Ollama not available")
class TestLLMClient:
    """Integration tests for LLM client."""

    def test_generate(self):
        from src.rag.llm_client import OllamaLLM

        llm = OllamaLLM()
        response = llm.generate("Say 'hello' and nothing else.")

        assert isinstance(response, str)
        assert len(response) > 0

    def test_generate_with_system(self):
        from src.rag.llm_client import OllamaLLM

        llm = OllamaLLM()
        response = llm.generate(
            "What are you?",
            system="You are a helpful camera expert.",
        )

        assert isinstance(response, str)
