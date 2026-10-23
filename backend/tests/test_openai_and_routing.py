"""
tests/test_openai_and_routing.py
---------------------------------
Unit tests for OpenAIClient and LLM Router failover with OpenAI.
"""

import pytest
from unittest.mock import MagicMock, patch
from app.llm.openai_client import OpenAIClient
from app.llm import llm_router
from app.utils.error_handler import GroqQuotaExceededError, LLMClientError
from app.api.dependencies import SessionLocal


def test_openai_client_generate_response_mock():
    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance

        mock_choice = MagicMock()
        mock_choice.message.content = "OpenAI response test"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_client_instance.chat.completions.create.return_value = mock_resp

        client = OpenAIClient()
        reply = client.generate_response(
            system_prompt="You are a companion.",
            chat_history=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hey!"},
                {"role": "user", "content": "how are you?"},
            ],
            user_message="how are you?",
        )

        assert reply == "OpenAI response test"
        mock_client_instance.chat.completions.create.assert_called_once()
        sent_messages = mock_client_instance.chat.completions.create.call_args.kwargs["messages"]
        assert [m["content"] for m in sent_messages].count("how are you?") == 1


def test_openai_client_health_check_mock():
    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance

        client = OpenAIClient()
        assert client.health_check() is True

        mock_client_instance.models.list.side_effect = Exception("Auth error")
        assert client.health_check() is False


def test_llm_router_direct_openai():
    db = SessionLocal()
    try:
        mock_client = MagicMock()
        mock_client.generate_response.return_value = "OpenAI direct reply"
        with patch("app.llm.llm_router._get_openai", return_value=mock_client):
            reply = llm_router.route_and_generate(db, "System prompt", [], "Hello")
            assert reply == "OpenAI direct reply"
            assert llm_router.get_current_provider(db) == "openai"
    finally:
        db.close()
