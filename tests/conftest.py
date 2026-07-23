"""Shared test configuration and fixtures."""

import os
import pytest
from unittest.mock import MagicMock, patch


# Set a dummy API key so OpenAI client initialises without error in tests
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-key-for-unit-tests")


@pytest.fixture
def mock_llm():
    """Patch ChatOpenAI at the chain module level so tests never hit the real API."""
    with patch("src.qa.chain.ChatOpenAI") as mock_cls:
        mock_instance = MagicMock()
        mock_structured = MagicMock()
        mock_instance.with_structured_output.return_value = mock_structured
        mock_cls.return_value = mock_instance
        yield mock_structured
