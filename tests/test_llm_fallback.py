"""
Tests for LLM fallback functionality.

This module tests the automatic fallback mechanism that switches to a secondary
LLM when the primary LLM fails for more than the configured timeout.
"""

import asyncio
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from temporalio import activity
from temporalio.testing import ActivityEnvironment

from shared.llm_manager import LLMManager


@pytest.fixture
def mock_env_vars():
    """Set up test environment variables."""
    env_vars = {
        "LLM_MODEL": "openai/gpt-4",
        "LLM_KEY": "test-primary-key",
        "LLM_FALLBACK_MODEL": "anthropic/claude-3",
        "LLM_FALLBACK_KEY": "test-fallback-key",
        "LLM_FALLBACK_TIMEOUT_MINUTES": "2",
        "LLM_RECOVERY_CHECK_INTERVAL_MINUTES": "5",
    }
    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.mark.asyncio
async def test_llm_manager_initialization(mock_env_vars):
    """Test that LLMManager initializes with correct configuration."""
    manager = LLMManager()

    assert manager.primary_model == "openai/gpt-4"
    assert manager.primary_key == "test-primary-key"
    assert manager.fallback_model == "anthropic/claude-3"
    assert manager.fallback_key == "test-fallback-key"
    assert manager.fallback_timeout_minutes == 2
    assert manager.recovery_check_interval_minutes == 5
    assert not manager.using_fallback
    assert manager.primary_failure_start is None


@pytest.mark.asyncio
async def test_successful_primary_llm_call(mock_env_vars):
    """Test successful call to primary LLM."""
    manager = LLMManager()

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]

    with patch("shared.llm_manager.completion", return_value=mock_response) as mock_completion:
        messages = [{"role": "user", "content": "Test message"}]
        response = await manager.call_llm(messages)

        assert response == mock_response
        assert not manager.using_fallback
        assert manager.primary_failure_start is None
        mock_completion.assert_called_once()


@pytest.mark.asyncio
async def test_fallback_after_timeout(mock_env_vars):
    """Test that system switches to fallback after timeout period."""
    manager = LLMManager()

    # Simulate primary failure
    manager.primary_failure_start = datetime.now() - timedelta(minutes=3)

    mock_fallback_response = MagicMock()
    mock_fallback_response.choices = [MagicMock(message=MagicMock(content="Fallback response"))]

    with patch("shared.llm_manager.completion") as mock_completion:
        # First call fails (primary), second succeeds (fallback)
        mock_completion.side_effect = [
            Exception("Primary LLM failed"),
            mock_fallback_response,
        ]

        messages = [{"role": "user", "content": "Test message"}]
        response = await manager.call_llm(messages)

        assert response == mock_fallback_response
        assert manager.using_fallback
        assert mock_completion.call_count == 2

        # Verify fallback was called with correct model
        fallback_call = mock_completion.call_args_list[1]
        assert fallback_call[1]["model"] == "anthropic/claude-3"
        assert fallback_call[1]["api_key"] == "test-fallback-key"


@pytest.mark.asyncio
async def test_no_fallback_before_timeout(mock_env_vars):
    """Test that system doesn't switch to fallback before timeout."""
    manager = LLMManager()

    # Simulate recent primary failure (less than timeout)
    manager.primary_failure_start = datetime.now() - timedelta(seconds=30)

    with patch("shared.llm_manager.completion") as mock_completion:
        mock_completion.side_effect = Exception("Primary LLM failed")

        messages = [{"role": "user", "content": "Test message"}]

        with pytest.raises(Exception, match="Primary LLM failed"):
            await manager.call_llm(messages)

        assert not manager.using_fallback
        mock_completion.assert_called_once()


@pytest.mark.asyncio
async def test_recovery_to_primary(mock_env_vars):
    """Test recovery back to primary LLM when it becomes available."""
    manager = LLMManager()

    # Start in fallback mode
    manager.using_fallback = True
    manager.last_recovery_check = datetime.now() - timedelta(minutes=10)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="OK"))]

    with patch("shared.llm_manager.completion", return_value=mock_response) as mock_completion:
        messages = [{"role": "user", "content": "Test message"}]
        response = await manager.call_llm(messages)

        assert response == mock_response
        assert not manager.using_fallback  # Should have recovered
        assert manager.primary_failure_start is None

        # Should have made health check and actual call
        assert mock_completion.call_count == 2


@pytest.mark.asyncio
async def test_both_llms_fail(mock_env_vars):
    """Test error handling when both primary and fallback LLMs fail."""
    manager = LLMManager()

    # Simulate primary failure timeout
    manager.primary_failure_start = datetime.now() - timedelta(minutes=3)

    with patch("shared.llm_manager.completion") as mock_completion:
        mock_completion.side_effect = Exception("All LLMs failed")

        messages = [{"role": "user", "content": "Test message"}]

        with pytest.raises(Exception, match="Both primary and fallback LLMs failed"):
            await manager.call_llm(messages)

        assert manager.using_fallback
        assert mock_completion.call_count == 2


@pytest.mark.asyncio
async def test_no_fallback_configured():
    """Test behavior when no fallback LLM is configured."""
    with patch.dict(os.environ, {"LLM_MODEL": "openai/gpt-4", "LLM_KEY": "test-key"}):
        manager = LLMManager()

        assert manager.fallback_model is None

        with patch("shared.llm_manager.completion") as mock_completion:
            mock_completion.side_effect = Exception("Primary LLM failed")

            messages = [{"role": "user", "content": "Test message"}]

            with pytest.raises(Exception, match="Primary LLM failed"):
                await manager.call_llm(messages)

            assert not manager.using_fallback
            mock_completion.assert_called_once()


@pytest.mark.asyncio
async def test_get_status(mock_env_vars):
    """Test the get_status method."""
    manager = LLMManager()

    # Test initial status
    status = manager.get_status()
    assert status["current_model"] == "openai/gpt-4"
    assert not status["using_fallback"]
    assert status["primary_failure_start"] is None
    assert status["fallback_configured"]

    # Test status during failure
    manager.primary_failure_start = datetime.now()
    manager.using_fallback = True
    status = manager.get_status()
    assert status["current_model"] == "anthropic/claude-3"
    assert status["using_fallback"]
    assert status["primary_failure_start"] is not None
    assert status["failure_duration_seconds"] is not None


@pytest.mark.asyncio
async def test_integration_with_tool_activities(mock_env_vars):
    """Test integration of LLMManager with ToolActivities."""
    from activities.tool_activities import ToolActivities
    from models.data_types import ToolPromptInput

    # Create activity environment for testing
    env = ActivityEnvironment()

    activities = ToolActivities()

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content='{"result": "success"}'))
    ]

    with patch.object(activities.llm_manager, "call_llm", return_value=mock_response):
        input_data = ToolPromptInput(
            prompt="Test prompt",
            context_instructions="Test context"
        )

        result = await env.run(activities.agent_toolPlanner, input_data)
        assert result == {"result": "success"}