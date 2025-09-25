"""
Tests for LLM fallback functionality.

This module tests the automatic fallback mechanism that switches to a secondary
LLM when the primary LLM fails for more than the configured timeout.
"""

import asyncio
import os
import shutil
import tempfile
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
        "LLM_FALLBACK_DURATION_MINUTES": "2",
        "LLM_RECOVERY_CHECK_INTERVAL_MINUTES": "5",
    }
    with patch.dict(os.environ, env_vars):
        # Reset singleton before each test
        LLMManager._reset_singleton()
        yield env_vars
        # Reset singleton after each test to ensure isolation
        LLMManager._reset_singleton()


@pytest.fixture
def temp_debug_dir():
    """Create a temporary directory for debug files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_llm_manager_initialization(mock_env_vars):
    """Test that LLMManager initializes with correct configuration."""
    manager = LLMManager()

    assert manager.primary_model == "openai/gpt-4"
    assert manager.primary_key == "test-primary-key"
    assert manager.fallback_model == "anthropic/claude-3"
    assert manager.fallback_key == "test-fallback-key"
    assert manager.fallback_duration_minutes == 2
    assert manager.recovery_check_interval_minutes == 5
    assert not manager.using_fallback
    assert manager.primary_failure_time is None


@pytest.mark.asyncio
async def test_successful_primary_llm_call(mock_env_vars):
    """Test successful call to primary LLM."""
    manager = LLMManager()

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]

    with patch(
        "shared.llm_manager.completion", return_value=mock_response
    ) as mock_completion:
        messages = [{"role": "user", "content": "Test message"}]
        response = await manager.call_llm(messages)

        assert response == mock_response
        assert not manager.using_fallback
        assert manager.primary_failure_time is None
        mock_completion.assert_called_once()


@pytest.mark.asyncio
async def test_fallback_after_timeout(mock_env_vars):
    """Test that system switches to fallback after timeout period."""
    manager = LLMManager()

    # Simulate primary failure
    manager.primary_failure_time = datetime.now() - timedelta(minutes=3)

    mock_fallback_response = MagicMock()
    mock_fallback_response.choices = [
        MagicMock(message=MagicMock(content="Fallback response"))
    ]

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
async def test_immediate_fallback_on_primary_failure(mock_env_vars):
    """Test that system immediately switches to fallback on primary failure."""
    manager = LLMManager()

    mock_fallback_response = MagicMock()
    mock_fallback_response.choices = [
        MagicMock(message=MagicMock(content="Fallback response"))
    ]

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
async def test_recovery_to_primary(mock_env_vars):
    """Test recovery back to primary LLM when it becomes available."""
    manager = LLMManager()

    # Start in fallback mode
    manager.using_fallback = True
    manager.last_recovery_check = datetime.now() - timedelta(minutes=10)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="OK"))]

    with patch(
        "shared.llm_manager.completion", return_value=mock_response
    ) as mock_completion:
        messages = [{"role": "user", "content": "Test message"}]
        response = await manager.call_llm(messages)

        assert response == mock_response
        assert not manager.using_fallback  # Should have recovered
        assert manager.primary_failure_time is None

        # Should have made health check and actual call
        assert mock_completion.call_count == 2


@pytest.mark.asyncio
async def test_both_llms_fail(mock_env_vars):
    """Test error handling when both primary and fallback LLMs fail."""
    manager = LLMManager()

    # Simulate primary failure timeout
    manager.primary_failure_time = datetime.now() - timedelta(minutes=3)

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
    env_vars = {
        "LLM_MODEL": "openai/gpt-4",
        "LLM_KEY": "test-key",
        "LLM_FALLBACK_MODEL": "",  # Explicitly clear fallback
        "LLM_FALLBACK_KEY": "",
        "LLM_DEBUG_OUTPUT": "false",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        LLMManager._reset_singleton()
        manager = LLMManager()

        assert manager.fallback_model is None or manager.fallback_model == ""

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
    assert status["primary_failure_time"] is None
    assert status["fallback_configured"]

    # Test status during failure
    manager.primary_failure_time = datetime.now()
    manager.using_fallback = True
    status = manager.get_status()
    assert status["current_model"] == "anthropic/claude-3"
    assert status["using_fallback"]
    assert status["primary_failure_time"] is not None
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
            prompt="Test prompt", context_instructions="Test context"
        )

        result = await env.run(activities.agent_tool_planner, input_data)
        assert result == {"result": "success"}


# Debug Output Tests
@pytest.mark.asyncio
async def test_debug_output_disabled_by_default():
    """Test that debug output is disabled by default."""
    env_vars = {
        "LLM_MODEL": "openai/gpt-4",
        "LLM_KEY": "test-key",
        "LLM_DEBUG_OUTPUT": "false",  # Explicitly set to false
    }
    with patch.dict(os.environ, env_vars, clear=True):
        LLMManager._reset_singleton()
        manager = LLMManager()
        assert not manager.debug_output_enabled


@pytest.mark.asyncio
async def test_debug_output_enabled(temp_debug_dir):
    """Test that debug output can be enabled and configured."""
    env_vars = {
        "LLM_MODEL": "openai/gpt-4",
        "LLM_KEY": "test-key",
        "LLM_DEBUG_OUTPUT": "true",
        "LLM_DEBUG_OUTPUT_DIR": temp_debug_dir,
    }

    with patch.dict(os.environ, env_vars):
        LLMManager._reset_singleton()
        manager = LLMManager()
        assert manager.debug_output_enabled
        assert manager.debug_output_dir == temp_debug_dir


@pytest.mark.asyncio
async def test_save_debug_output(temp_debug_dir):
    """Test that debug output is saved correctly."""
    env_vars = {
        "LLM_MODEL": "openai/gpt-4",
        "LLM_KEY": "test-key",
        "LLM_DEBUG_OUTPUT": "true",
        "LLM_DEBUG_OUTPUT_DIR": temp_debug_dir,
    }

    with patch.dict(os.environ, env_vars):
        LLMManager._reset_singleton()
        manager = LLMManager()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, world!"},
        ]
        kwargs = {"temperature": 0.7, "max_tokens": 100}

        mock_response = MagicMock()
        with patch("shared.llm_manager.completion", return_value=mock_response):
            await manager.call_llm(messages, **kwargs)

        # Check that debug file was created
        debug_files = [
            f for f in os.listdir(temp_debug_dir) if f.startswith("llm_call_")
        ]
        assert len(debug_files) == 1

        # Verify file contents
        debug_file_path = os.path.join(temp_debug_dir, debug_files[0])
        with open(debug_file_path, "r") as f:
            content = f.read()

        assert "=== LLM Debug Output ===" in content
        assert "Model: openai/gpt-4" in content
        assert "Using Fallback: False" in content
        assert "temperature" in content
        assert "max_tokens" in content
        assert "As (SYSTEM) :" in content
        assert "You are a helpful assistant." in content
        assert "As (USER) :" in content
        assert "Hello, world!" in content
        assert "=== FOR MANUAL TESTING ===" in content


@pytest.mark.asyncio
async def test_debug_output_fallback_model(temp_debug_dir):
    """Test debug output when using fallback model."""
    env_vars = {
        "LLM_MODEL": "openai/gpt-4",
        "LLM_KEY": "test-key",
        "LLM_FALLBACK_MODEL": "anthropic/claude-3",
        "LLM_FALLBACK_KEY": "fallback-key",
        "LLM_DEBUG_OUTPUT": "true",
        "LLM_DEBUG_OUTPUT_DIR": temp_debug_dir,
        "LLM_FALLBACK_DURATION_MINUTES": "0",  # Immediate fallback
    }

    with patch.dict(os.environ, env_vars):
        LLMManager._reset_singleton()
        manager = LLMManager()

        # Force fallback mode
        manager.using_fallback = True

        messages = [{"role": "user", "content": "Test fallback"}]

        mock_response = MagicMock()
        with patch("shared.llm_manager.completion", return_value=mock_response):
            await manager.call_llm(messages)

        # Check debug file
        debug_files = [
            f for f in os.listdir(temp_debug_dir) if f.startswith("llm_call_")
        ]
        assert len(debug_files) == 1

        debug_file_path = os.path.join(temp_debug_dir, debug_files[0])
        with open(debug_file_path, "r") as f:
            content = f.read()

        assert "Model: anthropic/claude-3" in content
        assert "Using Fallback: True" in content


@pytest.mark.asyncio
async def test_cleanup_old_debug_files(temp_debug_dir):
    """Test that old debug files are cleaned up, keeping only 20 most recent."""
    # Create 25 fake debug files with different timestamps
    for i in range(25):
        filename = f"llm_call_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:03d}.txt"
        filepath = os.path.join(temp_debug_dir, filename)
        with open(filepath, "w") as f:
            f.write(f"Debug file {i}")

        # Modify file times to simulate different creation times
        timestamp = (datetime.now() - timedelta(hours=i)).timestamp()
        os.utime(filepath, (timestamp, timestamp))

    env_vars = {
        "LLM_MODEL": "openai/gpt-4",
        "LLM_KEY": "test-key",
        "LLM_DEBUG_OUTPUT": "true",
        "LLM_DEBUG_OUTPUT_DIR": temp_debug_dir,
    }

    with patch.dict(os.environ, env_vars):
        LLMManager._reset_singleton()
        manager = LLMManager()

        # This should trigger cleanup
        manager._cleanup_old_debug_files()

        # Check that only 20 files remain
        debug_files = [
            f for f in os.listdir(temp_debug_dir) if f.startswith("llm_call_")
        ]
        assert len(debug_files) == 20

        # Verify the most recent files are kept (files 0-19, which have newer timestamps)
        remaining_files = sorted(debug_files)
        for i, filename in enumerate(remaining_files):
            # Files should be sorted by name (which includes timestamp)
            assert f"_{i:03d}.txt" in filename


@pytest.mark.asyncio
async def test_cleanup_called_during_save_debug_output(temp_debug_dir):
    """Test that cleanup is automatically called when saving debug output."""
    # Create 22 existing files
    for i in range(22):
        filename = f"llm_call_20240101_120000_{i:03d}.txt"
        filepath = os.path.join(temp_debug_dir, filename)
        with open(filepath, "w") as f:
            f.write(f"Old debug file {i}")

    env_vars = {
        "LLM_MODEL": "openai/gpt-4",
        "LLM_KEY": "test-key",
        "LLM_DEBUG_OUTPUT": "true",
        "LLM_DEBUG_OUTPUT_DIR": temp_debug_dir,
    }

    with patch.dict(os.environ, env_vars):
        LLMManager._reset_singleton()
        manager = LLMManager()

        messages = [{"role": "user", "content": "Test cleanup"}]

        mock_response = MagicMock()
        with patch("shared.llm_manager.completion", return_value=mock_response):
            await manager.call_llm(messages)

        # Should have cleaned up old files and created 1 new one
        debug_files = [
            f for f in os.listdir(temp_debug_dir) if f.startswith("llm_call_")
        ]
        assert (
            len(debug_files) <= 21
        )  # 20 old + 1 new, but cleanup may have removed some old ones


@pytest.mark.asyncio
async def test_debug_output_error_handling(temp_debug_dir):
    """Test error handling when debug output fails."""
    # Make directory read-only to simulate write error
    os.chmod(temp_debug_dir, 0o444)

    env_vars = {
        "LLM_MODEL": "openai/gpt-4",
        "LLM_KEY": "test-key",
        "LLM_DEBUG_OUTPUT": "true",
        "LLM_DEBUG_OUTPUT_DIR": temp_debug_dir,
    }

    try:
        with patch.dict(os.environ, env_vars):
            LLMManager._reset_singleton()
            manager = LLMManager()

            messages = [{"role": "user", "content": "Test error handling"}]

            mock_response = MagicMock()
            with patch("shared.llm_manager.completion", return_value=mock_response):
                # Should not raise exception even if debug output fails
                response = await manager.call_llm(messages)
                assert response == mock_response

    finally:
        # Restore permissions for cleanup
        os.chmod(temp_debug_dir, 0o755)


@pytest.mark.asyncio
async def test_cleanup_error_handling(temp_debug_dir):
    """Test error handling during file cleanup."""
    # Create a file and make it undeletable
    protected_file = os.path.join(temp_debug_dir, "llm_call_protected.txt")
    with open(protected_file, "w") as f:
        f.write("Protected file")
    os.chmod(protected_file, 0o444)

    # Create other normal files
    for i in range(25):
        filename = f"llm_call_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:03d}.txt"
        filepath = os.path.join(temp_debug_dir, filename)
        with open(filepath, "w") as f:
            f.write(f"Debug file {i}")

    env_vars = {
        "LLM_MODEL": "openai/gpt-4",
        "LLM_KEY": "test-key",
        "LLM_DEBUG_OUTPUT": "true",
        "LLM_DEBUG_OUTPUT_DIR": temp_debug_dir,
    }

    with patch.dict(os.environ, env_vars):
        LLMManager._reset_singleton()
        manager = LLMManager()

        # Cleanup should handle protected file gracefully
        try:
            manager._cleanup_old_debug_files()
            # Should not raise an exception
        except Exception as e:
            pytest.fail(f"Cleanup should handle file deletion errors gracefully: {e}")

        # Should have cleaned up some files, even if some failed
        debug_files = [
            f for f in os.listdir(temp_debug_dir) if f.startswith("llm_call_")
        ]
        assert len(debug_files) >= 1  # At least the protected file should remain


@pytest.mark.asyncio
async def test_debug_output_with_complex_messages(temp_debug_dir):
    """Test debug output with complex message structures."""
    env_vars = {
        "LLM_MODEL": "openai/gpt-4",
        "LLM_KEY": "test-key",
        "LLM_DEBUG_OUTPUT": "true",
        "LLM_DEBUG_OUTPUT_DIR": temp_debug_dir,
    }

    with patch.dict(os.environ, env_vars):
        LLMManager._reset_singleton()
        manager = LLMManager()

        # Messages with special characters and multiline content
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant.\nYou should be helpful and harmless.",
            },
            {"role": "user", "content": "What's 2+2?\nPlease explain your reasoning."},
            {"role": "assistant", "content": "2+2=4\n\nThis is basic arithmetic."},
            {
                "role": "user",
                "content": "Thanks! Can you help with \"quotes\" and 'apostrophes'?",
            },
        ]

        mock_response = MagicMock()
        with patch("shared.llm_manager.completion", return_value=mock_response):
            await manager.call_llm(messages)

        debug_files = [
            f for f in os.listdir(temp_debug_dir) if f.startswith("llm_call_")
        ]
        assert len(debug_files) == 1

        debug_file_path = os.path.join(temp_debug_dir, debug_files[0])
        with open(debug_file_path, "r") as f:
            content = f.read()

        # Verify all messages are present with proper formatting
        assert "As (SYSTEM) :" in content
        assert (
            "You are a helpful assistant.\nYou should be helpful and harmless."
            in content
        )
        assert "As (USER) :" in content
        assert "What's 2+2?\nPlease explain your reasoning." in content
        assert "As (ASSISTANT) :" in content
        assert "2+2=4\n\nThis is basic arithmetic." in content
        assert "quotes" in content and "apostrophes" in content
