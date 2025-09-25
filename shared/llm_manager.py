"""
LLM Manager with automatic fallback support.

This module provides a robust LLM client that automatically switches to a fallback
LLM when the primary LLM fails for more than a specified duration (default 2 minutes).
"""

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from litellm import completion
from temporalio import activity

load_dotenv(override=True)


class LLMManager:
    """Manages LLM calls with automatic fallback to secondary models."""

    def __init__(self):
        """Initialize LLM Manager with primary and fallback configurations."""
        # Primary LLM configuration
        self.primary_model = os.environ.get("LLM_MODEL", "openai/gpt-4")
        self.primary_key = os.environ.get("LLM_KEY")
        self.primary_base_url = os.environ.get("LLM_BASE_URL")

        # Fallback LLM configuration
        self.fallback_model = os.environ.get("LLM_FALLBACK_MODEL")
        self.fallback_key = os.environ.get("LLM_FALLBACK_KEY")
        self.fallback_base_url = os.environ.get("LLM_FALLBACK_BASE_URL")

        # Failure tracking
        self.primary_failure_start: Optional[datetime] = None
        self.using_fallback = False
        self.fallback_timeout_minutes = int(os.environ.get("LLM_FALLBACK_TIMEOUT_MINUTES", "2"))

        # Recovery check settings
        self.last_recovery_check: Optional[datetime] = None
        self.recovery_check_interval_minutes = int(
            os.environ.get("LLM_RECOVERY_CHECK_INTERVAL_MINUTES", "5")
        )

        # Debug file settings
        self.debug_output_enabled = os.environ.get("LLM_DEBUG_OUTPUT", "false").lower() == "true"
        self.debug_output_dir = os.environ.get("LLM_DEBUG_OUTPUT_DIR", "./debug_llm_calls")

        self._log_configuration()

    def _log_configuration(self):
        """Log the LLM configuration for debugging."""
        print(f"LLM Manager initialized:")
        print(f"  Primary model: {self.primary_model}")
        print(f"  Primary base URL: {self.primary_base_url or 'default'}")

        if self.fallback_model:
            print(f"  Fallback model: {self.fallback_model}")
            print(f"  Fallback base URL: {self.fallback_base_url or 'default'}")
            print(f"  Fallback timeout: {self.fallback_timeout_minutes} minutes")
            print(f"  Recovery check interval: {self.recovery_check_interval_minutes} minutes")
        else:
            print("  No fallback model configured")

        if self.debug_output_enabled:
            print(f"  Debug output enabled: {self.debug_output_dir}")

    async def call_llm(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call LLM with automatic fallback support.

        Args:
            messages: The messages to send to the LLM
            **kwargs: Additional arguments to pass to litellm completion

        Returns:
            The LLM response

        Raises:
            Exception: If both primary and fallback LLMs fail
        """
        # Save debug output if enabled
        if self.debug_output_enabled:
            await self._save_debug_output(messages, kwargs)
        # Check if we should try to recover from fallback mode
        # If the fallback llm is in use and the conditions to try
        # the recovery back to the primary have been met, check
        # if the primary is available.
        if self.using_fallback and await self._should_try_recovery():
            if await self._try_primary():
                self.using_fallback = False
                self.primary_failure_start = None
                activity.logger.info("Successfully recovered to primary LLM")

        # Determine which LLM to use
        if self.using_fallback:
            return await self._call_fallback_llm(messages, **kwargs)

        # Try primary LLM
        try:
            response = await self._call_primary_llm(messages, **kwargs)
            # Reset failure tracking on success
            if self.primary_failure_start:
                activity.logger.info(
                    f"Primary LLM recovered after {datetime.now() - self.primary_failure_start}"
                )
            self.primary_failure_start = None
            return response

        except Exception as primary_error:
            activity.logger.warning(f"Primary LLM failed: {str(primary_error)}")

            # Track failure time
            if not self.primary_failure_start:
                self.primary_failure_start = datetime.now()
                activity.logger.info("Started tracking primary LLM failure time")

            # Check if we should switch to fallback
            if self._should_use_fallback():
                activity.logger.info(
                    f"Switching to fallback LLM after {datetime.now() - self.primary_failure_start} of failures"
                )
                self.using_fallback = True
                return await self._call_fallback_llm(messages, **kwargs)

            # Re-raise the error if no fallback or not yet time to switch
            raise primary_error

    async def _call_primary_llm(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Call the primary LLM."""
        completion_kwargs = {
            "model": self.primary_model,
            "messages": messages,
            "api_key": self.primary_key,
            **kwargs
        }

        if self.primary_base_url:
            completion_kwargs["base_url"] = self.primary_base_url

        response = completion(**completion_kwargs)
        return response

    async def _call_fallback_llm(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Call the fallback LLM."""
        if not self.fallback_model:
            raise Exception("No fallback LLM configured")

        completion_kwargs = {
            "model": self.fallback_model,
            "messages": messages,
            "api_key": self.fallback_key,
            **kwargs
        }

        if self.fallback_base_url:
            completion_kwargs["base_url"] = self.fallback_base_url

        try:
            response = completion(**completion_kwargs)
            activity.logger.info(f"Successfully used fallback LLM: {self.fallback_model}")
            return response
        except Exception as fallback_error:
            activity.logger.error(f"Fallback LLM also failed: {str(fallback_error)}")
            raise Exception(
                f"Both primary and fallback LLMs failed. "
                f"Primary: {self.primary_model}, Fallback: {self.fallback_model}"
            )

    def _should_use_fallback(self) -> bool:
        """
        Check if we should switch to the fallback LLM.
        """

        # Is a fallback model defined?
        if not self.fallback_model:
            return False

        # Has the primary started failing?
        if not self.primary_failure_start:
            return False

        # Has the duration a primary is allowed to fail been exceeded?
        failure_duration = datetime.now() - self.primary_failure_start
        return failure_duration > timedelta(minutes=self.fallback_timeout_minutes)

    async def _should_try_recovery(self) -> bool:
        """ Check if we should try to recover to the primary LLM. """

        # If this is the first time a recovery check is attempted, attempt
        # to recover
        if not self.last_recovery_check:
            return True

        # another recovery should only be attempted after the predefined interval.
        # if the interval has been exceeded, a recovery should be retried.
        time_since_check = datetime.now() - self.last_recovery_check
        return time_since_check > timedelta(minutes=self.recovery_check_interval_minutes)

    async def _try_primary(self) -> bool:
        """
        Try to use the primary LLM with a simple test message.

        Returns:
            True if the primary LLM is working, False otherwise
        """
        self.last_recovery_check = datetime.now()

        try:
            test_messages = [
                {"role": "user", "content": "Reply with 'OK' if you're working"}
            ]

            completion_kwargs = {
                "model": self.primary_model,
                "messages": test_messages,
                "api_key": self.primary_key,
                "max_tokens": 10,
                "timeout": 10  # Short timeout for health check
            }

            if self.primary_base_url:
                completion_kwargs["base_url"] = self.primary_base_url

            response = completion(**completion_kwargs)
            activity.logger.info("Primary LLM health check succeeded")
            return True

        except Exception as e:
            activity.logger.debug(f"Primary LLM health check failed: {str(e)}")
            return False

    def get_current_model(self) -> str:
        """Get the currently active model name."""
        return self.fallback_model if self.using_fallback else self.primary_model

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the LLM manager."""
        return {
            "current_model": self.get_current_model(),
            "using_fallback": self.using_fallback,
            "primary_failure_start": self.primary_failure_start.isoformat() if self.primary_failure_start else None,
            "failure_duration_seconds": (
                (datetime.now() - self.primary_failure_start).total_seconds()
                if self.primary_failure_start else None
            ),
            "fallback_configured": bool(self.fallback_model),
            "last_recovery_check": self.last_recovery_check.isoformat() if self.last_recovery_check else None
        }

    async def _save_debug_output(self, messages: List[Dict[str, str]], kwargs: Dict[str, Any]) -> None:
        """Save LLM messages in a format that can be cut/pasted into an LLM interface."""
        try:
            # Create debug directory if it doesn't exist
            os.makedirs(self.debug_output_dir, exist_ok=True)

            # Clean up old files, keeping only the 20 most recent
            self._cleanup_old_debug_files()

            # Generate timestamp-based filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            filename = f"llm_call_{timestamp}.txt"
            filepath = os.path.join(self.debug_output_dir, filename)

            # Write to file
            with open(filepath, 'w') as f:
                # Write header information
                f.write(f"=== LLM Debug Output ===\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Model: {self.get_current_model()}\n")
                f.write(f"Using Fallback: {self.using_fallback}\n")
                f.write(f"Extra Args: {kwargs}\n")
                f.write("=" * 50 + "\n\n")

                # Write each message in a readable format
                for i, message in enumerate(messages, 1):
                    role = message.get("role", "unknown")
                    content = message.get("content", "")

                    f.write(f"As ({role.upper()}) :\n")
                    f.write(f"{content}\n")
                    f.write("\n" + "-" * 30 + "\n\n")

                # Add a section for easy copying
                f.write("=== FOR MANUAL TESTING ===\n")
                f.write("Copy the messages above and paste into your LLM interface.\n")

            activity.logger.debug(f"Saved LLM debug output to {filepath}")

        except Exception as e:
            activity.logger.warning(f"Failed to save LLM debug output: {str(e)}")

    def _cleanup_old_debug_files(self) -> None:
        """Keep only the 20 most recent debug files, delete older ones."""
        try:
            # Get all debug files in the directory
            debug_files = []
            for filename in os.listdir(self.debug_output_dir):
                if filename.startswith("llm_call_") and filename.endswith(".txt"):
                    filepath = os.path.join(self.debug_output_dir, filename)
                    if os.path.isfile(filepath):
                        # Get file modification time
                        mtime = os.path.getmtime(filepath)
                        debug_files.append((filepath, mtime))

            # Sort by modification time (newest first)
            debug_files.sort(key=lambda x: x[1], reverse=True)

            # Keep only the 20 most recent files, delete the rest
            if len(debug_files) > 20:
                files_to_delete = debug_files[20:]
                for filepath, _ in files_to_delete:
                    try:
                        os.remove(filepath)
                        activity.logger.debug(f"Deleted old debug file: {filepath}")
                    except OSError as e:
                        activity.logger.warning(f"Failed to delete old debug file {filepath}: {str(e)}")

        except Exception as e:
            activity.logger.warning(f"Failed to cleanup old debug files: {str(e)}")
