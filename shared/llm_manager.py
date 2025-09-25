"""
LLM Manager with automatic fallback support.

This module provides a robust LLM client that automatically switches to a fallback
LLM when the primary LLM fails. The system includes:

- **Immediate Fallback**: Switches to fallback LLM immediately on any primary failure
- **Automatic Recovery**: Periodically checks if primary LLM has recovered
- **Health Monitoring**: Tracks failure times and recovery status
- **Debug Output**: Optional debug file output for troubleshooting
- **Singleton Pattern**: Ensures consistent state across the application

Environment Variables:
    LLM_MODEL: Primary LLM model (e.g., "openai/gpt-4")
    LLM_KEY: API key for primary LLM
    LLM_BASE_URL: Optional custom base URL for primary LLM
    LLM_FALLBACK_MODEL: Fallback LLM model
    LLM_FALLBACK_KEY: API key for fallback LLM
    LLM_FALLBACK_BASE_URL: Optional custom base URL for fallback LLM
    LLM_FALLBACK_DURATION_MINUTES: Max time to use fallback before forcing primary retry (default: 5)
    LLM_RECOVERY_CHECK_INTERVAL_MINUTES: How often to check if primary recovered (default: 5)
    LLM_DEBUG_OUTPUT: Enable debug file output ("true"/"false", default: "false")
    LLM_DEBUG_OUTPUT_DIR: Directory for debug files (default: "./debug_llm_calls")

Usage:
    manager = LLMManager()
    response = await manager.call_llm([{"role": "user", "content": "Hello"}])
    status = manager.get_status()
"""

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from litellm import completion
from temporalio import activity

load_dotenv(override=True)


class LLMManager:
    """
    Manages LLM calls with automatic fallback to secondary models.

    This class implements a singleton pattern to ensure consistent state across
    the application. It provides robust LLM calling with immediate fallback
    on failures and automatic recovery detection.

    Key Features:
    - Singleton pattern ensures one instance per process
    - Immediate fallback switching on any primary LLM failure
    - Periodic health checks to detect primary LLM recovery
    - Configurable fallback duration and recovery intervals
    - Optional debug file output for troubleshooting
    - Comprehensive logging for monitoring and debugging

    State Management:
    - using_fallback: Boolean indicating if currently using fallback LLM
    - primary_failure_time: Timestamp of when primary LLM first failed
    - last_recovery_check: Timestamp of last recovery attempt
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            print(f"[LLMManager.__new__] Creating new singleton instance")
            cls._instance = super(LLMManager, cls).__new__(cls)
        else:
            print(f"[LLMManager.__new__] Returning existing singleton instance")
        return cls._instance

    def __init__(self):
        """Initialize LLM Manager with primary and fallback configurations."""
        # Only initialize once due to singleton pattern
        if LLMManager._initialized:
            print(f"[LLMManager.__init__] Singleton already initialized, skipping")
            return
        print(f"[LLMManager.__init__] Initializing singleton instance")
        LLMManager._initialized = True
        # Primary LLM configuration
        print(
            f"[LLMManager.__init__] Loading primary LLM configuration from environment"
        )
        self.primary_model = os.environ.get("LLM_MODEL", "openai/gpt-4")
        self.primary_key = os.environ.get("LLM_KEY")
        self.primary_base_url = os.environ.get("LLM_BASE_URL")
        print(
            f"[LLMManager.__init__] Primary LLM: model={self.primary_model}, key={'set' if self.primary_key else 'not set'}, base_url={self.primary_base_url or 'default'}"
        )

        # Fallback LLM configuration
        print(
            f"[LLMManager.__init__] Loading fallback LLM configuration from environment"
        )
        self.fallback_model = os.environ.get("LLM_FALLBACK_MODEL")
        self.fallback_key = os.environ.get("LLM_FALLBACK_KEY")
        self.fallback_base_url = os.environ.get("LLM_FALLBACK_BASE_URL")
        print(
            f"[LLMManager.__init__] Fallback LLM: model={self.fallback_model or 'not set'}, key={'set' if self.fallback_key else 'not set'}, base_url={self.fallback_base_url or 'default'}"
        )

        # Failure tracking
        print(f"[LLMManager.__init__] Initializing failure tracking state")
        self.primary_failure_time: Optional[datetime] = None
        self.using_fallback = False
        self.fallback_duration_minutes = int(
            os.environ.get("LLM_FALLBACK_DURATION_MINUTES", "5")
        )
        print(
            f"[LLMManager.__init__] Fallback duration: {self.fallback_duration_minutes} minutes"
        )

        # Recovery check settings
        self.last_recovery_check: Optional[datetime] = None
        self.recovery_check_interval_minutes = int(
            os.environ.get("LLM_RECOVERY_CHECK_INTERVAL_MINUTES", "5")
        )
        print(
            f"[LLMManager.__init__] Recovery check interval: {self.recovery_check_interval_minutes} minutes"
        )

        # Debug file settings
        self.debug_output_enabled = (
            os.environ.get("LLM_DEBUG_OUTPUT", "false").lower() == "true"
        )
        self.debug_output_dir = os.environ.get(
            "LLM_DEBUG_OUTPUT_DIR", "./debug_llm_calls"
        )
        print(
            f"[LLMManager.__init__] Debug output: enabled={self.debug_output_enabled}, dir={self.debug_output_dir}"
        )

        print(
            f"[LLMManager.__init__] Initialization complete, logging final configuration"
        )
        self._log_configuration()

    def _log_configuration(self):
        """Log the LLM configuration for debugging."""
        print(f"[LLMManager._log_configuration] LLM Manager initialized:")
        print(f"  Primary model: {self.primary_model}")
        print(f"  Primary API key: {'***set***' if self.primary_key else 'not set'}")
        print(f"  Primary base URL: {self.primary_base_url or 'default'}")

        if self.fallback_model:
            print(f"  Fallback model: {self.fallback_model}")
            print(
                f"  Fallback API key: {'***set***' if self.fallback_key else 'not set'}"
            )
            print(f"  Fallback base URL: {self.fallback_base_url or 'default'}")
            print(f"  Fallback duration: {self.fallback_duration_minutes} minutes")
            print(
                f"  Recovery check interval: {self.recovery_check_interval_minutes} minutes"
            )
        else:
            print(f"  No fallback model configured")

        if self.debug_output_enabled:
            print(f"  Debug output enabled: {self.debug_output_dir}")
        else:
            print(f"  Debug output disabled")

        print(f"  Initial state: using_fallback=False, primary_failure_time=None")

    async def call_llm(
        self, messages: List[Dict[str, str]], **kwargs
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
        activity.logger.debug(
            f"[LLMManager.call_llm] Starting LLM call with {len(messages)} messages"
        )
        activity.logger.debug(
            f"[LLMManager.call_llm] Current state: using_fallback={self.using_fallback}, primary_failure_time={self.primary_failure_time}"
        )
        activity.logger.debug(
            f"[LLMManager.call_llm] Primary model: {self.primary_model}, Fallback model: {self.fallback_model}"
        )

        # Save debug output if enabled
        if self.debug_output_enabled:
            activity.logger.debug(
                f"[LLMManager.call_llm] Debug output enabled, saving to {self.debug_output_dir}"
            )
            await self._save_debug_output(messages, kwargs)
        else:
            activity.logger.debug(f"[LLMManager.call_llm] Debug output disabled")
        # Check if we should try to recover from fallback mode
        if self.using_fallback:
            activity.logger.debug(
                f"[LLMManager.call_llm] Currently using fallback, checking for recovery"
            )
            if await self._should_try_recovery():
                activity.logger.debug(
                    f"[LLMManager.call_llm] Recovery check interval passed, attempting primary health check"
                )
                if await self._try_primary():
                    activity.logger.debug(
                        f"[LLMManager.call_llm] Primary health check successful, switching back to primary"
                    )
                    self.using_fallback = False
                    self.primary_failure_time = None
                    activity.logger.info("Successfully recovered to primary LLM")
                else:
                    activity.logger.debug(
                        f"[LLMManager.call_llm] Primary health check failed, staying in fallback mode"
                    )
                    # Still in fallback mode, check if fallback period has expired
                    if self._fallback_period_expired():
                        activity.logger.debug(
                            f"[LLMManager.call_llm] Fallback period expired, forcing primary retry"
                        )
                        activity.logger.info(
                            "Fallback period expired, forcing attempt to use primary LLM"
                        )
                        self.using_fallback = False
                        self.primary_failure_time = None
                    else:
                        activity.logger.debug(
                            f"[LLMManager.call_llm] Fallback period not expired, continuing with fallback"
                        )
            else:
                activity.logger.debug(
                    f"[LLMManager.call_llm] Recovery check interval not reached yet"
                )
        else:
            activity.logger.debug(
                f"[LLMManager.call_llm] Not currently using fallback, proceeding with primary"
            )

        # Determine which LLM to use
        if self.using_fallback:
            activity.logger.debug(
                f"[LLMManager.call_llm] Using fallback LLM: {self.fallback_model}"
            )
            return await self._call_fallback_llm(messages, **kwargs)

        # Try primary LLM
        activity.logger.debug(
            f"[LLMManager.call_llm] Attempting primary LLM call: {self.primary_model}"
        )
        try:
            response = await self._call_primary_llm(messages, **kwargs)
            # Reset failure tracking on success
            if self.primary_failure_time:
                activity.logger.debug(
                    f"[LLMManager.call_llm] Primary LLM call successful, resetting failure tracking"
                )
                activity.logger.info(f"Primary LLM recovered after fallback period")
            else:
                activity.logger.debug(
                    f"[LLMManager.call_llm] Primary LLM call successful (no previous failures)"
                )
            self.primary_failure_time = None
            return response

        except Exception as primary_error:
            activity.logger.debug(
                f"[LLMManager.call_llm] Primary LLM call failed: {type(primary_error).__name__}: {str(primary_error)}"
            )
            activity.logger.warning(f"Primary LLM failed: {str(primary_error)}")

            # Switch to fallback immediately on any failure (if available)
            if self._should_use_fallback():
                activity.logger.debug(
                    f"[LLMManager.call_llm] Fallback available, switching to fallback mode"
                )
                self.primary_failure_time = datetime.now()
                self.using_fallback = True
                activity.logger.info(
                    f"Switching to fallback LLM due to primary failure"
                )
                return await self._call_fallback_llm(messages, **kwargs)

            # Re-raise the error if no fallback configured
            activity.logger.debug(
                f"[LLMManager.call_llm] No fallback configured, re-raising primary error"
            )
            raise primary_error

    async def _call_primary_llm(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """Call the primary LLM."""
        activity.logger.debug(
            f"[LLMManager._call_primary_llm] Starting primary LLM call"
        )
        completion_kwargs = {
            "model": self.primary_model,
            "messages": messages,
            "api_key": self.primary_key,
            **kwargs,
        }

        if self.primary_base_url:
            completion_kwargs["base_url"] = self.primary_base_url
            activity.logger.debug(
                f"[LLMManager._call_primary_llm] Using custom base URL: {self.primary_base_url}"
            )
        else:
            activity.logger.debug(
                f"[LLMManager._call_primary_llm] Using default base URL for model: {self.primary_model}"
            )

        activity.logger.debug(
            f"[LLMManager._call_primary_llm] Calling litellm completion with timeout=10s"
        )
        response = completion(**completion_kwargs, timeout=10)  # 10 seconds
        activity.logger.debug(
            f"[LLMManager._call_primary_llm] Primary LLM call successful"
        )

        return response

    async def _call_fallback_llm(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """Call the fallback LLM."""
        activity.logger.debug(
            f"[LLMManager._call_fallback_llm] Starting fallback LLM call"
        )
        if not self.fallback_model:
            activity.logger.debug(
                f"[LLMManager._call_fallback_llm] No fallback model configured"
            )
            raise Exception("No fallback LLM configured")

        completion_kwargs = {
            "model": self.fallback_model,
            "messages": messages,
            "api_key": self.fallback_key,
            **kwargs,
        }

        if self.fallback_base_url:
            completion_kwargs["base_url"] = self.fallback_base_url
            activity.logger.debug(
                f"[LLMManager._call_fallback_llm] Using custom base URL: {self.fallback_base_url}"
            )
        else:
            activity.logger.debug(
                f"[LLMManager._call_fallback_llm] Using default base URL for model: {self.fallback_model}"
            )

        try:
            activity.logger.debug(
                f"[LLMManager._call_fallback_llm] Calling litellm completion with timeout=10s"
            )
            response = completion(**completion_kwargs, timeout=10)  # 10 seconds
            activity.logger.debug(
                f"[LLMManager._call_fallback_llm] Fallback LLM call successful"
            )
            activity.logger.info(
                f"Successfully used fallback LLM: {self.fallback_model}"
            )
            return response
        except Exception as fallback_error:
            activity.logger.debug(
                f"[LLMManager._call_fallback_llm] Fallback LLM call failed: {type(fallback_error).__name__}: {str(fallback_error)}"
            )
            activity.logger.error(f"Fallback LLM also failed: {str(fallback_error)}")
            raise Exception(
                f"Both primary and fallback LLMs failed. "
                f"Primary: {self.primary_model}, Fallback: {self.fallback_model}"
            )

    def _should_use_fallback(self) -> bool:
        """
        Check if we should switch to the fallback LLM.

        Returns True if a fallback LLM is configured and available.
        The system now switches immediately on any primary failure if fallback is available.

        Returns:
            bool: True if fallback should be used, False otherwise
        """
        result = bool(self.fallback_model)
        activity.logger.debug(
            f"[LLMManager._should_use_fallback] Fallback available: {result} (fallback_model={self.fallback_model})"
        )
        return result

    def _fallback_period_expired(self) -> bool:
        """
        Check if the fallback period has expired and we should force retry primary.

        After a certain duration using the fallback LLM, the system will force
        a retry of the primary LLM even if health checks are failing. This prevents
        getting stuck in fallback mode indefinitely.

        Returns:
            bool: True if fallback period has expired, False otherwise
        """
        if not self.primary_failure_time:
            activity.logger.debug(
                f"[LLMManager._fallback_period_expired] No primary failure time recorded, period not expired"
            )
            return False

        failure_duration = datetime.now() - self.primary_failure_time
        duration_minutes = failure_duration.total_seconds() / 60
        expired = failure_duration > timedelta(minutes=self.fallback_duration_minutes)
        activity.logger.debug(
            f"[LLMManager._fallback_period_expired] Duration since failure: {duration_minutes:.1f} minutes, threshold: {self.fallback_duration_minutes} minutes, expired: {expired}"
        )
        return expired

    async def _should_try_recovery(self) -> bool:
        """
        Check if we should try to recover to the primary LLM.

        Recovery attempts are throttled by the recovery check interval to avoid
        excessive health check requests. The first recovery attempt happens immediately
        when switching to fallback mode, then subsequent attempts are spaced out.

        Returns:
            bool: True if we should attempt recovery, False otherwise
        """
        activity.logger.debug(
            f"[LLMManager._should_try_recovery] Checking if recovery should be attempted"
        )

        # If this is the first time a recovery check is attempted, attempt
        # to recover
        if not self.last_recovery_check:
            activity.logger.debug(
                f"[LLMManager._should_try_recovery] First recovery attempt, proceeding"
            )
            return True

        # another recovery should only be attempted after the predefined interval.
        # if the interval has been exceeded, a recovery should be retried.
        time_since_check = datetime.now() - self.last_recovery_check
        minutes_since_check = time_since_check.total_seconds() / 60
        should_retry = time_since_check > timedelta(
            minutes=self.recovery_check_interval_minutes
        )
        activity.logger.debug(
            f"[LLMManager._should_try_recovery] Time since last check: {minutes_since_check:.1f} minutes, interval: {self.recovery_check_interval_minutes} minutes, should retry: {should_retry}"
        )
        return should_retry

    async def _try_primary(self) -> bool:
        """
        Try to use the primary LLM with a simple test message.

        Returns:
            True if the primary LLM is working, False otherwise
        """
        activity.logger.debug(
            f"[LLMManager._try_primary] Starting primary LLM health check"
        )
        self.last_recovery_check = datetime.now()
        activity.logger.debug(
            f"[LLMManager._try_primary] Updated last_recovery_check timestamp"
        )

        try:
            test_messages = [
                {"role": "user", "content": "Reply with 'OK' if you're working"}
            ]
            activity.logger.debug(
                f"[LLMManager._try_primary] Using test message for health check"
            )

            completion_kwargs = {
                "model": self.primary_model,
                "messages": test_messages,
                "api_key": self.primary_key,
                "max_tokens": 10,
                "timeout": 10,  # Short timeout for health check
            }

            if self.primary_base_url:
                completion_kwargs["base_url"] = self.primary_base_url
                activity.logger.debug(
                    f"[LLMManager._try_primary] Using custom base URL for health check: {self.primary_base_url}"
                )
            else:
                activity.logger.debug(
                    f"[LLMManager._try_primary] Using default base URL for health check"
                )

            activity.logger.debug(
                f"[LLMManager._try_primary] Calling litellm completion for health check"
            )
            response = completion(**completion_kwargs)
            activity.logger.debug(
                f"[LLMManager._try_primary] Health check call successful"
            )
            activity.logger.info("Primary LLM health check succeeded")
            return True

        except Exception as e:
            activity.logger.debug(
                f"[LLMManager._try_primary] Health check failed: {type(e).__name__}: {str(e)}"
            )
            activity.logger.debug(f"Primary LLM health check failed: {str(e)}")
            return False

    def get_current_model(self) -> str:
        """
        Get the currently active model name.

        Returns:
            str: The name of the currently active LLM model (primary or fallback)
        """
        current = self.fallback_model if self.using_fallback else self.primary_model
        activity.logger.debug(
            f"[LLMManager.get_current_model] Current active model: {current} (using_fallback={self.using_fallback})"
        )
        return current

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the LLM manager.

        Returns comprehensive status information including current model,
        fallback state, failure tracking, and recovery check information.

        Returns:
            Dict[str, Any]: Status dictionary containing:
                - current_model: Currently active model name
                - using_fallback: Whether currently using fallback LLM
                - primary_failure_time: ISO timestamp of primary failure (if any)
                - failure_duration_seconds: Seconds since primary failure (if any)
                - fallback_configured: Whether fallback LLM is configured
                - last_recovery_check: ISO timestamp of last recovery check (if any)
        """
        activity.logger.debug(f"[LLMManager.get_status] Generating status report")
        status = {
            "current_model": self.get_current_model(),
            "using_fallback": self.using_fallback,
            "primary_failure_time": self.primary_failure_time.isoformat()
            if self.primary_failure_time
            else None,
            "failure_duration_seconds": (
                (datetime.now() - self.primary_failure_time).total_seconds()
                if self.primary_failure_time
                else None
            ),
            "fallback_configured": bool(self.fallback_model),
            "last_recovery_check": self.last_recovery_check.isoformat()
            if self.last_recovery_check
            else None,
        }
        activity.logger.debug(f"[LLMManager.get_status] Status: {status}")
        return status

    @classmethod
    def _reset_singleton(cls):
        """Reset the singleton instance. Only for testing purposes."""
        print(f"[LLMManager._reset_singleton] Resetting singleton instance")
        cls._instance = None
        cls._initialized = False

    async def _save_debug_output(
        self, messages: List[Dict[str, str]], kwargs: Dict[str, Any]
    ) -> None:
        """Save LLM messages in a format that can be cut/pasted into an LLM interface."""
        activity.logger.debug(
            f"[LLMManager._save_debug_output] Starting debug output save"
        )
        try:
            # Create debug directory if it doesn't exist
            activity.logger.debug(
                f"[LLMManager._save_debug_output] Ensuring debug directory exists: {self.debug_output_dir}"
            )
            os.makedirs(self.debug_output_dir, exist_ok=True)

            # Clean up old files, keeping only the 20 most recent
            activity.logger.debug(
                f"[LLMManager._save_debug_output] Cleaning up old debug files"
            )
            self._cleanup_old_debug_files()

            # Generate timestamp-based filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[
                :-3
            ]  # Include milliseconds
            filename = f"llm_call_{timestamp}.txt"
            filepath = os.path.join(self.debug_output_dir, filename)
            activity.logger.debug(
                f"[LLMManager._save_debug_output] Writing debug output to: {filepath}"
            )

            # Write to file
            with open(filepath, "w") as f:
                # Write header information
                f.write(f"=== LLM Debug Output ===\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Model: {self.get_current_model()}\n")
                f.write(f"Using Fallback: {self.using_fallback}\n")
                f.write(f"Extra Args: {kwargs}\n")
                f.write("=" * 50 + "\n\n")

                # Write each message in a readable format
                activity.logger.debug(
                    f"[LLMManager._save_debug_output] Writing {len(messages)} messages to debug file"
                )
                for i, message in enumerate(messages, 1):
                    role = message.get("role", "unknown")
                    content = message.get("content", "")

                    f.write(f"As ({role.upper()}) :\n")
                    f.write(f"{content}\n")
                    f.write("\n" + "-" * 30 + "\n\n")

                # Add a section for easy copying
                f.write("=== FOR MANUAL TESTING ===\n")
                f.write("Copy the messages above and paste into your LLM interface.\n")

            activity.logger.debug(
                f"[LLMManager._save_debug_output] Successfully saved debug output to {filepath}"
            )
            activity.logger.debug(f"Saved LLM debug output to {filepath}")

        except Exception as e:
            activity.logger.debug(
                f"[LLMManager._save_debug_output] Failed to save debug output: {type(e).__name__}: {str(e)}"
            )
            activity.logger.warning(f"Failed to save LLM debug output: {str(e)}")

    def _cleanup_old_debug_files(self) -> None:
        """Keep only the 20 most recent debug files, delete older ones."""
        activity.logger.debug(
            f"[LLMManager._cleanup_old_debug_files] Starting cleanup of old debug files"
        )
        try:
            # Get all debug files in the directory
            debug_files = []
            activity.logger.debug(
                f"[LLMManager._cleanup_old_debug_files] Scanning directory: {self.debug_output_dir}"
            )
            for filename in os.listdir(self.debug_output_dir):
                if filename.startswith("llm_call_") and filename.endswith(".txt"):
                    filepath = os.path.join(self.debug_output_dir, filename)
                    if os.path.isfile(filepath):
                        # Get file modification time
                        mtime = os.path.getmtime(filepath)
                        debug_files.append((filepath, mtime))

            activity.logger.debug(
                f"[LLMManager._cleanup_old_debug_files] Found {len(debug_files)} debug files"
            )

            # Sort by modification time (newest first)
            debug_files.sort(key=lambda x: x[1], reverse=True)

            # Keep only the 20 most recent files, delete the rest
            if len(debug_files) > 20:
                files_to_delete = debug_files[20:]
                activity.logger.debug(
                    f"[LLMManager._cleanup_old_debug_files] Need to delete {len(files_to_delete)} old files"
                )
                for filepath, _ in files_to_delete:
                    try:
                        os.remove(filepath)
                        activity.logger.debug(
                            f"[LLMManager._cleanup_old_debug_files] Deleted old debug file: {filepath}"
                        )
                        activity.logger.debug(f"Deleted old debug file: {filepath}")
                    except OSError as e:
                        activity.logger.debug(
                            f"[LLMManager._cleanup_old_debug_files] Failed to delete {filepath}: {str(e)}"
                        )
                        activity.logger.warning(
                            f"Failed to delete old debug file {filepath}: {str(e)}"
                        )
            else:
                activity.logger.debug(
                    f"[LLMManager._cleanup_old_debug_files] No cleanup needed, {len(debug_files)} files <= 20 limit"
                )

        except Exception as e:
            activity.logger.debug(
                f"[LLMManager._cleanup_old_debug_files] Cleanup failed: {type(e).__name__}: {str(e)}"
            )
            activity.logger.warning(f"Failed to cleanup old debug files: {str(e)}")
