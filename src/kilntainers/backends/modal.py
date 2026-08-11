"""Modal backend implementation."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import modal
else:
    modal = None

from kilntainers.backends.base import Backend, ExecRequest, ExecResult, Sandbox
from kilntainers.config import BackendConfig
from kilntainers.errors import BackendError, SandboxDiedError


def _get_modal_sdk(*, preserve_event_loop_policy: bool = True) -> Any:
    """Import Modal only when the Modal backend is actually used.

    Importing Modal changes the global asyncio event-loop policy on Windows.
    Imports from an already-running process preserve its policy so Modal cannot
    contaminate later event loops used by other backends. ModalBackend's startup
    hook deliberately allows the change when Modal is the selected backend.
    """
    global modal
    if modal is None:
        previous_policy = (
            asyncio.get_event_loop_policy() if preserve_event_loop_policy else None
        )
        try:
            modal = importlib.import_module("modal")
        finally:
            if previous_policy is not None:
                asyncio.set_event_loop_policy(previous_policy)
    return modal


class _OutputLimitExceeded(Exception):
    """Internal signal: combined output exceeded the configured limit."""

    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class ModalBackendConfig(BackendConfig):
    """Configuration for the Modal backend.

    Populated from CLI args by ModalBackend.config_from_args().
    Consumed by ModalBackend.
    """

    # Authentication (optional — falls back to Modal's default auth)
    token_id: str | None = None
    token_secret: str | None = None

    # Modal app
    app_name: str = "kilntainers"

    # Sandbox environment
    image: str | None = None  # None = debian_slim default
    shell: str = "/bin/bash"
    network_enabled: bool = False

    # Resources
    cpu: float = 1.0
    memory: int = 512  # MiB
    gpu: str | None = None
    region: str | None = None

    # Sandbox lifetime
    sandbox_timeout: int = 3600  # seconds (max 24h)


class ModalBackend(Backend):
    """Modal backend implementation.

    Manages Modal cloud sandbox lifecycle and command execution
    through the Modal Python SDK.
    """

    @classmethod
    def prepare_runtime(cls) -> None:
        """Load Modal before loop creation so its Windows policy takes effect."""
        _get_modal_sdk(preserve_event_loop_policy=False)

    @classmethod
    def add_cli_arguments(cls, group: argparse._ArgumentGroup) -> None:
        """Register Modal-specific CLI arguments."""
        group.add_argument(
            "--modal-token-id",
            default=None,
            help="Modal token ID (overrides environment/default auth)",
        )
        group.add_argument(
            "--modal-token-secret",
            default=None,
            help="Modal token secret (overrides environment/default auth)",
        )
        group.add_argument(
            "--modal-app-name",
            default="kilntainers",
            help="Modal app name (default: kilntainers)",
        )
        group.add_argument(
            "--modal-cpu",
            type=float,
            default=1.0,
            help="CPU cores (fractional, default: 1.0)",
        )
        group.add_argument(
            "--modal-memory",
            type=int,
            default=512,
            help="Memory in MiB (default: 512)",
        )
        group.add_argument(
            "--gpu",
            default=None,
            help='GPU type (e.g., "A10G", "H100")',
        )
        group.add_argument(
            "--region",
            default=None,
            help='Geographic region (e.g., "us-east")',
        )
        group.add_argument(
            "--sandbox-timeout",
            type=int,
            default=3600,
            help="Sandbox lifetime timeout in seconds (default: 3600, max 86400)",
        )

    @classmethod
    def config_from_args(cls, args: argparse.Namespace) -> BackendConfig:
        """Build ModalBackendConfig from parsed CLI arguments."""
        # Use core --shell with a backend-specific default
        shell = args.shell if args.shell is not None else "/bin/bash"
        return ModalBackendConfig(
            token_id=args.modal_token_id,
            token_secret=args.modal_token_secret,
            app_name=args.modal_app_name,
            image=args.image,
            shell=shell,
            network_enabled=args.network,
            cpu=args.modal_cpu,
            memory=args.modal_memory,
            gpu=args.gpu,
            region=args.region,
            sandbox_timeout=args.sandbox_timeout,
            default_timeout=args.timeout,
        )

    def __init__(self, config: ModalBackendConfig) -> None:
        super().__init__(config)
        # Override parent's _config with more specific type for type checker
        self._config: ModalBackendConfig = config
        self._app: modal.App | None = None

    def _configure_auth(self) -> None:
        """Set Modal auth environment variables if custom tokens are provided."""
        if self._config.token_id is not None:
            os.environ["MODAL_TOKEN_ID"] = self._config.token_id
        if self._config.token_secret is not None:
            os.environ["MODAL_TOKEN_SECRET"] = self._config.token_secret

    async def _validate(self) -> None:
        """Validate Modal prerequisites.

        Checks that Modal authentication is configured and the API is reachable.
        """
        # Configure auth from CLI args (if provided)
        self._configure_auth()
        modal_sdk = _get_modal_sdk()

        try:
            self._app = await modal_sdk.App.lookup.aio(
                self._config.app_name,
                create_if_missing=True,
            )
        except modal_sdk.exception.AuthError:
            raise BackendError(
                "Modal authentication failed. Either:\n"
                "  - Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables\n"
                "  - Run 'modal token set' to configure credentials\n"
                "  - Pass --modal-token-id and --modal-token-secret"
            )
        except modal_sdk.exception.ConnectionError:
            raise BackendError(
                "Cannot connect to Modal. Check your network connection "
                "and Modal service status at https://status.modal.com"
            )

    def _build_image(self) -> modal.Image:
        """Construct the Modal Image from configuration."""
        modal_sdk = _get_modal_sdk()
        if self._config.image is None:
            return modal_sdk.Image.debian_slim()
        else:
            return modal_sdk.Image.from_registry(self._config.image)

    async def _create_sandbox(self) -> ModalSandbox:
        """Create a Modal sandbox.

        Performs the full startup sequence:
        1. Build image
        2. Build GPU config (if specified)
        3. Create sandbox
        4. Create sandbox wrapper
        5. Verify readiness (cleanup if fails)
        """
        # 1. Build image
        image = self._build_image()
        modal_sdk = _get_modal_sdk()

        # 2. Build GPU config (if specified)
        gpu_config = self._config.gpu

        # 3. Create sandbox
        try:
            sb = await modal_sdk.Sandbox.create.aio(
                app=self._app,
                image=image,
                timeout=self._config.sandbox_timeout,
                cpu=self._config.cpu,
                memory=self._config.memory,
                gpu=gpu_config,
                region=self._config.region,
                block_network=not self._config.network_enabled,
            )
        except modal_sdk.exception.InvalidError as e:
            raise BackendError(f"Failed to create Modal sandbox: {e}")
        except modal_sdk.exception.NotFoundError:
            raise BackendError(
                f"Modal image not found: '{self._config.image}'. "
                f"Check that the image name is correct and the registry is accessible."
            )

        # 4. Create sandbox wrapper
        sandbox = ModalSandbox(
            modal_sandbox=sb,
            shell=self._config.shell,
        )

        # 5. Verify readiness
        try:
            await sandbox._verify_readiness()
        except Exception:
            await sandbox.stop()
            raise

        return sandbox

    def tool_instructions(self) -> str | None:
        """Return tool description for Modal backend.

        Returns None if using a custom image (baked-in description only
        applies to default debian_slim image).
        """
        if self._config.image is not None:
            return None

        shell_name = self._config.shell.rsplit("/", 1)[-1]  # basename
        timeout = self._config.default_timeout

        return (
            f"Execute a shell command in a remote cloud sandbox (Modal). "
            f"Commands run in {shell_name}. Each call is independent — "
            f"no state (shell variables, working directory) persists between calls. Use the working_directory "
            f"parameter or chain commands with && to control execution context."
            f"\n\n"
            f"To write files or pass data without shell escaping, use the "
            f'stdin parameter (e.g., command="cat > file.txt" with content '
            f"in stdin). Commands time out after {timeout} seconds by default "
            f"(override with the timeout parameter for long-running operations)."
        )


class ModalSandbox(Sandbox):
    """Modal sandbox implementation.

    Wraps a running Modal Sandbox. Handles command execution with
    timeout and output-limit enforcement, stop, and death detection.
    """

    def __init__(
        self,
        *,
        modal_sandbox: modal.Sandbox,
        shell: str,
    ) -> None:
        self._modal_sandbox = modal_sandbox
        self._shell = shell
        self._stopped = False
        self._stop_requested = False
        self._exec_lock = asyncio.Lock()

    @property
    def sandbox_id(self) -> str:
        """Return Modal's object_id for this sandbox."""
        return self._modal_sandbox.object_id

    async def _verify_readiness(self) -> None:
        """Verify the sandbox accepts exec calls and the shell works."""
        try:
            p = await self._modal_sandbox.exec.aio(
                self._shell,
                "-c",
                "echo kilntainers-ready",
                timeout=10,
            )
            stdout = await p.stdout.read.aio()
        except Exception as e:
            raise BackendError(f"Sandbox readiness check failed: {e}")

        if "kilntainers-ready" not in stdout:
            raise BackendError(
                f"Sandbox started but readiness check failed (unexpected output). "
                f"Shell '{self._shell}' may not be available in the image. "
                f"Use --shell /bin/sh for images without bash."
            )

    def _build_exec_args(self, request: ExecRequest) -> list[str]:
        """Build the positional args for Modal Sandbox.exec()."""
        if request.command is not None:
            # Command mode: wrap in shell
            return [self._shell, "-c", request.command]
        else:
            # Args mode: pass directly
            assert request.args is not None
            return request.args

    def _build_exec_kwargs(self, request: ExecRequest) -> dict:
        """Build keyword args for Modal Sandbox.exec()."""
        kwargs: dict = {
            "timeout": request.timeout,
        }
        if request.working_directory is not None:
            kwargs["workdir"] = request.working_directory
        return kwargs

    async def _read_streams(
        self,
        process: modal.container_process.ContainerProcess,
        output_limit: int,
    ) -> tuple[str, str]:
        """Read stdout and stderr with combined limit enforcement.

        Raises _OutputLimitExceeded if combined output exceeds the limit.
        """
        total_bytes = 0
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        async def read_stream(
            stream,
            chunks: list[str],
        ) -> None:
            nonlocal total_bytes
            async for line in stream:
                line_bytes = len(line.encode("utf-8"))
                total_bytes += line_bytes
                if total_bytes > output_limit:
                    raise _OutputLimitExceeded()
                chunks.append(line)

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(read_stream(process.stdout, stdout_chunks))
                tg.create_task(read_stream(process.stderr, stderr_chunks))
        except* _OutputLimitExceeded:
            # When the output limit is exceeded, raise a single exception
            # The except* syntax catches _OutputLimitExceeded from any task
            raise _OutputLimitExceeded()

        return "".join(stdout_chunks), "".join(stderr_chunks)

    async def _read_output_with_limit(
        self,
        process: modal.container_process.ContainerProcess,
        output_limit: int,
        request_timeout: int,
    ) -> tuple[str, str]:
        """Read process output with combined size enforcement.

        Returns (stdout_str, stderr_str).
        Raises _OutputLimitExceeded if combined output exceeds the limit.
        Raises asyncio.TimeoutError if client-side safety timeout fires.
        """
        return await asyncio.wait_for(
            self._read_streams(process, output_limit),
            timeout=request_timeout + 1,  # safety buffer
        )

    async def _do_exec(self, request: ExecRequest) -> ExecResult:
        """Core exec implementation."""
        exec_args = self._build_exec_args(request)
        exec_kwargs = self._build_exec_kwargs(request)
        modal_sdk = _get_modal_sdk()

        start_time = time.monotonic()

        try:
            process = await self._modal_sandbox.exec.aio(
                *exec_args,
                **exec_kwargs,
            )

            # Write stdin if provided
            if request.stdin is not None:
                process.stdin.write(request.stdin.encode("utf-8"))
                process.stdin.write_eof()
                await process.stdin.drain.aio()

            # Read output with limit enforcement
            stdout_str, stderr_str = await self._read_output_with_limit(
                process,
                request.output_limit,
                request.timeout,
            )

            # Wait for process to finish and get exit code.
            # Client-side safety timeout prevents indefinite hang if
            # Modal's API stalls after output streams close.
            await asyncio.wait_for(
                process.wait.aio(),
                timeout=request.timeout + 1,
            )
            exit_code = process.returncode

            elapsed_ms = int((time.monotonic() - start_time) * 1000)

            return ExecResult(
                stdout=stdout_str,
                stderr=stderr_str,
                exit_code=exit_code,
                exec_duration_ms=elapsed_ms,
            )

        except _OutputLimitExceeded:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            return ExecResult(
                stdout="",
                stderr=(
                    f"[kilntainers: output limit exceeded "
                    f"({request.output_limit} bytes). Command terminated. "
                    f"No output returned. Re-run with head, tail, or grep "
                    f"to manage output size.]"
                ),
                exit_code=1,
                exec_duration_ms=elapsed_ms,
            )

        except asyncio.TimeoutError:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            return ExecResult(
                stdout="",
                stderr=f"[kilntainers: command timed out after {request.timeout}s]",
                exit_code=124,
                exec_duration_ms=elapsed_ms,
            )

        except modal_sdk.exception.SandboxTerminatedError:
            if not self._stop_requested:
                raise SandboxDiedError(
                    f"Sandbox {self.sandbox_id} died during command execution"
                )
            raise SandboxDiedError("Sandbox has been stopped")

        except modal_sdk.exception.SandboxTimeoutError:
            # Sandbox lifetime timeout expired (not exec timeout)
            raise SandboxDiedError(
                f"Sandbox {self.sandbox_id} lifetime timeout expired"
            )

    async def exec(self, request: ExecRequest) -> ExecResult:
        """Execute a command in the sandbox.

        Uses a lock to serialize exec calls within this sandbox.
        """
        if self._stopped:
            raise SandboxDiedError("Sandbox has been stopped")

        async with self._exec_lock:
            return await self._do_exec(request)

    async def stop(self) -> None:
        """Stop the sandbox and release all resources.

        Idempotent — safe to call on an already-stopped sandbox.
        """
        if self._stopped:
            return
        self._stopped = True
        self._stop_requested = True

        try:
            await asyncio.wait_for(
                self._modal_sandbox.terminate.aio(),
                timeout=10,
            )
        except asyncio.TimeoutError:
            pass  # Best-effort — Modal may take time to terminate
        except Exception:
            pass  # Best-effort cleanup

    async def wait_for_death(self) -> None:
        """Block until the sandbox dies unexpectedly.

        Resolves when the sandbox finishes for reasons other than
        stop() being called. Does not return when stop() is called —
        in that case, blocks until cancelled by the MCP layer.
        """
        try:
            await self._modal_sandbox.wait.aio(raise_on_termination=False)
        except asyncio.CancelledError:
            # Normal shutdown — MCP layer cancelled this task before stop()
            raise

        if self._stop_requested:
            # Sandbox exited because stop() was called — expected.
            # Block forever; the MCP layer will cancel this task.
            try:
                await asyncio.Future()  # never completes
            except asyncio.CancelledError:
                return

        # Sandbox exited unexpectedly (OOM, lifetime timeout, preemption).
        # Returning signals death to the MCP layer.
