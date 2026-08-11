"""CLI argument parsing and main entry point."""

import argparse
import os
import signal
import sys
import threading
from typing import NoReturn

from kilntainers.backends import (
    get_available_backend_names,
    get_backend_class,
)
from kilntainers.config import BackendConfig, ServerConfig
from kilntainers.errors import BackendError
from kilntainers.server import create_server

# Sentinel for detecting unset HTTP-only arguments
_UNSET = object()


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        An ArgumentParser with all kilntainers arguments organized into groups.
    """
    parser = argparse.ArgumentParser(
        prog="kilntainers",
        description=(
            "MCP server providing isolated Linux sandboxes "
            "for LLM agent shell execution."
        ),
        usage="%(prog)s [-h] [--backend {docker,go_busybox,modal,wasm}] [--transport {stdio,http}] [...]",
    )

    # Get available backend names for choices
    available_backends = get_available_backend_names()

    # --- Core parameters ---
    core = parser.add_argument_group("core options")
    core.add_argument(
        "--backend",
        default="docker",
        choices=available_backends,
        help=f"Backend to use (default: docker). Available: {', '.join(available_backends)}",
    )
    core.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "http"],
        help="MCP transport (default: stdio)",
    )
    core.add_argument(
        "--host",
        default=_UNSET,
        help="HTTP bind address (default: 127.0.0.1, HTTP mode only)",
    )
    core.add_argument(
        "--port",
        type=int,
        default=_UNSET,
        help="HTTP listen port (default: 8435, HTTP mode only)",
    )
    core.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Default exec timeout in seconds (default: 120)",
    )
    core.add_argument(
        "--output-limit",
        type=int,
        default=2_097_152,
        help="Max combined stdout+stderr bytes per exec (default: 2097152 = 2 MiB)",
    )
    core.add_argument(
        "--session-timeout",
        type=int,
        default=_UNSET,
        help="Idle session timeout in seconds (default: 300, HTTP mode only)",
    )
    core.add_argument(
        "--shell",
        default="/bin/bash",
        help="Shell binary for command mode (e.g., /bin/bash, ash). Default: /bin/bash.",
    )
    core.add_argument(
        "--network",
        action="store_true",
        default=False,
        help="Enable network access in sandboxes (default: disabled)",
    )

    # --- Tool description ---
    desc = parser.add_argument_group("tool description")
    desc.add_argument(
        "--tool-instruction-override",
        default=None,
        help="Replace the entire sandbox_exec tool description",
    )
    desc.add_argument(
        "--extended-tool-instruction",
        default=None,
        help="Append to the backend's default tool description",
    )

    # --- Backend-specific parameters (delegated to each backend) ---
    for name in available_backends:
        group = parser.add_argument_group(f"{name} backend options")
        try:
            backend_cls = get_backend_class(name)
            backend_cls.add_cli_arguments(group)
        except BackendError:
            # Dependencies not installed — skip this backend's CLI args.
            # The backend name still appears in --backend choices so the
            # user can discover it, and gets a clear install message if selected.
            pass

    return parser


def build_configs(
    args: argparse.Namespace,
) -> tuple[ServerConfig, BackendConfig]:
    """Build config dataclasses from parsed arguments.

    This function maps flat CLI arguments to the typed config objects
    consumed by the server and backend layers. Server config is built
    here; backend config is delegated to the backend class.

    Args:
        args: Parsed command-line arguments from argparse.

    Returns:
        A tuple of (ServerConfig, BackendConfig).
    """
    # Handle HTTP-only args that may be _UNSET
    host = "127.0.0.1" if args.host is _UNSET else args.host
    port = 8435 if args.port is _UNSET else args.port
    session_timeout = 300 if args.session_timeout is _UNSET else args.session_timeout

    server_config = ServerConfig(
        transport=args.transport,
        host=host,
        port=port,
        default_timeout=args.timeout,
        output_limit=args.output_limit,
        tool_instruction_override=args.tool_instruction_override,
        extended_tool_instruction=args.extended_tool_instruction,
        session_timeout=session_timeout,
    )

    # Delegate backend config construction to the backend class
    backend_cls = get_backend_class(args.backend)
    backend_config = backend_cls.config_from_args(args)

    return server_config, backend_config


def _startup_error(message: str) -> NoReturn:
    """Write an error message to stderr and exit with code 1.

    Used for all startup/configuration errors.

    Args:
        message: The error message to display.

    Raises:
        SystemExit: Always exits with code 1.
    """
    sys.stderr.write(f"kilntainers: error: {message}\n")
    sys.exit(1)


def validate_config(server_config: ServerConfig) -> None:
    """Validate configuration constraints that span multiple parameters.

    Raises SystemExit with a descriptive message on failure.
    Individual argument type validation is handled by argparse.
    Cross-cutting constraints are checked here.

    Args:
        server_config: The server configuration to validate.

    Raises:
        SystemExit: If validation fails, with code 1 and an error message.
    """
    # HTTP-only parameters in stdio mode
    # When called from main(), _parsed_args contains the actual parsed args
    # When called from tests, we rely on comparing config values to defaults
    if server_config.transport == "stdio":
        # Check if user explicitly passed HTTP-only args
        # We check if values differ from defaults, indicating explicit setting
        if server_config.host != "127.0.0.1":
            _startup_error(
                "--host is only valid with --transport http. "
                "In stdio mode, there is no HTTP server to bind."
            )
        if server_config.port != 8435:
            _startup_error(
                "--port is only valid with --transport http. "
                "In stdio mode, there is no HTTP server to bind."
            )
        if server_config.session_timeout != 300:
            _startup_error(
                "--session-timeout is only valid with --transport http. "
                "In stdio mode, the session lives as long as the process."
            )

    # Mutual exclusivity: tool description params
    if (
        server_config.tool_instruction_override is not None
        and server_config.extended_tool_instruction is not None
    ):
        _startup_error(
            "Cannot use both --tool-instruction-override and "
            "--extended-tool-instruction. Use override to replace "
            "the description entirely, or extended to append to "
            "the backend default."
        )

    # Timeout must be positive
    if server_config.default_timeout < 1:
        _startup_error("--timeout must be at least 1 second.")

    # Output limit must be positive
    if server_config.output_limit < 1:
        _startup_error("--output-limit must be at least 1 byte.")


async def _async_main(
    server_config: ServerConfig,
    backend_config: BackendConfig,
    backend_name: str,
) -> None:
    """Async startup: build server, run.

    This function performs all async startup operations:
    - Creates the backend (validation happens lazily)
    - Creates the MCP server
    - Runs the transport (blocking until shutdown)

    Args:
        server_config: Server configuration.
        backend_config: Backend configuration.
        backend_name: Name of the backend to use.

    Raises:
        SystemExit: If server creation fails.
    """
    # Create backend (validation happens lazily on first sandbox_exec)
    backend_class = get_backend_class(backend_name)
    backend = backend_class(backend_config)

    # Create the MCP server (assembles tool description, registers tool)
    try:
        mcp = create_server(backend, server_config)
    except BackendError as e:
        _startup_error(str(e))

    # Run the transport (blocks until shutdown)
    transport = "stdio" if server_config.transport == "stdio" else "streamable-http"
    mcp.run(transport=transport)


def main() -> None:
    """CLI entry point. Parses args, configures, and runs the server.

    This is the main entry point for the kilntainers command. It:
    1. Parses CLI arguments
    2. Builds configuration objects
    3. Validates configuration constraints
    4. Creates the backend (validation happens lazily on first exec)
    5. Creates and runs the MCP server

    Never returns normally (exits on KeyboardInterrupt or server shutdown).
    """
    parser = build_parser()
    args = parser.parse_args()

    server_config, backend_config = build_configs(args)
    validate_config(server_config)

    # Create backend (validation happens lazily on first sandbox_exec)
    backend_name = args.backend
    backend_class = get_backend_class(backend_name)
    backend_class.prepare_runtime()
    backend = backend_class(backend_config)

    # Create the MCP server (assembles tool description, registers tool)
    try:
        mcp = create_server(backend, server_config)
    except BackendError as e:
        _startup_error(str(e))

    # Run the transport (blocks until shutdown)
    # mcp.run() manages its own event loop, so we call it directly
    transport = "stdio" if server_config.transport == "stdio" else "streamable-http"

    # Register SIGTERM handler to convert to SIGINT for clean shutdown
    # FastMCP handles SIGINT (Ctrl+C) gracefully, so we redirect SIGTERM to the same path
    def _handle_sigterm(signum: int, frame: object) -> None:
        """
        Convert SIGTERM to SIGINT for clean shutdown (triggers mcp library graceful shutdown).

        Watchdog timmer needed as `mcp` library doesn't exit on SIGTERM.

        `mcp` library is adding sigterm support, but not in a release yet.
        """
        # Schedule forced exit as fallback in case graceful shutdown hangs.
        # Uses a daemon thread so it won't block normal exit if shutdown succeeds.
        timer = threading.Timer(5.0, lambda: os._exit(0))
        timer.daemon = True
        timer.start()
        os.kill(os.getpid(), signal.SIGINT)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _handle_sigterm)

    try:
        mcp.run(transport=transport)
    except KeyboardInterrupt:
        pass  # Clean exit on Ctrl+C
