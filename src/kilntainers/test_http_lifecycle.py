"""Unit tests for HTTP lifecycle components.

These tests don't require a running server or Docker.
"""

from unittest.mock import MagicMock

import pytest

from kilntainers.backends.test_docker_integration import validate_engine_available
from kilntainers.backends.test_utils import MockBackend
from kilntainers.config import BackendConfig, ServerConfig
from kilntainers.server import create_lifespan, create_server


class TestHTTPLifespan:
    """Tests for HTTP-specific lifespan behavior."""

    @pytest.mark.asyncio
    async def test_http_lifespan_creates_sandbox_lazily(self):
        """HTTP lifespan creates a sandbox lazily per session."""
        backend = MockBackend(BackendConfig())
        lifespan_fn = create_lifespan(backend, "http")
        mock_server = MagicMock()

        async with lifespan_fn(mock_server) as ctx:
            # Sandbox is None initially (lazy creation)
            assert ctx.sandbox is None
            assert ctx.death_task is None

            # Create sandbox lazily
            await ctx.get_or_create_sandbox()
            assert ctx.sandbox is not None
            assert ctx.death_task is not None
            assert not ctx.death_task.cancelled()

        # After exit, death task should be cancelled
        assert ctx.death_task.cancelled()

    @pytest.mark.asyncio
    async def test_http_lifespan_no_sigterm_on_death(self):
        """HTTP lifespan does NOT send SIGTERM when sandbox dies."""
        import asyncio
        import os
        import signal

        backend = MockBackend(BackendConfig())
        sigterm_calls: list[tuple[int, int]] = []

        # Mock os.kill to capture calls
        original_kill = os.kill

        def mock_kill(pid: int, sig: int) -> None:  # type: ignore[assignment]
            if sig == signal.SIGTERM:
                sigterm_calls.append((pid, sig))

        os.kill = mock_kill  # type: ignore[assignment]

        try:
            lifespan_fn = create_lifespan(backend, "http")
            mock_server = MagicMock()

            async with lifespan_fn(mock_server) as ctx:
                # Create sandbox first to start death monitoring
                await ctx.get_or_create_sandbox()

                # Simulate sandbox death via the MockSandbox's method
                mock_sandbox = ctx.sandbox
                if hasattr(mock_sandbox, "simulate_death"):
                    mock_sandbox.simulate_death()  # type: ignore[attr-defined]

                # Wait a bit for death task to process
                await asyncio.sleep(0.2)

        finally:
            os.kill = original_kill  # type: ignore[assignment]

        # In HTTP mode, SIGTERM should NOT be sent
        assert len(sigterm_calls) == 0, (
            "HTTP mode should not send SIGTERM on sandbox death"
        )


class TestHTTPServerCreation:
    """Tests for creating server with HTTP transport config."""

    def test_create_server_with_http_config(self):
        """create_server works with HTTP transport configuration."""
        backend = MockBackend(BackendConfig())
        config = ServerConfig(
            transport="http",
            host="0.0.0.0",
            port=9000,
            session_timeout=600,
        )

        mcp = create_server(backend, config)

        assert mcp is not None
        assert mcp.name == "Kilntainers"

    def test_create_server_with_stdio_config(self):
        """create_server works with stdio transport configuration."""
        backend = MockBackend(BackendConfig())
        config = ServerConfig(
            transport="stdio",
        )

        mcp = create_server(backend, config)

        assert mcp is not None
        assert mcp.name == "Kilntainers"


class TestHTTPSessionIsolation:
    """E2E tests for HTTP session isolation between clients.

    In HTTP mode, each connection/session gets its own lifespan context,
    which creates a separate sandbox. These tests verify that separate
    connections get separate VMs with isolated state.
    """

    @pytest.mark.e2e
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_separate_http_connections_get_separate_vms(self):
        """Separate HTTP connections get separate VMs with isolated state.

        E2E test that starts an actual HTTP server and connects with 2 MCP clients
        via streamable-http. Verifies that:
        1. Each HTTP connection gets a separate sandbox/VM
        2. VMs persist across requests within the same connection (reuse)
        3. Changes in one VM are not visible in another

        Test scenario (interleaved):
        - Client 1: touch a.txt
        - Client 2: touch b.txt
        - Client 1: ls (should see a.txt only)
        - Client 2: ls (should see b.txt only)
        """
        import asyncio
        import json
        import signal
        import socket
        import subprocess
        import sys
        import uuid

        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        # Skip if docker is not available
        validate_engine_available("docker")

        # Reserve a dynamic port to avoid conflicts between parallel workers.
        with socket.socket() as port_socket:
            port_socket.bind(("127.0.0.1", 0))
            server_port = port_socket.getsockname()[1]
        test_container_label = f"kilntainers-http-test={uuid.uuid4().hex}"

        # Use the current interpreter directly so terminating this process also
        # terminates the server, rather than leaving a child behind an `uv run`
        # wrapper.
        server_proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "kilntainers",
            "--transport",
            "http",
            "--host",
            "127.0.0.1",
            "--port",
            str(server_port),
            f"--docker-run-flag=--label={test_container_label}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            ),
        )

        # Wait until the server accepts connections, with a bounded deadline.
        startup_deadline = asyncio.get_running_loop().time() + 15
        server_ready = False
        while True:
            if server_proc.returncode is not None:
                break
            try:
                _reader, writer = await asyncio.open_connection(
                    "127.0.0.1", server_port
                )
                writer.close()
                await writer.wait_closed()
                server_ready = True
                break
            except OSError:
                if asyncio.get_running_loop().time() >= startup_deadline:
                    break
                await asyncio.sleep(0.1)

        # Check if server started successfully
        if server_proc.returncode is not None:
            stderr_str = ""
            if server_proc.stderr is not None:
                stderr_output = await server_proc.stderr.read()
                stderr_str = stderr_output.decode() if stderr_output else ""
            raise RuntimeError(
                f"Server failed to start. Return code: {server_proc.returncode}. Stderr: {stderr_str}"
            )
        if not server_ready:
            server_proc.terminate()
            try:
                await asyncio.wait_for(server_proc.wait(), timeout=15)
            except TimeoutError:
                server_proc.kill()
                await server_proc.wait()
            raise RuntimeError("Server did not accept connections within 15 seconds")

        server_url = f"http://127.0.0.1:{server_port}/mcp"

        # Events to coordinate execution order (no sleep/timing dependencies)
        client1_ready = asyncio.Event()
        client2_ready = asyncio.Event()
        client1_done_touch = asyncio.Event()
        client2_done_touch = asyncio.Event()

        results_client1: list[dict] = []
        results_client2: list[dict] = []

        # Client 1 session
        async def client1_session():
            async with streamable_http_client(server_url) as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    # Initialize
                    await session.initialize()

                    client1_ready.set()
                    await client2_ready.wait()  # Wait for client 2 to be ready

                    # Client 1: touch a.txt
                    result = await session.call_tool(
                        "sandbox_exec", {"command": "touch a.txt"}
                    )
                    results_client1.append({"cmd": "touch a.txt", "result": result})
                    client1_done_touch.set()

                    await client2_done_touch.wait()  # Wait for client 2 to touch b.txt

                    # Client 1: ls (should see a.txt)
                    result = await session.call_tool("sandbox_exec", {"command": "ls"})
                    results_client1.append({"cmd": "ls", "result": result})

        # Client 2 session
        async def client2_session():
            async with streamable_http_client(server_url) as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    # Initialize
                    await session.initialize()

                    client2_ready.set()
                    await client1_ready.wait()  # Wait for client 1 to be ready

                    # Client 2: touch b.txt
                    result = await session.call_tool(
                        "sandbox_exec", {"command": "touch b.txt"}
                    )
                    results_client2.append({"cmd": "touch b.txt", "result": result})
                    client2_done_touch.set()

                    await client1_done_touch.wait()  # Wait for client 1 to touch a.txt

                    # Client 2: ls (should see b.txt, NOT a.txt)
                    result = await session.call_tool("sandbox_exec", {"command": "ls"})
                    results_client2.append({"cmd": "ls", "result": result})

        try:
            # Run both client sessions concurrently
            await asyncio.gather(client1_session(), client2_session())

            # Client 1 should see a.txt in its ls output
            client1_ls_result = results_client1[1]["result"]
            client1_ls_text = client1_ls_result.content[0].text
            client1_ls_data = json.loads(client1_ls_text)
            assert client1_ls_data["exit_code"] == 0, "Client 1 ls failed"
            assert "a.txt" in client1_ls_data["stdout"], "Client 1 should see a.txt"
            assert "b.txt" not in client1_ls_data["stdout"], (
                "Client 1 should NOT see b.txt"
            )

            # Client 2 should see b.txt in its ls output
            client2_ls_result = results_client2[1]["result"]
            client2_ls_text = client2_ls_result.content[0].text
            client2_ls_data = json.loads(client2_ls_text)
            assert client2_ls_data["exit_code"] == 0, "Client 2 ls failed"
            assert "b.txt" in client2_ls_data["stdout"], "Client 2 should see b.txt"
            assert "a.txt" not in client2_ls_data["stdout"], (
                "Client 2 should NOT see a.txt"
            )

        finally:
            # Clean up server
            if server_proc.returncode is None:
                if sys.platform == "win32":
                    server_proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    server_proc.terminate()
                try:
                    await asyncio.wait_for(server_proc.wait(), timeout=15)
                except TimeoutError:
                    server_proc.kill()
                    await server_proc.wait()
            else:
                await server_proc.wait()

            # Session shutdown is best-effort if the server is interrupted.
            # Remove only containers uniquely labeled by this test run.
            container_query = await asyncio.create_subprocess_exec(
                "docker",
                "ps",
                "-aq",
                "--filter",
                f"label={test_container_label}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            query_stdout, _query_stderr = await asyncio.wait_for(
                container_query.communicate(), timeout=10
            )
            container_ids = query_stdout.decode().split()
            if container_ids:
                container_stop = await asyncio.create_subprocess_exec(
                    "docker",
                    "stop",
                    *container_ids,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(
                    container_stop.communicate(),
                    timeout=20,
                )
