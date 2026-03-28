"""
MCP Client: spins up MCP servers and calls tools via JSON-RPC over stdio.
Supports synthetic tool variants with controlled failure injection.
"""
import json
import random
import subprocess
import os
import select
import time
from pathlib import Path


def _load_synthetic_configs():
    """Load synthetic tool configs for failure injection."""
    config_path = Path(__file__).parent.parent / "data" / "synthetic_configs.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


_SYNTHETIC_CONFIGS = _load_synthetic_configs()


class MCPClient:
    """Manages MCP server processes and executes tool calls via JSON-RPC."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def call_tool(self, server_id: str, tool_name: str, arguments: dict,
                  install: dict) -> dict:
        """Agent calls the tools, gets the information.

        For synthetic tools, routes to the real underlying tool and applies
        failure injection based on the synthetic config.
        """
        # Check if this is a synthetic tool
        synth_cfg = _SYNTHETIC_CONFIGS.get(server_id)
        if synth_cfg:
            return self._call_synthetic(server_id, tool_name, arguments,
                                        install, synth_cfg)

        return self._call_real(server_id, tool_name, arguments, install)

    def _call_synthetic(self, server_id: str, tool_name: str, arguments: dict,
                        install: dict, cfg: dict) -> dict:
        """Route synthetic tool to real tool with failure injection.

        Failure modes:
          - random_failure: returns error X% of the time
          - truncate: calls real tool but truncates result to N chars
          - stale: returns canned irrelevant response X% of the time
          - niche: fails if arguments don't match specific patterns
        """
        fail_type = cfg["type"]

        # Pre-call failures (don't even call the real tool)
        if fail_type == "random_failure":
            if random.random() < cfg["fail_rate"]:
                return {
                    "content": [{"type": "text", "text": "Error: Service temporarily unavailable. Please try again later."}],
                    "isError": True,
                }

        elif fail_type == "stale":
            if random.random() < cfg.get("stale_rate", 0.6):
                return {
                    "content": [{"type": "text", "text": cfg["stale_response"]}],
                    "isError": True,
                }

        elif fail_type == "niche":
            arg_str = json.dumps(arguments).lower()
            matches = any(pattern.lower() in arg_str for pattern in cfg["works_for"])
            fail_rate = cfg["fail_rate_match"] if matches else cfg["fail_rate_mismatch"]
            if random.random() < fail_rate:
                return {
                    "content": [{"type": "text", "text": "Error: Unable to process this input type."}],
                    "isError": True,
                }

        # Call the real underlying tool
        real_server_id = cfg["real_server_id"]
        real_tool_name = cfg["real_tool_name"]
        real_install = self._get_real_install(real_server_id)
        if not real_install:
            return {"error": f"Real server {real_server_id} not found"}

        result = self._call_real(real_server_id, real_tool_name, arguments, real_install)

        # Post-call degradation (modify the real result)
        if fail_type == "truncate":
            max_chars = cfg.get("max_chars", 100)
            if "content" in result:
                for item in result["content"]:
                    if item.get("type") == "text" and len(item.get("text", "")) > max_chars:
                        item["text"] = item["text"][:max_chars] + "... [content truncated]"
            elif isinstance(result, dict) and "result" in result:
                text = str(result["result"])
                if len(text) > max_chars:
                    result["result"] = text[:max_chars] + "... [content truncated]"

        return result

    def _get_real_install(self, server_id: str) -> dict:
        """Look up install config for a real (non-synthetic) server."""
        tools_path = Path(__file__).parent.parent / "data" / "tools.json"
        with open(tools_path) as f:
            servers = json.load(f)
        for s in servers:
            if s["id"] == server_id and "_synthetic" not in s:
                return s.get("install", {})
        return {}

    def _call_real(self, server_id: str, tool_name: str, arguments: dict,
                   install: dict) -> dict:
        """Call a real MCP tool via JSON-RPC."""
        command = install.get("command", [])
        env_vars = install.get("env", {})

        if not command:
            return {"error": f"No install command for server {server_id}"}

        # Build environment with any required env vars (API keys etc.)
        env = os.environ.copy()
        for k, v in env_vars.items():
            env[k] = v

        proc = None
        try:
            # Spin up the MCP server process
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env,
            )

            # Send initialize request
            self._send(proc, {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "toolflix", "version": "0.1.0"},
                }
            })

            # Wait for init response
            self._read_until_id(proc, target_id=1)

            # Send initialized notification
            self._send(proc, {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            })

            # Send tool call request
            self._send(proc, {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments,
                }
            })

            # Wait for tool call response
            result = self._read_until_id(proc, target_id=2)

            # Clean up
            proc.stdin.close()
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

            if result is None:
                return {"error": "No response for tool call"}

            if "error" in result:
                return {"error": result["error"]}

            return result.get("result", {})

        except TimeoutError:
            proc.kill()
            proc.wait()
            return {"error": f"Timeout after {self.timeout}s"}
        except FileNotFoundError:
            return {"error": f"Command not found: {command[0]}"}
        except Exception as e:
            if proc:
                try:
                    proc.kill()
                    proc.wait()
                except OSError:
                    pass
            return {"error": str(e)}

    def _send(self, proc: subprocess.Popen, msg: dict):
        """Send a JSON-RPC message to the server."""
        data = json.dumps(msg) + "\n"
        proc.stdin.write(data.encode())
        proc.stdin.flush()

    def _read_until_id(self, proc: subprocess.Popen, target_id: int) -> dict | None:
        """Read stdout lines until we get a response with the target id."""
        deadline = time.time() + self.timeout
        buffer = ""

        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError()

            # Check if there's data to read
            ready, _, _ = select.select([proc.stdout], [], [], min(remaining, 1.0))
            if not ready:
                # Check if process died
                if proc.poll() is not None:
                    return None
                continue

            chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, 'read1') else os.read(proc.stdout.fileno(), 4096)
            if not chunk:
                return None

            buffer += chunk.decode("utf-8", errors="replace")

            # Try to parse complete lines
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    if msg.get("id") == target_id:
                        return msg
                except json.JSONDecodeError:
                    continue

        raise TimeoutError()
