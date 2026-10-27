"""Letta Code bridge with local context state or persistent cloud agents."""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
import logging
import os
import shutil
import signal
import subprocess
import tarfile
import tempfile
import threading
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, web
from jsonschema import Draft202012Validator, SchemaError, ValidationError

logger = logging.getLogger(__name__)

# The model a run uses when neither the task nor the agent-config names one; the
# local backend forwards it verbatim as the chat model.
DEFAULT_MODEL = "openai/gpt-4.1-mini"
ERROR_MESSAGE_LIMIT = 4000
ROLE_HEADER = "AgentEnv-Role"
# What agent-env gives an agent for [model.headers]: a JSON object of header names to values.
MODEL_HEADERS_ENV = "LITELLM_EXTRA_HEADERS"

# Root under which each A2A context_id gets its own Letta Code data dir; lives
# under $HOME so host-mode instances don't share session state.
SESSIONS_ROOT = os.path.join(os.environ.get("HOME", "/data"), "sessions")
_CLI_TIMEOUT_S = 120


def _base_url(environ: Mapping[str, str]) -> str:
    """The endpoint's chat-completions root: agent-env's LITELLM_BASE_URL has no /v1 suffix."""
    base = (environ.get("LITELLM_BASE_URL") or "").rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def context_data_dir(context_id: str) -> Path:
    """Per-context Letta Code data dir. A stable short hash keeps the path clean
    and filesystem-safe regardless of the context_id's charset."""
    ctx_hash = hashlib.sha1(context_id.encode()).hexdigest()[:16]
    return Path(SESSIONS_ROOT) / ctx_hash


def _context_env(context_id: str, api_key: str | None = None, base_url: str = "https://api.letta.com") -> dict[str, str]:
    home = context_data_dir(context_id)
    home.mkdir(parents=True, exist_ok=True)
    environ = {key: value for key, value in os.environ.items()
               if not key.startswith("LETTA_")}
    if api_key:
        environ.update(LETTA_API_KEY=api_key, LETTA_BASE_URL=base_url, LETTA_BACKEND="api")
    else:
        environ["LETTA_BACKEND"] = "local"
    return {**environ, "HOME": str(home), "LETTA_LOCAL_BACKEND_DIR": str(home / ".letta/lc-local-backend"),
            "LETTA_LOCAL_BACKEND_EXPERIMENTAL": "0" if api_key else "1"}


async def _communicate(*cmd: str, env: Mapping[str, str], timeout_s: int) -> tuple[int, bytes, bytes]:
    """Run a short CLI command to its end; one that outlives ``timeout_s`` is killed."""
    proc = await _started(asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=dict(env), start_new_session=True,
    ))
    _RUNNING.add(proc)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except BaseException:
        _kill_cli(proc)
        await _reap(proc)
        raise
    finally:
        _RUNNING.discard(proc)
    return proc.returncode, out, err



async def export_session(context_id: str, session_id: str) -> bytes:
    """Capture the conversation, agent, git-backed memory and local settings together."""
    root = context_data_dir(context_id)
    (root / "snapshot-session.json").write_text(json.dumps({"session_id": session_id}))
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        agent_file = root / "agent-id.json"
        cloud = agent_file.exists() and json.loads(agent_file.read_text()).get("backend") == "cloud"
        if cloud:
            for name in ("agent-id.json", "context.json", "snapshot-session.json"):
                archive.add(root / name, arcname="state/" + name)
        else:
            archive.add(root, arcname="state")
    return buffer.getvalue()


async def import_session(context_id: str, json_path: str) -> str:
    """Stage a complete local state tree, then replace the target context's state."""
    root = context_data_dir(context_id)
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent) as directory:
        with tarfile.open(json_path, "r:gz") as archive:
            archive.extractall(directory, filter="data")
        staged = Path(directory) / "state"
        session_id = json.loads((staged / "snapshot-session.json").read_text())["session_id"]
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Snapshot has no conversation id")
        # Memory repos and settings may contain absolute paths to the original context.
        source = json.loads((staged / "context.json").read_text())["home"]
        for path in staged.rglob("*.json"):
            if ".git" not in path.parts:
                path.write_text(path.read_text().replace(source, str(root)))
        if root.exists():
            shutil.rmtree(root)
        shutil.move(str(staged), root)
    return session_id


@dataclass
class LettaCodeResult:
    response: str
    tool_call_count: int
    session_id: str | None
    trajectory: list[dict] = field(default_factory=list)
    error: str | None = None
    error_code: str | None = None
    structured_output: object | None = None


def validate_output_format(output_format: dict) -> dict:
    if output_format.get("type") != "json_schema" or not isinstance(output_format.get("schema"), dict):
        raise ValueError("output_format must contain type=json_schema and a schema object")
    try:
        Draft202012Validator.check_schema(output_format["schema"])
    except SchemaError as exc:
        raise ValueError("output_format contains an invalid JSON schema") from exc
    return output_format["schema"]


# The attribution the last run sent, which /ext/attribution-probe reads back so the
# validator can confirm project_id/task_id were wired through.
_last_attribution: dict[str, str | None] = {
    "project_id": None,
    "task_id": None,
    "last_seen_at_utc": None,
}
_attribution_lock = threading.Lock()


def _record_attribution(project_id: str | None, task_id: str | None) -> None:
    if not project_id and not task_id:
        return
    with _attribution_lock:
        _last_attribution["project_id"] = project_id or None
        _last_attribution["task_id"] = task_id or None
        _last_attribution["last_seen_at_utc"] = datetime.now(timezone.utc).isoformat()


def get_last_attribution_snapshot() -> dict[str, str | None]:
    with _attribution_lock:
        return dict(_last_attribution)


def _attribution_headers(project_id: str | None, task_id: str | None) -> dict[str, str]:
    """LiteLLM cost-attribution headers, x-litellm-customer-id + x-litellm-tags; {} when nothing to attribute."""
    headers: dict[str, str] = {}
    if project_id:
        headers["x-litellm-customer-id"] = project_id
        tags = [f"projectId:{project_id}"]
        if task_id:
            tags.append(f"taskId:{task_id}")
        headers["x-litellm-tags"] = ",".join(tags)
    elif task_id:
        headers["x-litellm-tags"] = f"taskId:{task_id}"
    return headers


def model_headers(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    try:
        headers = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"ignoring {MODEL_HEADERS_ENV}, which isn't JSON")
        return {}
    if not isinstance(headers, dict):
        logger.warning(f"ignoring {MODEL_HEADERS_ENV}, which isn't a JSON object")
        return {}
    return {str(name): str(value) for name, value in headers.items()}


def provider_headers(environ: Mapping[str, str], attribution: Mapping[str, str]) -> dict[str, str]:
    """The endpoint's [model.headers], then the run's attribution, each replacing a header of the same name in any
    case."""
    return {name.lower(): value for layer in (model_headers(environ.get(MODEL_HEADERS_ENV)), attribution)
            for name, value in layer.items()}


def _mcp_headers(headers: Mapping[str, str] | None, role: str | None) -> dict[str, str]:
    """A server's registered headers; the run's role, when it has one, replaces a role header they name."""
    merged = dict(headers or {})
    if role:
        merged = {name: value for name, value in merged.items() if name.lower() != ROLE_HEADER.lower()}
        merged[ROLE_HEADER] = role
    return merged


_PROCESS_CLEANUP_TIMEOUT_S = 5
_RUNNING: set = set()


def kill_running_clis() -> int:
    """Kill every CLI still running a turn, and what each started; how many. The agent's shutdown does: the CLIs, in
    sessions of their own, would outlive its server, as when a reinstall stops it mid-turn."""
    running = [proc for proc in _RUNNING if proc.returncode is None]
    for proc in running:
        _kill_cli(proc)
    return len(running)


def _kill_cli(proc) -> None:
    """Kill the CLI and everything under it: its process group, and, found by parent, the tools a CLI starts in a
    session of their own, which no group or session kill reaches."""
    pid = getattr(proc, "pid", None)
    if pid is None:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        return
    if proc.returncode is not None:
        return  # the CLI is gone, and its pid may be another process's by now
    tree = [pid, *_descendants(pid)]
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)
    for member in tree:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(member, signal.SIGKILL)


async def _started(spawn):
    """The CLI ``spawn`` starts. A turn cancelled while it starts lets it finish starting, then kills it and what it
    started: asyncio would kill the CLI alone, and leave its tools running."""
    start = asyncio.ensure_future(spawn)
    try:
        return await asyncio.shield(start)
    except asyncio.CancelledError:
        with contextlib.suppress(Exception):
            proc = await start
            _kill_cli(proc)
            await _reap(proc)
        raise


async def _reap(proc) -> None:
    """Wait for the killed CLI, but not for a process that left its tree and still holds its pipes open."""
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(proc.wait(), _PROCESS_CLEANUP_TIMEOUT_S)


def _descendants(pid: int) -> list[int]:
    """The processes under ``pid``, by parent, read from /proc; none where there is no /proc."""
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc") if os.path.isdir("/proc") else ():
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as stat:
                parent = int(stat.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(parent, []).append(int(entry))
    found, pending = [], [pid]
    while pending:
        for child in children.get(pending.pop(), ()):
            found.append(child)
            pending.append(child)
    return found



@asynccontextmanager
async def model_relay(headers: dict[str, str]):
    """Keep endpoint credentials and task attribution on requests to the configured model endpoint only."""
    base = _base_url(os.environ)
    async with ClientSession(timeout=ClientTimeout(total=None), auto_decompress=False) as client:
        async def forward(request):
            outgoing = {"authorization": "Bearer " + os.environ.get("LITELLM_API_KEY", ""), **headers}
            if request.content_type:
                outgoing["content-type"] = request.headers.get("Content-Type", "application/json")
            async with client.request(request.method, base + request.path_qs.removeprefix("/v1"),
                                      data=await request.read(), headers=outgoing) as upstream:
                response = web.StreamResponse(status=upstream.status, headers={
                    name: value for name, value in upstream.headers.items()
                    if name.lower() not in {"content-length", "transfer-encoding", "connection"}
                })
                await response.prepare(request)
                # SSE consumers commonly close immediately after the terminal event.
                with contextlib.suppress(ConnectionResetError):
                    async for chunk in upstream.content.iter_any():
                        await response.write(chunk)
                    await response.write_eof()
                return response
        app = web.Application(client_max_size=100 * 1024**2)
        app.router.add_route("*", "/{path:.*}", forward)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            yield f"http://127.0.0.1:{port}/v1"
        finally:
            await runner.cleanup()


def _cli(env: Mapping[str, str]) -> list[str]:
    return ["letta"] if env.get("LETTA_API_KEY") else ["letta", "--backend", "local"]


@asynccontextmanager
async def _provider(env, context_id, model, attribution):
    if not env.get("LETTA_API_KEY"):
        async with model_relay(provider_headers(os.environ, attribution)) as relay:
            await _command(env, "connect", "openai-compatible", "--base-url", relay, "--api-key", "relay")
            yield "openai-compatible/" + model
        return
    # The cloud provider API used by CLI 0.34.25 carries a URL and key, but no arbitrary headers.
    # Never silently discard endpoint headers (which may themselves be authentication).
    if model_headers(os.environ.get(MODEL_HEADERS_ENV)):
        raise ValueError("Letta Cloud BYOK cannot forward [model.headers]; use local mode for this endpoint")
    base = _base_url(os.environ)
    if not os.environ.get("LITELLM_BASE_URL"):
        raise ValueError("Cloud mode requires the task's [model] endpoint")
    # A provider belongs to a context and endpoint/key combination. Credential rotation gets a new
    # provider without changing another context's in-flight calls. No raw credential is persisted locally.
    identity = json.dumps([context_id, base, os.environ.get("LITELLM_API_KEY", "")])
    name = "agentenv-" + hashlib.sha256(identity.encode()).hexdigest()[:32]
    async with ClientSession(headers={"Authorization": "Bearer " + env["LETTA_API_KEY"]},
                             timeout=ClientTimeout(total=_CLI_TIMEOUT_S)) as client:
        url = env["LETTA_BASE_URL"].rstrip("/") + "/v1/providers"
        async with client.get(url) as response:
            response.raise_for_status()
            providers = await response.json()
        if not any(provider["name"] == name for provider in providers):
            async with client.post(url, json={"name": name, "provider_type": "openai", "base_url": base,
                                               "api_key": os.environ.get("LITELLM_API_KEY", "")}) as response:
                # A second process may have created the same deterministic provider.
                if response.status == 409:
                    async with client.get(url) as retry:
                        retry.raise_for_status()
                        if not any(item["name"] == name for item in await retry.json()):
                            response.raise_for_status()
                else:
                    response.raise_for_status()
    yield name + "/" + model


async def _command(env: dict[str, str], *args: str) -> bytes:
    code, out, err = await _communicate(*_cli(env), *args, env=env, timeout_s=120)
    if code:
        raise RuntimeError(f"Letta Code {args[0]} failed: {err.decode(errors='replace')[-ERROR_MESSAGE_LIMIT:]}")
    return out


async def _cloud_agent(env, context_id, handle, agent_id=None):
    """Resolve the context on the server and bind both inference and compaction to BYOK."""
    base = env["LETTA_BASE_URL"].rstrip("/") + "/v1/agents"
    name = "agentenv-" + hashlib.sha256(context_id.encode()).hexdigest()[:32]
    async with ClientSession(headers={"Authorization": "Bearer " + env["LETTA_API_KEY"]},
                             timeout=ClientTimeout(total=_CLI_TIMEOUT_S)) as client:
        if not agent_id:
            async with client.get(base, params={"name": name}) as response:
                response.raise_for_status()
                agents = await response.json()
            if agents:
                agent_id = agents[0]["id"]
        config = {"model": handle, "compaction_settings": {"model": handle}}
        if agent_id:
            async with client.patch(base + "/" + agent_id, json=config) as response:
                response.raise_for_status()
        else:
            async with client.post(base, json={
                **config, "name": name, "tags": ["origin:letta-code", "git-memory-enabled"],
                "include_base_tools": False, "include_base_tool_rules": False,
                "tools": [], "initial_message_sequence": [],
            }) as response:
                response.raise_for_status()
                agent_id = (await response.json())["id"]
    return agent_id


def _configure_agent(home: Path, agent_id: str, servers: dict, role: str | None, system_prompt: str | None):
    # Core memory's system/ files are compiled into the existing agent's system prompt on every run.
    memory = home / ".letta/lc-local-backend/memfs" / agent_id / "memory"
    memory.mkdir(parents=True, exist_ok=True)
    instructions = system_prompt or ""
    if servers:
        instructions += ("\nUse the connected MCP tools directly for services and peer agents. "
                         "They are named mcp__<server>__<tool>. These services require no repository setup.\n")
    target = "task_instructions.md" if (memory / "MEMORY.md").exists() else "system/task_instructions.md"
    path = memory / target
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\nname: Task instructions\ndescription: Instructions for the current task\n---\n" +
                    (instructions or "Follow the user's task.\n"))
    # The local backend projects committed core memory, not uncommitted working files.
    for args in (["add", "--", target], ["-c", "user.name=Letta Code", "-c", "user.email=code@letta.com",
                                           "-c", "commit.gpgsign=false", "commit", "--quiet", "--allow-empty",
                                           "-m", "Update task instructions", "--", target]):
        subprocess.run(["git", *args], cwd=memory, check=True, capture_output=True)

    _configure_settings(home, agent_id, servers, role)


def _configure_settings(home, agent_id, servers, role, base_url=None):
    settings_path = home / ".letta/settings.json"
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    agents = settings.setdefault("agents", [])
    matching = [agent for agent in agents if agent.get("agentId") == agent_id]
    if not matching:
        matching = [{"agentId": agent_id, "baseUrl": base_url or "local:" + str(home / ".letta/lc-local-backend")}]
        agents.extend(matching)
    for agent in matching:
        agent["mcpServers"] = [
            {"name": name, "transport": "http", "url": info["url"],
             "headers": _mcp_headers(info.get("headers"), role)} for name, info in servers.items()
        ]
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings))


def _install_mcp_tools(home: Path, agent_id: str, tools: list[dict]) -> None:
    """A context-local mod registers the CLI-discovered schemas without renaming their tools."""
    directory = home / ".letta"
    (directory / "mods").mkdir(parents=True, exist_ok=True)
    (directory / "agentenv-mcp-tools.json").write_text(json.dumps({"agentId": agent_id, "tools": tools}))
    shutil.copyfile(Path(__file__).with_name("mcp_tools.ts"), directory / "mods/agentenv-mcp.ts")


async def run_letta_code(
    prompt: str, context_id: str, session_id: str | None = None, model: str | None = None,
    mcp_servers: dict[str, dict] | None = None, system_prompt: str | None = None,
    output_format: dict | None = None, timeout_s: int = 600, project_id: str | None = None,
    task_id: str | None = None, role: str | None = None, attachment_paths: list[str] | None = None,
    letta_api_key: str | None = None, letta_base_url: str = "https://api.letta.com",
    effort: str | None = None, event_callback: Callable[[dict], None] | None = None, workspace: str = "/app",
) -> LettaCodeResult:
    """A bounded CLI turn, including provider setup; cancellation kills the process tree."""
    try:
        async with asyncio.timeout(timeout_s):
            return await _turn(prompt, context_id, session_id, model, mcp_servers, system_prompt, output_format,
                               project_id, task_id, role, attachment_paths, effort, event_callback, workspace,
                               letta_api_key, letta_base_url)
    except TimeoutError:
        return LettaCodeResult(response="", tool_call_count=0, session_id=session_id,
                               error=f"Letta Code timed out after {timeout_s}s")


async def _turn(prompt, context_id, session_id, model, mcp_servers, system_prompt, output_format,
                project_id, task_id, role, attachment_paths, effort, event_callback, workspace,
                letta_api_key=None, letta_base_url="https://api.letta.com"):
    model = model or DEFAULT_MODEL
    env = _context_env(context_id, letta_api_key, letta_base_url)
    home = Path(env["HOME"])
    (home / "context.json").write_text(json.dumps({"home": str(home)}))
    attribution = _attribution_headers(project_id, task_id)
    async with _provider(env, context_id, model, attribution) as handle:
        agent_file = home / "agent-id.json"
        if agent_file.exists():
            saved = json.loads(agent_file.read_text())
            if (saved.get("backend", "local") == "cloud") != bool(letta_api_key):
                raise ValueError("Context backend differs from the configured backend")
            if letta_api_key and saved.get("base_url") != letta_base_url:
                raise ValueError("Cloud snapshot belongs to another Letta base URL")
            agent_id = saved["id"]
        else:
            agent_id = None
        if letta_api_key:
            agent_id = await _cloud_agent(env, context_id, handle, agent_id)
        elif agent_id is None:
            agent = json.loads(await _command(env, "agents", "create", "--model", handle))
            agent_id = agent["id"]
        agent_file.write_text(json.dumps({"id": agent_id, "backend": "cloud" if letta_api_key else "local",
                                          "base_url": letta_base_url if letta_api_key else None}))
        if letta_api_key:
            _configure_settings(home, agent_id, mcp_servers or {}, role, letta_base_url)
            if system_prompt:
                prompt = system_prompt + "\n\n" + prompt
        else:
            _configure_agent(home, agent_id, mcp_servers or {}, role, system_prompt)
        tools = json.loads(await _command(env, "mcp", "tools", "--full", "--agent", agent_id)) if mcp_servers else []
        _install_mcp_tools(home, agent_id, tools)
        skills = Path(os.environ.get("HOME", "/data")) / ".letta/skills"
        if skills.exists():
            shutil.copytree(skills, home / ".letta/skills", dirs_exist_ok=True)
        # The shell tools inherit these, so nested `letta mcp` calls select the same backend and agent.
        env["LETTA_AGENT_ID"] = agent_id
        if output_format is not None:
            schema = validate_output_format(output_format)
            prompt += "\nReturn only JSON matching this schema: " + json.dumps(schema)
        for path in attachment_paths or []:
            prompt += "\nRead the attached file at " + path
        cmd = [*_cli(env), "-p", prompt, "--output-format", "stream-json", "--yolo",
               "--model", handle, "--toolset", "default"]
        cmd += ["--conversation", session_id] if session_id else ["--agent", agent_id, "--new"]
        if effort:
            target = ["--conversation", session_id] if session_id else ["--agent", agent_id]
            await _command(env, "model", "set", handle, "--reasoning", effort, *target)
        proc = await _started(asyncio.create_subprocess_exec(
            *cmd, cwd=workspace, env=env, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True,
        ))
        _RUNNING.add(proc)
        parser = _EventParser(event_callback)
        if not letta_api_key:
            _record_attribution(project_id, task_id)
        else:
            with _attribution_lock:
                _last_attribution.update(project_id=None, task_id=None, last_seen_at_utc=None)
        try:
            stderr = await _stream_cli(proc, parser.on_line)
        except BaseException:
            _kill_cli(proc)
            await _reap(proc)
            raise
        finally:
            _RUNNING.discard(proc)
        result = parser.result(stderr, proc.returncode)
        if output_format is not None and result.error is None:
            try:
                result.structured_output = json.loads(result.response)
                Draft202012Validator(schema).validate(result.structured_output)
            except (ValueError, ValidationError) as exc:
                result.error = f"Letta Code response does not match output_format: {exc}"
                result.error_code = "invalid_output_format"
        return result


# How much of the CLI's stdout one read takes; a line may span several reads.
_STDOUT_READ_BYTES = 64 * 1024


async def _stream_cli(proc, on_line: Callable[[str], None]) -> bytes:
    """``proc.communicate()``, handing ``on_line`` each stdout line as it arrives.

    Lines are split as ``stdout.decode(errors="replace").splitlines()`` splits
    the whole output, so they are the same lines; a line has no length limit,
    as with ``communicate``. Returns stderr once the CLI has exited."""
    stderr_task = asyncio.create_task(proc.stderr.read())
    try:
        # A line's pieces, joined once when its newline arrives: re-joining them on every read
        # would cost a long line the square of its length.
        pieces: list[bytes] = []
        while chunk := await proc.stdout.read(_STDOUT_READ_BYTES):
            *ends, rest = chunk.split(b"\n")
            for end in ends:
                pieces.append(end)
                for line in b"".join(pieces).decode(errors="replace").splitlines():
                    on_line(line)
                pieces = []
            pieces.append(rest)
        for line in b"".join(pieces).decode(errors="replace").splitlines():
            on_line(line)
        stderr = await stderr_task
        await proc.wait()
        return stderr
    finally:
        stderr_task.cancel()



def event_session_id(event: dict) -> str | None:
    return event.get("conversation_id")


class _EventParser:
    def __init__(self, event_callback=None):
        self.callback = event_callback
        self.trajectory = []
        self.session_id = None
        self.response = ""
        self.tools = set()
        self.completed = False
        self.error = None

    def on_line(self, line: str):
        try:
            event = json.loads(line)
        except ValueError:
            return
        if not isinstance(event, dict):
            return
        self.trajectory.append(event)
        self.session_id = event_session_id(event) or self.session_id
        if event.get("message_type") == "tool_call_message":
            self.tools.add((event.get("tool_call") or {}).get("tool_call_id") or event.get("uuid") or str(len(self.tools)))
        if event.get("type") == "result":
            self.completed = event.get("subtype") == "success"
            self.response = event.get("result") or ""
            if not self.completed:
                self.error = event.get("stop_reason") or self.response or "Letta Code did not complete"
        if self.callback:
            self.callback(event)

    def result(self, stderr: bytes, returncode: int | None) -> LettaCodeResult:
        error = None if self.completed else (self.error or f"Letta Code exited {returncode} without completing")
        if error:
            error = (error + "\n" + stderr.decode(errors="replace")[-ERROR_MESSAGE_LIMIT:])[-ERROR_MESSAGE_LIMIT:]
        return LettaCodeResult(response=self.response, tool_call_count=len(self.tools), session_id=self.session_id,
                               trajectory=self.trajectory, error=error)
