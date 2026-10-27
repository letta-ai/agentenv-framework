"""Exercise state restore, system instructions, event parsing and cancellation without model calls."""
import asyncio
import json
import subprocess
from contextlib import asynccontextmanager

import pytest

from tst.unit.agents.letta_code.agent import a2a_server, letta_code_bridge as bridge


@pytest.mark.asyncio
async def test_snapshot_restores_memory_and_conversation_in_another_context(monkeypatch, tmp_path):
    monkeypatch.setattr(bridge, "SESSIONS_ROOT", str(tmp_path / "sessions"))
    source = bridge.context_data_dir("source")
    source.mkdir(parents=True)
    (source / "context.json").write_text(json.dumps({"home": str(source)}))
    (source / "agent-id.json").write_text('{"id":"agent-local-1"}')
    memory = source / ".letta/lc-local-backend/memfs/agent-local-1/memory"
    memory.mkdir(parents=True)
    (memory / "MEMORY.md").write_text("Remember violet otters")
    conversation = source / "conversation.json"
    conversation.write_text(json.dumps({"id": "local-conv-7", "memory": str(memory), "messages": ["secret"]}))
    snapshot = tmp_path / "state.tgz"
    snapshot.write_bytes(await bridge.export_session("source", "local-conv-7"))
    target = bridge.context_data_dir("target")
    target.mkdir()
    (target / "stale").write_text("old target")
    assert await bridge.import_session("target", str(snapshot)) == "local-conv-7"
    assert not (target / "stale").exists()
    assert (target / memory.relative_to(source) / "MEMORY.md").read_text() == "Remember violet otters"
    restored = json.loads((target / "conversation.json").read_text())
    assert restored["messages"] == ["secret"]
    assert restored["memory"] == str(target / memory.relative_to(source))
    assert (source / "conversation.json").exists()


def test_system_prompt_is_committed_and_servers_keep_role_headers(tmp_path):
    memory = tmp_path / ".letta/lc-local-backend/memfs/agent-local-1/memory"
    memory.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(memory)], check=True)
    (memory / "MEMORY.md").write_text("index")
    bridge._configure_agent(tmp_path, "agent-local-1", {
        "mail": {"url": "https://mail.example/mcp", "headers": {"Authorization": "token", "AgentEnv-Role": "old"}},
    }, "reader", "End every answer with ZEBRA")
    committed = subprocess.check_output(["git", "show", "HEAD:task_instructions.md"], cwd=memory).decode()
    assert "End every answer with ZEBRA" in committed
    assert "mcp__<server>__<tool>" in committed
    settings = json.loads((tmp_path / ".letta/settings.json").read_text())
    server = settings["agents"][0]["mcpServers"][0]
    assert server["headers"] == {"Authorization": "token", "AgentEnv-Role": "reader"}
    bridge._configure_agent(tmp_path, "agent-local-1", {}, None, "Use TIGER now")
    committed = subprocess.check_output(["git", "show", "HEAD:task_instructions.md"], cwd=memory).decode()
    assert "TIGER" in committed and "ZEBRA" not in committed


def test_native_events_track_conversation_tools_and_stderr():
    events = []
    parser = bridge._EventParser(events.append)
    for event in [
        {"type": "system", "conversation_id": "local-conv-2"},
        {"type": "message", "message_type": "tool_call_message", "tool_call": {"tool_call_id": "call-1"}},
        {"type": "message", "message_type": "tool_call_message", "tool_call": {"tool_call_id": "call-1"}},
        {"type": "result", "subtype": "success", "result": "42"},
    ]:
        parser.on_line(json.dumps(event))
    result = parser.result(b"", 0)
    assert result.response == "42" and result.session_id == "local-conv-2"
    assert result.tool_call_count == 1 and result.error is None
    assert len(events) == 4
    assert "provider failed" in bridge._EventParser().result(b"provider failed", 1).error


@pytest.mark.asyncio
async def test_cancel_kills_the_turn_process(monkeypatch, tmp_path):
    monkeypatch.setattr(bridge, "SESSIONS_ROOT", str(tmp_path / "sessions"))
    agent = a2a_server.LettaCodeAgent()
    app = a2a_server.build_app(agent)
    assert app.state.agentenv_a2a.card.name == "Letta Code"
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def command(*args, **kwargs):
        return b'{"id":"agent-local-1"}'

    async def stream(proc, callback):
        started.set()
        await asyncio.Event().wait()

    class Proc:
        returncode = None
        def kill(self):
            self.returncode = -9
            stopped.set()
        async def wait(self):
            return self.returncode

    @asynccontextmanager
    async def relay(*args):
        yield "http://relay/v1"

    monkeypatch.setattr(bridge, "model_relay", relay)

    async def spawn(*args, **kwargs):
        return Proc()

    monkeypatch.setattr(bridge, "_command", command)
    monkeypatch.setattr(bridge, "_configure_agent", lambda *args: None)
    monkeypatch.setattr(bridge.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(bridge, "_stream_cli", stream)
    turn = asyncio.create_task(bridge.run_letta_code("wait", "ctx", workspace=str(tmp_path)))
    await asyncio.wait_for(started.wait(), 5)
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn
    assert stopped.is_set() and not bridge._RUNNING
