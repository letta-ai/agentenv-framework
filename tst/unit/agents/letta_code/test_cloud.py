"""Cloud selection, provider reuse, CLI execution and reference-only snapshots."""
import json
from contextlib import asynccontextmanager

import pytest

from tst.unit.agents.letta_code.agent import a2a_server, letta_code_bridge as bridge


@pytest.mark.parametrize("key", [None, "cloud-secret"])
def test_backend_is_selected_by_agent_config_not_inherited_credentials(monkeypatch, tmp_path, key):
    monkeypatch.setattr(bridge, "SESSIONS_ROOT", str(tmp_path))
    monkeypatch.setenv("LETTA_API_KEY", "unrelated-shell-key")
    monkeypatch.setenv("LETTA_BASE_URL", "https://unrelated.example")
    monkeypatch.setenv("LETTA_CODE_AGENT_ROLE", "subagent")
    monkeypatch.setenv("LETTA_LOCAL_BACKEND_EXPERIMENTAL", "1")
    env = bridge._context_env("ctx", key, "https://cloud.example")
    assert "LETTA_CODE_AGENT_ROLE" not in env
    assert env["LETTA_LOCAL_BACKEND_EXPERIMENTAL"] == ("0" if key else "1")
    assert env["LETTA_BACKEND"] == ("api" if key else "local")
    if key:
        assert env["LETTA_API_KEY"] == key and env["LETTA_BASE_URL"] == "https://cloud.example"
        assert bridge._cli(env) == ["letta"]
    else:
        assert "LETTA_API_KEY" not in env and "LETTA_BASE_URL" not in env
        assert bridge._cli(env) == ["letta", "--backend", "local"]


def test_key_is_write_only_in_agent_config():
    schema = a2a_server.LettaCodeConfig.model_json_schema()
    assert schema["properties"]["letta_api_key"]["writeOnly"]
    assert a2a_server.LettaCodeConfig().letta_base_url == "https://api.letta.com"


@pytest.mark.asyncio
async def test_cloud_provider_registers_gateway_once_and_reuses_it(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "https://gateway.example")
    monkeypatch.setenv("LITELLM_API_KEY", "gateway-key")
    monkeypatch.delenv(bridge.MODEL_HEADERS_ENV, raising=False)
    providers, writes = [], []

    class Response:
        status = 200
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def raise_for_status(self):
            pass
        async def json(self):
            return providers

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["headers"] == {"Authorization": "Bearer cloud-key"}
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def get(self, url):
            assert url == "https://cloud.example/v1/providers"
            return Response()
        def post(self, url, json):
            writes.append(json)
            providers.append({"name": json["name"]})
            return Response()

    monkeypatch.setattr(bridge, "ClientSession", Client)
    env = {"LETTA_API_KEY": "cloud-key", "LETTA_BASE_URL": "https://cloud.example"}
    handles = []
    for _ in range(2):
        async with bridge._provider(env, "ctx", "openai/gpt-4.1-mini", {}) as handle:
            handles.append(handle)
    assert handles[0] == handles[1] == writes[0]["name"] + "/openai/gpt-4.1-mini"
    assert len(writes) == 1
    assert writes[0]["base_url"] == "https://gateway.example/v1"
    assert writes[0]["api_key"] == "gateway-key" and writes[0]["provider_type"] == "openai"
    monkeypatch.setenv(bridge.MODEL_HEADERS_ENV, '{"X-Required": "secret"}')
    with pytest.raises(ValueError, match=r"\[model.headers\]"):
        async with bridge._provider(env, "ctx", "model", {}):
            pytest.fail("Must not discard required endpoint headers")


@pytest.mark.asyncio
@pytest.mark.parametrize("conversation", [None, "conv-existing"])
async def test_cloud_turn_resumes_agent_and_builds_cloud_process(monkeypatch, tmp_path, conversation):
    monkeypatch.setattr(bridge, "SESSIONS_ROOT", str(tmp_path))
    commands, processes = [], []

    @asynccontextmanager
    async def provider(env, context, model, attribution):
        assert env["LETTA_API_KEY"] == "cloud-key"
        assert model == "task-model"
        yield "run-provider/task-model"

    async def command(env, *args):
        commands.append(args)
        assert env["LETTA_BASE_URL"] == "https://cloud.example"
        return b'{"items":[{"id":"agent-existing"}]}' if args[:2] == ("agents", "list") else b'{}'

    class Proc:
        returncode = 0
    async def spawn(*args, **kwargs):
        processes.append((args, kwargs))
        return Proc()
    async def stream(proc, on_line):
        on_line('{"type":"system","conversation_id":"conv-result"}')
        on_line('{"type":"result","subtype":"success","result":"42"}')
        return b''

    async def cloud_agent(env, context, handle, agent_id):
        assert context == "ctx" and handle == "run-provider/task-model"
        return "agent-existing"

    monkeypatch.setattr(bridge, "_cloud_agent", cloud_agent)
    monkeypatch.setattr(bridge, "_provider", provider)
    monkeypatch.setattr(bridge, "_command", command)
    monkeypatch.setattr(bridge.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(bridge, "_stream_cli", stream)
    result = await bridge.run_letta_code("answer", "ctx", model="task-model", session_id=conversation,
                                         letta_api_key="cloud-key", letta_base_url="https://cloud.example",
                                         system_prompt="Use the task rules", effort="low", workspace=str(tmp_path))
    assert result.response == "42" and result.session_id == "conv-result" and result.error is None
    args, kwargs = processes[0]
    assert "--backend" not in args
    assert args[args.index("--model") + 1] == "run-provider/task-model"
    assert "Use the task rules" in args[args.index("-p") + 1]
    assert kwargs["env"]["LETTA_AGENT_ID"] == "agent-existing"
    assert "--conversation" in args if conversation else "--new" in args
    assert not any(command[:2] == ("agents", "create") for command in commands)
    assert any(command[:2] == ("model", "set") for command in commands)
    # A fresh server can rebind IDs; credentials and endpoint headers are not in the archive.
    home = bridge.context_data_dir("ctx")
    (home / "secret-settings").write_text("do not export")
    snapshot = tmp_path / "cloud.tgz"
    snapshot.write_bytes(await bridge.export_session("ctx", result.session_id))
    assert await bridge.import_session("restored", str(snapshot)) == "conv-result"
    restored = bridge.context_data_dir("restored")
    assert json.loads((restored / "agent-id.json").read_text())["id"] == "agent-existing"
    assert not (restored / "secret-settings").exists()
    assert not (restored / ".letta/settings.json").exists()


def test_config_resolves_only_the_selected_agent_secrets(monkeypatch):
    from agent_env.config.runtime import validated_agents_section
    from tst.util.config import config_with_document

    config = config_with_document({"agents": {"config": {
        "solver": {"letta_api_key": "secret:cloud", "letta_base_url": "https://cloud.example"},
        "other": {"letta_api_key": "secret:missing"},
    }}})
    requested = []
    def secret(key):
        requested.append(key)
        return "resolved-key"
    monkeypatch.setattr(config, "_resolve_secret", secret)
    assert config.get_agent_config("solver") == {
        "letta_api_key": "resolved-key", "letta_base_url": "https://cloud.example",
    }
    assert requested == ["cloud"]
    assert config.get_agent_config("absent") == {}
    from agent_env.config.errors import ConfigError
    with pytest.raises(ConfigError, match="table"):
        validated_agents_section({"config": {"solver": "bad"}})


@pytest.mark.asyncio
@pytest.mark.parametrize("bound", [None, "agent-existing"])
async def test_cloud_agent_reuses_context_and_routes_compaction_through_gateway(monkeypatch, bound):
    writes = []
    class Response:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def raise_for_status(self):
            pass
        async def json(self):
            return [{"id": "agent-existing"}]
    class Client:
        def __init__(self, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def get(self, url, params):
            assert params["name"].startswith("agentenv-")
            return Response()
        def patch(self, url, json):
            writes.append((url, json))
            return Response()
        def post(self, *args, **kwargs):
            pytest.fail("Must reuse the existing cloud agent")
    monkeypatch.setattr(bridge, "ClientSession", Client)
    agent = await bridge._cloud_agent({"LETTA_API_KEY": "key", "LETTA_BASE_URL": "https://cloud.example"},
                                       "ctx", "provider/model", bound)
    assert agent == "agent-existing"
    assert writes == [("https://cloud.example/v1/agents/agent-existing", {
        "model": "provider/model", "compaction_settings": {"model": "provider/model"},
    })]


@pytest.mark.asyncio
async def test_cloud_agent_creation_uses_task_model_and_persistent_memory(monkeypatch):
    created = []
    class Response:
        def __init__(self, data):
            self.data = data
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def raise_for_status(self):
            pass
        async def json(self):
            return self.data
    class Client:
        def __init__(self, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def get(self, url, params):
            return Response([])
        def post(self, url, json):
            created.append(json)
            return Response({"id": "new-cloud-agent"})
    monkeypatch.setattr(bridge, "ClientSession", Client)
    assert await bridge._cloud_agent({"LETTA_API_KEY": "key", "LETTA_BASE_URL": "https://cloud.example"},
                                      "ctx", "gateway/model") == "new-cloud-agent"
    assert created[0]["model"] == created[0]["compaction_settings"]["model"] == "gateway/model"
    assert "git-memory-enabled" in created[0]["tags"]
    assert not created[0]["include_base_tools"]  # tools execute in the CLI's task sandbox


@pytest.mark.asyncio
async def test_restored_context_cannot_silently_switch_backends(monkeypatch, tmp_path):
    monkeypatch.setattr(bridge, "SESSIONS_ROOT", str(tmp_path))
    home = bridge.context_data_dir("ctx")
    home.mkdir()
    (home / "agent-id.json").write_text('{"id":"cloud-agent","backend":"cloud","base_url":"https://cloud.example"}')
    @asynccontextmanager
    async def provider(*args):
        yield "provider/model"
    monkeypatch.setattr(bridge, "_provider", provider)
    with pytest.raises(ValueError, match="backend"):
        await bridge.run_letta_code("hello", "ctx")
    with pytest.raises(ValueError, match="base URL"):
        await bridge.run_letta_code("hello", "ctx", letta_api_key="key", letta_base_url="https://other.example")


@pytest.mark.asyncio
async def test_deployment_delivers_secret_backed_cloud_config_via_negotiated_extension(monkeypatch):
    from types import SimpleNamespace

    from agent_env.a2a_agent import A2AAgent
    from agent_env.task_step.context import TaskStepContext
    from agent_env.task_step.task_steps import deploy_agent
    from tst.util.config import config_with_document

    config = config_with_document({"agents": {"config": {"solver": {
        "letta_api_key": "secret:cloud", "letta_base_url": "https://cloud.example",
    }}}})
    monkeypatch.setattr(config, "_resolve_secret", lambda key: "resolved-cloud-key")
    monkeypatch.setattr(deploy_agent, "get_config", lambda: config)
    card = a2a_server.build_app(a2a_server.LettaCodeAgent()).state.agentenv_a2a.card.model_dump(
        mode="json", by_alias=True,
    )
    class Agent:
        metadata = {}
        async def deploy(self, **kwargs):
            return SimpleNamespace(a2a_url="https://agent.example", agent_card=card, sandbox_id="sb",
                                   sandbox_type="fake", instance_id="instance", network_policy=None)
        async def close(self):
            pytest.fail("A successful configured deployment should stay running")
    monkeypatch.setattr(A2AAgent, "get", classmethod(lambda cls, *a, **kw: Agent()))
    delivered = []
    async def post(url, fields):
        delivered.append((url, fields))
    monkeypatch.setattr(deploy_agent.protocol, "post_agent_config", post)
    await deploy_agent.DeployAgentTaskStep(id="deploy", version=None, a2a_agent_id="letta-code",
                                          agent_name="solver").execute(TaskStepContext())
    assert ("https://agent.example/ext/agent-config", {
        "letta_api_key": "resolved-cloud-key", "letta_base_url": "https://cloud.example",
    }) in delivered
