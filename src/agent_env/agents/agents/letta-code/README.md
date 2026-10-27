# Letta Code A2A agent

The [Letta Code](https://github.com/letta-ai/letta-code) CLI wrapped with `agentenv_protocol.a2a_agent`, shipped
in the `agents` bundle as `@local/agentenv-framework/agents/letta-code`, with alias `letta-code`. This directory
is the whole build context. The Dockerfile pins the Python base by digest and Node by version and checksum;
`package-lock.json` pins `@letta-ai/letta-code` to 0.34.25.

By default each turn runs `letta --backend local -p --output-format stream-json --yolo`. The bridge creates a local agent with
`letta agents create`; the first turn uses `--agent <id> --new`, and subsequent turns use
`--conversation <id>`. Each context has a separate home under `$HOME/sessions/`. Different contexts
run concurrently; turns in one context and snapshot operations are synchronized.

In local mode, the bridge connects the backend with `letta connect openai-compatible`. A per-turn loopback relay forwards
requests to `$LITELLM_BASE_URL/v1`, with `LITELLM_API_KEY`, `[model.headers]` (`LITELLM_EXTRA_HEADERS`), and
`project_id` / `task_id` as `x-litellm-customer-id` and `x-litellm-tags`. The model defaults to
`openai/gpt-4.1-mini`. Endpoint credentials stay at the relay. Registered MCP servers retain their names and headers;
a task role replaces `AgentEnv-Role`. Before each turn, `letta mcp tools --full` discovers the exact schemas. A context-local mod registers them as
direct Letta Code tools through `letta.tools.register`; each tool executes `letta mcp call` with its full
`mcp__<server>__<tool>` name. Peer tools use the same path.

| Extension | Implementation |
|---|---|
| `agent-config/v1` | SDK set/get; `model`, `effort`, `system_prompt`, `output_format`, `project_id`, `task_id`, `timeout_seconds`, write-only `letta_api_key`, and `letta_base_url`. Local task instructions are committed to core memory; cloud task instructions are prepended to each turn. JSON schema answers are validated locally. |
| `mcp-config/v1` | SDK registration and listing; servers passed to the context's agent settings each turn, with their discovered schemas exposed as direct tools through the public mods API. |
| `skill-config/v1` | Inline and bundle skills installed under `$HOME/.letta/skills/`, copied into each context's home before its turn. |
| `trajectory/v1` | Native stream-json events appended live, format `letta-code-events/1`. |
| `snapshot/v1` | Local save/load archives and restores the context home, conversations and git-backed memory, rewriting home paths. Cloud save/load records and rebinds the agent ID, conversation ID and base URL; it references live server memory rather than copying or rolling it back. Both support optional workspace archives. GNU tar changelog captures completed-turn file increments, including removed roots. |
| `peer-agents/v1` | SDK default; its `peer_agents` MCP server is registered alongside task servers. |
| `attribution-probe/v1` | Reports attribution forwarded by the local relay. Cloud provider calls cannot carry per-turn attribution; they do not update this probe. |
| `install/v1` | Template installer brings checksum-verified Node and uv, the pinned CLI, and a private Python into a container or a Linux host with glibc. Installs git if missing. |
| `staging/v1` | SDK default. |

The image runs as user `agent` in `/app`, with state under `/data`, and includes git and `pdftoppm`. PDF attachments
are rendered as page images; other attachments are materialized and their paths passed to the CLI. Headless mode
never loads node-pty, so installation skips dependency scripts and rebuilds only the CLI, sharp and protobufjs.
The end of CLI stderr is included in failures. Cancellation and shutdown kill the CLI's process tree.

Run `agent-env run agents --task letta-code-smoke` for the shell smoke task and
`agent-env a2a-agent validate --id letta-code --sandbox local` for the repository's validator.

## Backends

Local mode is the default when `letta_api_key` is absent or empty. Configure cloud mode in
`.agentenv/config.toml`, using the **deployed agent name** (`default-agent` in the shipped smoke task):

```toml
[agents.config.default-agent]
letta_api_key = "secret:LETTA_API_KEY"
letta_base_url = "https://api.letta.com" # optional, this is the default

[model]
base_url = "https://your-public-gateway.example"
api_key = "secret:MODEL_API_KEY"
default = "openai/gpt-4.1-mini"
```

`[agents.config.<deployed-name>]` supplies negotiated `agent-config/v1` fields at deployment.
`secret:` references resolve through `[stores.secret]`; `env:` references also work. The default
secret store reads the named environment variable, but secrets never belong literally in TOML.
The key is write-only and omitted from agent-config reads. This uses the repository's config file
and secret store instead of introducing a new environment switch. `agent.toml` remains optional;
putting a key in `[default_env_vars]` there would persist credentials with the agent definition.
The bridge sets the CLI's existing authentication variables internally and ignores inherited
`LETTA_*` settings so unrelated CLI sessions cannot select this run's backend or identity.

With a key, the bridge omits `--backend local` and uses the CLI's default cloud backend. It
registers an OpenAI-compatible BYOK provider using the same `/v1/providers` API as the pinned
CLI's `connect` implementation (`provider_type = "openai"`). The provider points directly to
`$LITELLM_BASE_URL/v1` and stores the task endpoint's key on Letta Cloud. Both the agent model and
compaction model use `<provider-name>/<task-model>`, retaining gateway routing. The endpoint must
be reachable from Letta Cloud; a sandbox loopback relay or private gateway cannot serve it.
A deterministic provider name hashes the context, endpoint and key, so sequential runs reuse it;
key rotation creates a new provider. Providers remain on the account for reuse.

**Header limitation:** the pinned provider API supports a base URL and API key, but no arbitrary
model-call headers. Cloud mode therefore rejects nonempty `[model.headers]` instead of silently
dropping required headers. It cannot forward per-turn `x-litellm-customer-id` / `x-litellm-tags`,
so Scale gateway routing still applies but project/task header attribution does not. Use local
mode when those headers are required. The attribution probe does not claim cloud forwarding.

An agent's stable name is `agentenv-` plus the first 32 hex characters of SHA-256 of the A2A
context ID. Cloud mode looks it up before creating it, preserving its server-side memory across
runs that use the same context. A new context starts a new agent; snapshot load can explicitly
bind another context to the saved agent and conversation. Without a saved conversation ID,
a run starts a new conversation on the reused agent. Keep independent concurrent runs on
separate contexts; cross-process creation of the same context is not serialized.

Cloud snapshots contain only IDs and the base URL, never the cloud key or server memory.
Restore requires a configured key with access to the original agent on the same base URL.
It resumes the saved conversation against the agent's **current** memory: no offline restore,
point-in-time rollback, account migration or cloning is provided. Workspace save/load remains
available. Local snapshots retain their full state-copy behavior.
