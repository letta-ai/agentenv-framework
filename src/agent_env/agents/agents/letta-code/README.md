# Letta Code A2A agent

The [Letta Code](https://github.com/letta-ai/letta-code) CLI wrapped with `agentenv_protocol.a2a_agent`, shipped
in the `agents` bundle as `@local/agentenv-framework/agents/letta-code`, with alias `letta-code`. This directory
is the whole build context. The Dockerfile pins the Python base by digest and Node by version and checksum;
`package-lock.json` pins `@letta-ai/letta-code` to 0.34.25.

Each turn runs `letta --backend local -p --output-format stream-json --yolo`. The bridge creates an agent with
`letta agents create`; the first turn uses `--agent <id> --new`, and subsequent turns use
`--agent <id> --conversation <id>`. Each context has a separate home under `$HOME/sessions/`. Different contexts
run concurrently; turns in one context and snapshot operations are synchronized.

The bridge connects the local backend with `letta connect openai-compatible`. A per-turn loopback relay forwards
requests to `$LITELLM_BASE_URL/v1`, with `LITELLM_API_KEY`, `[model.headers]` (`LITELLM_EXTRA_HEADERS`), and
`project_id` / `task_id` as `x-litellm-customer-id` and `x-litellm-tags`. The model defaults to
`openai/gpt-4.1-mini`. Endpoint credentials stay at the relay. Registered MCP servers retain their names and headers;
a task role replaces `AgentEnv-Role`. Task instructions explain discovery and the full `mcp__<server>__<tool>`
call form for the CLI's MCP tools, including peer tools.

| Extension | Implementation |
|---|---|
| `agent-config/v1` | SDK set/get; `model`, `effort`, `system_prompt`, `output_format`, `project_id`, `task_id`, `timeout_seconds`. Before every turn, task instructions are committed to the existing agent's core memory, which the local backend compiles into its system prompt. JSON schema answers are validated locally. |
| `mcp-config/v1` | SDK registration and listing; servers passed to the context's agent settings each turn, with exact-name discovery instructions. |
| `skill-config/v1` | Inline and bundle skills installed under `$HOME/.letta/skills/`, copied into each context's home before its turn. |
| `trajectory/v1` | Native stream-json events appended live, format `letta-code-events/1`. |
| `snapshot/v1` | Save archives the complete context home, including conversations, agent records, settings and git-backed memory; load stages it and restores it into the target context, rewriting stored home paths and binding the saved conversation. Optional workspace archives restore `/app`. GNU tar changelog captures task-level file increments, including removed roots, using the OpenCode template's implementation. |
| `peer-agents/v1` | SDK default; its `peer_agents` MCP server is registered alongside task servers. |
| `attribution-probe/v1` | Reports the project and task attribution used by the last CLI turn. |
| `install/v1` | Template installer brings checksum-verified Node and uv, the pinned CLI, and a private Python into a container or a Linux host with glibc. Installs git if missing. |
| `staging/v1` | SDK default. |

The image runs as user `agent` in `/app`, with state under `/data`, and includes git and `pdftoppm`. PDF attachments
are rendered as page images; other attachments are materialized and their paths passed to the CLI. Headless mode
never loads node-pty, so installation skips dependency scripts and rebuilds only the CLI, sharp and protobufjs.
The end of CLI stderr is included in failures. Cancellation and shutdown kill the CLI's process tree.

Run `agent-env run agents --task letta-code-smoke` for the shell smoke task and
`agent-env a2a-agent validate --id letta-code --sandbox local` for the repository's validator.
