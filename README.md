# rs-agent

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Rust implementation of the Lattice AI Agent Framework. `rs-agent` gives you a production-ready agent orchestrator with pluggable LLMs, tool calling (including UTCP), retrieval-capable memory, CodeMode execution, and multi-agent coordination.

## Highlights
- **Single agent interface**: `Agent` orchestrates LLM calls, memory, tool invocations, file attachments, and TOON encoding.
- **Pluggable models**: Feature-flagged adapters for Gemini, Ollama, Anthropic, and OpenAI behind the `LLM` trait.
- **Tool system**: Implement the `Tool` trait once, register in the `ToolCatalog`, or bridge external tools via UTCP.
- **Memory options**: `SessionMemory` with recent-context windowing, MMR reranking, and optional Postgres/Qdrant/Mongo stores.
- **CodeMode + UTCP**: Ship `codemode.run_code` as a tool, or let the CodeMode orchestrator route natural language into tool chains.
- **Multi-agent ready**: Compose coordinator/specialist agents, or register an agent as a UTCP provider for agent-as-a-tool workflows.

## Install
- From git:
  ```bash
  cargo add rs-agent --git https://github.com/Protocol-Lattice/rs-agent
  ```
- To slim dependencies, disable defaults and pick features:
  ```bash
  cargo add rs-agent --git https://github.com/Protocol-Lattice/rs-agent \
    --no-default-features --features "ollama"
  ```
- Defaults: `gemini`, `memory`. Enable other providers/backends via feature flags listed below.

## Quickstart
Set `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) for Gemini, or swap in any `LLM` implementation you control.

```rust
use rs_agent::{Agent, AgentOptions, GeminiLLM};
use rs_agent::memory::{InMemoryStore, SessionMemory};
use std::sync::Arc;

#[tokio::main]
async fn main() -> rs_agent::Result<()> {
    tracing_subscriber::fmt::init();

    let model = Arc::new(GeminiLLM::new("gemini-2.0-flash")?);
    let memory = Arc::new(SessionMemory::new(Box::new(InMemoryStore::new()), 8));

    let agent = Agent::new(model, memory, AgentOptions::default())
        .with_system_prompt("You are a concise Rust assistant.");

    let reply = agent.generate("demo-session", "Why use Rust for agents?").await?;
    println!("{reply}");
    Ok(())
}
```

## Add a Tool
Register custom tools and they become part of the agent's context and invocation flow.

```rust
use rs_agent::{Tool, ToolRequest, ToolResponse, ToolSpec, AgentError};
use serde_json::json;
use std::collections::HashMap;
use async_trait::async_trait;

struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "echo".into(),
            description: "Echoes the provided input".into(),
            input_schema: json!({
                "type": "object",
                "properties": { "input": { "type": "string" } },
                "required": ["input"]
            }),
            examples: None,
        }
    }

    async fn invoke(&self, req: ToolRequest) -> rs_agent::Result<ToolResponse> {
        let input = req
            .arguments
            .get("input")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AgentError::ToolError("missing input".into()))?;

        Ok(ToolResponse {
            content: input.to_string(),
            metadata: None,
        })
    }
}

// After constructing an `agent` (see Quickstart), register and call the tool
let catalog = agent.tools();
catalog.register(Box::new(EchoTool))?;

let mut args = HashMap::new();
args.insert("input".to_string(), json!("hi"));

let response = agent.invoke_tool("session", "echo", args).await?;
```

## UTCP and CodeMode
- **UTCP bridge**: Register UTCP providers and expose their tools through the `ToolCatalog`. Your agent can also self-register as a UTCP provider for agent-as-a-tool scenarios (see `examples/utcp_integration.rs`).
- **CodeMode**: Exposes `codemode.run_code` and an optional Codemode orchestrator that turns natural language into tool chains or executable snippets. Integration patterns live in `src/agent/codemode.rs` and the agent tests.

## Memory and Context
- `SessionMemory` keeps per-session short-term context with token-aware trimming.
- MMR reranking (`mmr_rerank`) improves retrieval diversity when using embeddings.
- Backends: in-memory by default; opt into Postgres (pgvector), Qdrant, or MongoDB via features.
- Attach files to a generation call (`generate_with_files`) and encode results compactly with `generate_toon`.

## Examples
Run the included examples to see common patterns:
- Quickstart: `cargo run --example quickstart`
- Tool catalog + custom tools: `cargo run --example tool_catalog`
- Memory + checkpoint/restore + files: `cargo run --example memory_checkpoint`
- Multi-agent coordination: `cargo run --example multi_agent`
- UTCP integration + agent-as-tool: `cargo run --example utcp_integration`

## Feature Flags
| Feature | Description | Default |
|---------|-------------|---------|
| `gemini` | Google Gemini LLM via `google-generative-ai-rs` | Yes (default) |
| `ollama` | Local Ollama models via `ollama-rs` | No |
| `anthropic` | Anthropic Claude via `anthropic-sdk` | No |
| `openai` | OpenAI-compatible models via `async-openai` | No |
| `memory` | Embeddings via `fastembed`; enables memory utilities | Yes (default) |
| `postgres` | Postgres store with pgvector | No |
| `qdrant` | Qdrant vector store | No |
| `mongodb` | MongoDB-backed memory store | No |
| `all-providers` | Enable all LLM providers | No |
| `all-memory` | Enable all memory backends | No |

## Environment
| Variable | Purpose |
|----------|---------|
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Required for `GeminiLLM` |
| `ANTHROPIC_API_KEY` | Required for `AnthropicLLM` |
| `OPENAI_API_KEY` | Required for `OpenAILLM` |
| `OLLAMA_HOST` (optional) | Override Ollama host if not localhost |
| Database connection strings | Supply to `PostgresStore::new`, `QdrantStore::new`, or `MongoStore::new` when those features are enabled |

## Status and Roadmap
- Already in place: Agent orchestrator, LLM adapters (Gemini/Ollama/Anthropic/OpenAI), tool catalog, UTCP bridge + agent-as-tool, CodeMode integration, memory backends (in-memory/Postgres/Qdrant/Mongo), checkpoint/restore, TOON encoding, examples and unit tests.
- Next focus: streaming responses, richer retrieval evaluation, tighter UTCP tool discovery/search ergonomics, and more end-to-end tutorials.

## Contributing
Issues and PRs are welcome! Please format (`cargo fmt`), lint (`cargo clippy`), and add tests where it makes sense.

## License
Apache 2.0. See `LICENSE`.
