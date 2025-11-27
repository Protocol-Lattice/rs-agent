# rs-agent

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**rs-agent** is a Rust implementation of the Lattice AI Agent Framework, providing clean abstractions for building production AI agents with LLMs, tool calling, retrieval-augmented memory, and multi-agent coordination.

## Why rs-agent?

Building production AI agents requires more than just LLM calls. You need:

- **Pluggable LLM providers** that swap without rewriting logic
- **Tool calling** that works across different model APIs
- **Memory systems** that remember context across conversations
- **Multi-agent coordination** for complex workflows
- **Testing infrastructure** that doesn't hit external APIs

rs-agent provides all of this with idiomatic Rust patterns and async support.

## Features

- 🧩 **Modular Architecture** – Compose agents from reusable components
- 🤖 **Multi-Agent Support** – Coordinate specialist agents
- 🔧 **Rich Tooling** – Implement the `Tool` trait once, use everywhere
- 🧠 **Smart Memory** – RAG-powered memory with vector search
- 🔌 **Model Agnostic** – Adapters for Gemini, Ollama, Anthropic, or bring your own
- 📡 **UTCP Ready** – First-class Universal Tool Calling Protocol support
- ⚡ **High Performance** – Built with Rust for speed and safety

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rs-agent = { path = "../rs-agent" }
tokio = { version = "1.41", features = ["full"] }
```

### Basic Usage

```rust
use rs_agent::{Agent, AgentOptions};
use rs_agent::memory::{InMemoryStore, SessionMemory};
use std::sync::Arc;

#[tokio::main]
async fn main() -> rs_agent::Result<()> {
    // Create memory store
    let store = Box::new(InMemoryStore::new());
    let memory = Arc::new(SessionMemory::new(store, 10));

    // Create model (implement your LLM trait)
    let model = Arc::new(YourLLMImplementation::new());

    // Create agent
    let agent = Agent::new(model, memory, AgentOptions::default())
        .with_system_prompt("You are a helpful assistant");

    // Generate response
    let response = agent
        .generate("session_123", "What is Rust?")
        .await?;

    println!("{}", response);
    Ok(())
}
```

## Project Structure

```
rs-agent/
├── src/
│   ├── agent/       # Main agent orchestration
│   ├── memory/      # Memory engine and vector stores
│   ├── models/      # LLM provider adapters
│   ├── tools/       # Tool system and catalog
│   ├── types.rs     # Core type definitions
│   └── error.rs     # Error types
├── examples/        # Usage examples
└── tests/          # Integration tests
```

## Core Concepts

### Agent

The `Agent` struct is the main orchestrator that handles:
- LLM interactions
- Memory management
- Tool invocation
- Context building

### Memory System

rs-agent includes a sophisticated memory system:

```rust
use rs_agent::memory::{InMemoryStore, SessionMemory};
use std::sync::Arc;

let store = Box::new(InMemoryStore::new());
let memory = Arc::new(SessionMemory::new(store, 10));
```

Features:
- **Session-based** – Isolated conversations
- **Context windowing** – Automatic trimming
- **Vector search** – Semantic memory retrieval
- **Multiple backends** – In-memory, PostgreSQL, Qdrant

### Tool System

Create custom tools by implementing the `Tool` trait:

```rust
use rs_agent::{Tool, ToolSpec, ToolRequest, ToolResponse, Result};
use async_trait::async_trait;

struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "echo".to_string(),
            description: "Echoes the input".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Text to echo"
                    }
                },
                "required": ["input"]
            }),
            examples: None,
        }
    }

    async fn invoke(&self, req: ToolRequest) -> Result<ToolResponse> {
        let input = req.arguments.get("input")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        Ok(ToolResponse {
            content: input.to_string(),
            metadata: None,
        })
    }
}
```

### UTCP Integration

rs-agent integrates with the [Universal Tool Calling Protocol](https://github.com/universal-tool-calling-protocol) via `rs-utcp`, enabling cross-platform agent orchestration.

```rust
use rs_utcp::{
    config::UtcpClientConfig,
    providers::text::TextProvider,
    repository::in_memory::InMemoryToolRepository,
    tag::tag_search::TagSearchStrategy,
    UtcpClient,
};
use std::collections::HashMap;
use std::sync::Arc;

let repo = Arc::new(InMemoryToolRepository::new());
let search = Arc::new(TagSearchStrategy::new(repo.clone(), 1.0));
let utcp = Arc::new(UtcpClient::create(UtcpClientConfig::new(), repo, search).await?);

// Load tools from a UTCP provider and expose them to the agent
let tools = agent
    .register_utcp_provider(
        utcp.clone(),
        Arc::new(TextProvider::new("example".into(), Some("examples/utcp_tools".into()), None)),
    )
    .await?;

// Invoke UTCP tool like any other registered tool
let mut args = HashMap::new();
args.insert("text".into(), serde_json::json!("hi"));
let result = agent.invoke_tool("session", "example.echo", args).await?;
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Gemini API credentials | For Gemini models |
| `ANTHROPIC_API_KEY` | Anthropic API credentials | For Anthropic models |

## Development

### Running Tests

```bash
cargo test
```

### Running Examples

```bash
# Quickstart example
cargo run --example quickstart

# Tool catalog and custom tools
cargo run --example tool_catalog

# Memory + checkpointing
cargo run --example memory_checkpoint

# Multi-agent example
cargo run --example multi_agent

# UTCP integration
cargo run --example utcp_integration
```

## Features

- `gemini` - Google Gemini LLM support (default)
- `ollama` - Ollama local LLM support
- `anthropic` - Anthropic Claude support
- `memory` - Memory and embedding support (default)
- `postgres` - PostgreSQL backend for memory
- `qdrant` - Qdrant vector database support
- `all-providers` - All LLM providers
- `all-memory` - All memory backends

## Roadmap

- [ ] Gemini LLM implementation
- [ ] Ollama LLM implementation
- [ ] Anthropic LLM implementation
- [ ] PostgreSQL memory backend
- [ ] Qdrant memory backend
- [ ] Streaming support
- [ ] Tool orchestrator (LLM-driven tool selection)
- [ ] Code mode integration
- [ ] Sub-agent support
- [ ] Checkpoint/restore optimization

## Comparison with go-agent

rs-agent is a Rust port of [go-agent](https://github.com/Protocol-Lattice/go-agent), maintaining feature parity while leveraging Rust's:

- **Memory safety** without garbage collection
- **Zero-cost abstractions** for performance
- **Async/await** for efficient concurrency
- **Type system** for compile-time guarantees

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by [go-agent](https://github.com/Protocol-Lattice/go-agent)
- Built on [rs-utcp](https://github.com/Protocol-Lattice/rs-utcp)

---

**Star us on GitHub** if you find rs-agent useful! ⭐
