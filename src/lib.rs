//! # rs-agent
//!
//! Lattice AI Agent Framework for Rust
//!
//! `rs-agent` provides clean abstractions for building production AI agents with:
//! - Pluggable LLM providers (Gemini, Ollama, Anthropic)
//! - Tool calling with async support
//! - Memory systems with RAG capabilities
//! - UTCP integration for universal tool calling
//! - Multi-agent coordination
//!
//! ## Quick Start
//!
//! ```no_run
//! use rs_agent::{Agent, AgentOptions};
//! use rs_agent::memory::{InMemoryStore, SessionMemory};
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() {
//!     // Setup will go here
//! }
//! ```

pub mod agent;
pub mod agent_orchestrators;
pub mod agent_tool;
pub mod catalog;
pub mod error;
pub mod helpers;
pub mod memory;
pub mod models;
pub mod query;
pub mod tools;
pub mod types;
pub mod utcp;

// Re-export commonly used types
pub use agent::Agent;
pub use catalog::{StaticSubAgentDirectory, StaticToolCatalog};
pub use error::{AgentError, Result};
pub use memory::{mmr_rerank, InMemoryStore, MemoryRecord, MemoryStore, SessionMemory};
pub use models::LLM;
pub use rs_utcp::plugins::codemode::{CodeModeArgs, CodeModeUtcp, CodemodeOrchestrator};
pub use tools::{Tool, ToolCatalog};
pub use types::{
    AgentOptions, AgentState, File, GenerationResponse, Message, Role, SubAgent,
    SubAgentDirectory, ToolRequest, ToolResponse, ToolSpec,
};

// Re-export memory backends
#[cfg(feature = "postgres")]
pub use memory::PostgresStore;

#[cfg(feature = "qdrant")]
pub use memory::QdrantStore;

#[cfg(feature = "mongodb")]
pub use memory::MongoStore;

// Re-export LLM providers
#[cfg(feature = "gemini")]
pub use models::GeminiLLM;

#[cfg(feature = "ollama")]
pub use models::OllamaLLM;

#[cfg(feature = "anthropic")]
pub use models::AnthropicLLM;

#[cfg(feature = "openai")]
pub use models::OpenAILLM;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_agent_options() {
        let opts = AgentOptions::default();
        assert_eq!(opts.context_limit, Some(8192));
        assert!(opts.system_prompt.is_none());
    }
}
