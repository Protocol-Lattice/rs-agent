// This example demonstrates registering an rs-agent as an in-process UTCP tool.
// Requires the local `rs-utcp` dependency.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use rs_agent::memory::{InMemoryStore, SessionMemory};
use rs_agent::types::{File, GenerationResponse, Message};
use rs_agent::{Agent, AgentError, AgentOptions, Result, LLM};
use rs_utcp::config::UtcpClientConfig;
use rs_utcp::repository::in_memory::InMemoryToolRepository;
use rs_utcp::tag::tag_search::TagSearchStrategy;
use rs_utcp::UtcpClient;
use rs_utcp::UtcpClientInterface;

// Mock LLM that echoes the last message
struct MockLLM;

#[async_trait]
impl LLM for MockLLM {
    async fn generate(
        &self,
        messages: Vec<Message>,
        _files: Option<Vec<File>>,
    ) -> Result<GenerationResponse> {
        let last = messages.last().map(|m| m.content.as_str()).unwrap_or("");
        Ok(GenerationResponse {
            content: format!("UTCP-enabled response to: {}", last),
            metadata: None,
        })
    }

    fn model_name(&self) -> &str {
        "utcp-mock"
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("🔌 UTCP Integration Example\n");

    // Build UTCP client (in-memory repo + tag-based search)
    let repo = Arc::new(InMemoryToolRepository::new());
    let search = Arc::new(TagSearchStrategy::new(repo.clone(), 1.0));
    let utcp_client = UtcpClient::create(UtcpClientConfig::new(), repo, search)
        .await
        .map_err(|e| AgentError::UtcpError(e.to_string()))?;

    // Create agent
    let memory = Arc::new(SessionMemory::new(Box::new(InMemoryStore::new()), 10));
    let agent = Arc::new(
        Agent::new(Arc::new(MockLLM), memory, AgentOptions::default())
            .with_system_prompt("You are a UTCP-enabled agent."),
    );

    // Expose the agent as a UTCP tool
    agent
        .clone()
        .register_as_utcp_provider(&utcp_client, "local.agent", "Local rs-agent")
        .await?;

    println!("📡 Agent registered as UTCP tool 'local.agent'\n");

    // Call via UTCP client
    let mut args = HashMap::new();
    args.insert(
        "instruction".to_string(),
        serde_json::json!("Process this via UTCP"),
    );

    let utcp_response = utcp_client
        .call_tool("local.agent", args)
        .await
        .map_err(|e| AgentError::UtcpError(e.to_string()))?;
    println!(
        "🤖 UTCP Call Result: {}\n",
        utcp_response.as_str().unwrap_or("<non-string>")
    );

    println!("✅ UTCP integration ready!");
    println!("📚 See rs-utcp documentation for full integration details");

    Ok(())
}
