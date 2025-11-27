use async_trait::async_trait;
use rs_agent::memory::{InMemoryStore, SessionMemory};
use rs_agent::types::{File, GenerationResponse, Message};
use rs_agent::{Agent, AgentOptions, Result, LLM};
use std::sync::Arc;

// Mock LLM for demonstration
struct MockLLM {
    model_name: String,
}

impl MockLLM {
    fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
        }
    }
}

#[async_trait]
impl LLM for MockLLM {
    async fn generate(
        &self,
        messages: Vec<Message>,
        _files: Option<Vec<File>>,
    ) -> Result<GenerationResponse> {
        // Simple mock response based on last message
        let last_message = messages.last().map(|m| m.content.as_str()).unwrap_or("");

        let response = if last_message.to_lowercase().contains("rust") {
            "Rust is a systems programming language focused on safety, speed, and concurrency. \
             It provides memory safety without garbage collection and enables fearless concurrency."
        } else if last_message.to_lowercase().contains("agent") {
            "An AI agent is a software entity that can perceive its environment, make decisions, \
             and take actions to achieve goals. In rs-agent, we provide tools for building production-ready agents."
        } else {
            "I'm a helpful assistant powered by rs-agent. How can I help you today?"
        };

        Ok(GenerationResponse {
            content: response.to_string(),
            metadata: None,
        })
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("🚀 rs-agent Quickstart Example\n");

    // 1. Create memory store
    let store = Box::new(InMemoryStore::new());
    let memory = Arc::new(SessionMemory::new(store, 10));

    // 2. Create model
    let model = Arc::new(MockLLM::new("mock-llm"));

    // 3. Create agent
    let agent = Agent::new(model, memory, AgentOptions::default())
        .with_system_prompt("You are a helpful AI assistant built with rs-agent.");

    // 4. Have a conversation
    let session_id = "quickstart_session";

    println!("💬 User: What is Rust?");
    let response1 = agent.generate(session_id, "What is Rust?").await?;
    println!("🤖 Agent: {}\n", response1);

    println!("💬 User: Tell me about AI agents");
    let response2 = agent
        .generate(session_id, "Tell me about AI agents")
        .await?;
    println!("🤖 Agent: {}\n", response2);

    println!("💬 User: How are you?");
    let response3 = agent.generate(session_id, "How are you?").await?;
    println!("🤖 Agent: {}\n", response3);

    // 5. Flush memory
    agent.flush(session_id).await?;

    println!("✅ Quickstart complete!");
    println!("📝 The agent remembered context across multiple turns.");
    println!("🔍 Try implementing a real LLM provider to see it in action!");

    Ok(())
}
