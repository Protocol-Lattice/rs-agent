use async_trait::async_trait;
use rs_agent::memory::{InMemoryStore, SessionMemory};
use rs_agent::types::{File, GenerationResponse, Message, Role};
use rs_agent::{Agent, AgentOptions, Result, LLM};
use std::sync::Arc;

// LLM that includes prior user turns and file count in its response
struct ContextAwareLLM;

#[async_trait]
impl LLM for ContextAwareLLM {
    async fn generate(
        &self,
        messages: Vec<Message>,
        files: Option<Vec<File>>,
    ) -> Result<GenerationResponse> {
        let user_turns: Vec<String> = messages
            .iter()
            .filter(|m| m.role == Role::User)
            .map(|m| m.content.clone())
            .collect();

        let latest = user_turns
            .last()
            .cloned()
            .unwrap_or_else(|| "No user input found".to_string());

        let history = if user_turns.len() > 1 {
            let earlier: Vec<String> = user_turns
                .iter()
                .take(user_turns.len() - 1)
                .cloned()
                .collect();
            format!("Earlier context: {}", earlier.join(" | "))
        } else {
            "No earlier context recorded.".to_string()
        };

        let file_note = files
            .as_ref()
            .map(|f| format!("with {} file(s) attached", f.len()))
            .unwrap_or_else(|| "with no files attached".to_string());

        Ok(GenerationResponse {
            content: format!("{history} Latest request {file_note}: {latest}"),
            metadata: None,
        })
    }

    fn model_name(&self) -> &str {
        "context-aware-mock"
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("🧠 Memory + Checkpoint Example\n");

    // Start a session and capture a checkpoint
    let memory = Arc::new(SessionMemory::new(Box::new(InMemoryStore::new()), 8));
    let agent = Agent::new(Arc::new(ContextAwareLLM), memory, AgentOptions::default())
        .with_system_prompt("Keep track of previous user turns.");

    let session_id = "checkpoint_session";

    let first = agent
        .generate(session_id, "Remember the codename is Aurora.")
        .await?;
    println!("Turn 1 → {first}");

    let launch_notes = File {
        mime_type: "text/plain".to_string(),
        data: b"Launch checklist: fuel, telemetry, go/no-go.".to_vec(),
    };

    let second = agent
        .generate_with_files(session_id, "Here are the launch notes.", vec![launch_notes])
        .await?;
    println!("Turn 2 → {second}");

    let checkpoint = agent.checkpoint(session_id).await?;
    println!("Captured checkpoint ({} bytes)\n", checkpoint.len());

    // Simulate a restart with a fresh agent + memory
    let restored_memory = Arc::new(SessionMemory::new(Box::new(InMemoryStore::new()), 8));
    let restored_agent = Agent::new(
        Arc::new(ContextAwareLLM),
        restored_memory,
        AgentOptions::default(),
    )
    .with_system_prompt("Keep track of previous user turns.");

    restored_agent.restore(session_id, &checkpoint).await?;
    println!("Restored prior context into a new agent instance.\n");

    let toon = restored_agent
        .generate_toon(
            session_id,
            "What codename did we pick and what did I share last?",
        )
        .await?;

    let decoded: GenerationResponse =
        toon_format::decode_default(&toon).expect("valid TOON payload");

    println!("TOON payload:\n{toon}\n");
    println!("Decoded response:\n{}", decoded.content);

    Ok(())
}
