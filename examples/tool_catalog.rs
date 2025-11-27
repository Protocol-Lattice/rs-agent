use async_trait::async_trait;
use rs_agent::memory::{InMemoryStore, SessionMemory};
use rs_agent::tools::Tool;
use rs_agent::types::{
    File, GenerationResponse, Message, Role, ToolRequest, ToolResponse, ToolSpec,
};
use rs_agent::{Agent, AgentOptions, Result, LLM};
use std::collections::HashMap;
use std::sync::Arc;

// Simple LLM that echoes the latest user message and includes any tool context
struct RoutingLLM;

#[async_trait]
impl LLM for RoutingLLM {
    async fn generate(
        &self,
        messages: Vec<Message>,
        _files: Option<Vec<File>>,
    ) -> Result<GenerationResponse> {
        let latest_user = messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.content.clone())
            .unwrap_or_else(|| "No user input found".to_string());

        let tool_notes: Vec<String> = messages
            .iter()
            .filter(|m| m.role == Role::Tool)
            .map(|m| m.content.clone())
            .collect();

        let mut content = format!("Answering: {latest_user}");
        if !tool_notes.is_empty() {
            content.push_str(&format!(" (tool context: {})", tool_notes.join(" | ")));
        }

        Ok(GenerationResponse {
            content,
            metadata: None,
        })
    }

    fn model_name(&self) -> &str {
        "routing-mock"
    }
}

// Weather lookup tool with a concrete example payload
struct WeatherTool;

#[async_trait]
impl Tool for WeatherTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "weather".to_string(),
            description: "Returns a fake forecast for a city".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City to look up"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["city"]
            }),
            examples: Some(vec![serde_json::json!({
                "city": "Lisbon",
                "unit": "celsius"
            })]),
        }
    }

    async fn invoke(&self, req: ToolRequest) -> Result<ToolResponse> {
        let city = req
            .arguments
            .get("city")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown");
        let unit = req
            .arguments
            .get("unit")
            .and_then(|v| v.as_str())
            .unwrap_or("celsius");

        let (temp, scale) = match unit {
            "fahrenheit" => (72, "F"),
            _ => (22, "C"),
        };

        let report = format!("{city}: partly cloudy, {temp}°{scale}");

        Ok(ToolResponse {
            content: report,
            metadata: Some(HashMap::from([(
                "source".to_string(),
                "mock-weather".to_string(),
            )])),
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("🛠️ Tool Catalog Example\n");

    // Set up agent with routing LLM and session memory
    let memory = Arc::new(SessionMemory::new(Box::new(InMemoryStore::new()), 8));
    let agent = Agent::new(Arc::new(RoutingLLM), memory, AgentOptions::default())
        .with_system_prompt("You decide when tool outputs are relevant to the user.");

    // Register custom tools
    let catalog = agent.tools();
    catalog.register(Box::new(WeatherTool))?;

    println!("Registered tools:");
    for spec in catalog.specs() {
        let example = spec
            .examples
            .as_ref()
            .and_then(|e| e.first())
            .map(|v| v.to_string())
            .unwrap_or_else(|| "n/a".to_string());
        println!(
            "- {} → {} | example input: {}",
            spec.name, spec.description, example
        );
    }
    println!();

    // Invoke tool directly
    let mut args = HashMap::new();
    args.insert("city".to_string(), serde_json::json!("Lisbon"));
    args.insert("unit".to_string(), serde_json::json!("fahrenheit"));

    let tool_response = agent.invoke_tool("tools_session", "weather", args).await?;
    println!("🧰 Weather tool output: {tool_response}\n");

    // Follow up with the LLM; it will see the stored tool output in its context
    let reply = agent
        .generate(
            "tools_session",
            "Summarize the forecast and suggest what to wear.",
        )
        .await?;
    println!("🤖 Agent: {reply}");

    Ok(())
}
