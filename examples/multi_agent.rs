use async_trait::async_trait;
use rs_agent::memory::{InMemoryStore, SessionMemory};
use rs_agent::tools::{Tool, ToolCatalog};
use rs_agent::types::{File, GenerationResponse, Message, ToolRequest, ToolResponse, ToolSpec};
use rs_agent::{Agent, AgentOptions, Result, LLM};
use std::collections::HashMap;
use std::sync::Arc;

// Calculator tool
struct CalculatorTool;

#[async_trait]
impl Tool for CalculatorTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "calculator".to_string(),
            description: "Performs basic arithmetic operations".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The arithmetic operation to perform"
                    },
                    "a": {
                        "type": "number",
                        "description": "First operand"
                    },
                    "b": {
                        "type": "number",
                        "description": "Second operand"
                    }
                },
                "required": ["operation", "a", "b"]
            }),
            examples: None,
        }
    }

    async fn invoke(&self, req: ToolRequest) -> Result<ToolResponse> {
        let operation = req
            .arguments
            .get("operation")
            .and_then(|v| v.as_str())
            .ok_or_else(|| rs_agent::AgentError::ToolError("Missing operation".to_string()))?;

        let a = req
            .arguments
            .get("a")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| rs_agent::AgentError::ToolError("Missing operand a".to_string()))?;

        let b = req
            .arguments
            .get("b")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| rs_agent::AgentError::ToolError("Missing operand b".to_string()))?;

        let result = match operation {
            "add" => a + b,
            "subtract" => a - b,
            "multiply" => a * b,
            "divide" => {
                if b == 0.0 {
                    return Err(rs_agent::AgentError::ToolError(
                        "Division by zero".to_string(),
                    ));
                }
                a / b
            }
            _ => {
                return Err(rs_agent::AgentError::ToolError(format!(
                    "Unknown operation: {}",
                    operation
                )))
            }
        };

        Ok(ToolResponse {
            content: result.to_string(),
            metadata: None,
        })
    }
}

// Mock coordinator LLM
struct CoordinatorLLM;

#[async_trait]
impl LLM for CoordinatorLLM {
    async fn generate(
        &self,
        messages: Vec<Message>,
        _files: Option<Vec<File>>,
    ) -> Result<GenerationResponse> {
        let last = messages.last().map(|m| m.content.as_str()).unwrap_or("");
        let response = format!("I'm coordinating specialist agents. Last message: {}", last);
        Ok(GenerationResponse {
            content: response,
            metadata: None,
        })
    }

    fn model_name(&self) -> &str {
        "coordinator"
    }
}

// Mock specialist LLM
struct SpecialistLLM {
    specialty: String,
}

#[async_trait]
impl LLM for SpecialistLLM {
    async fn generate(
        &self,
        messages: Vec<Message>,
        _files: Option<Vec<File>>,
    ) -> Result<GenerationResponse> {
        let last = messages.last().map(|m| m.content.as_str()).unwrap_or("");
        let response = format!(
            "As a {} specialist, I can help with: {}",
            self.specialty, last
        );
        Ok(GenerationResponse {
            content: response,
            metadata: None,
        })
    }

    fn model_name(&self) -> &str {
        &self.specialty
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("🤖 Multi-Agent Coordination Example\n");

    // Create coordinator agent
    let coordinator_memory = Arc::new(SessionMemory::new(Box::new(InMemoryStore::new()), 10));
    let coordinator_agent = Agent::new(
        Arc::new(CoordinatorLLM),
        coordinator_memory,
        AgentOptions::default(),
    )
    .with_system_prompt("You coordinate multiple specialist agents.");

    // Create specialist agents
    let researcher_memory = Arc::new(SessionMemory::new(Box::new(InMemoryStore::new()), 10));
    let researcher = Agent::new(
        Arc::new(SpecialistLLM {
            specialty: "researcher".to_string(),
        }),
        researcher_memory,
        AgentOptions::default(),
    )
    .with_system_prompt("You are a research specialist.");

    let coder_memory = Arc::new(SessionMemory::new(Box::new(InMemoryStore::new()), 10));
    let coder = Agent::new(
        Arc::new(SpecialistLLM {
            specialty: "coder".to_string(),
        }),
        coder_memory,
        AgentOptions::default(),
    )
    .with_system_prompt("You are a coding specialist.");

    // Create tool catalog with calculator
    let catalog = Arc::new(ToolCatalog::new());
    catalog.register(Box::new(CalculatorTool)).unwrap();

    let math_agent = Agent::new(
        Arc::new(SpecialistLLM {
            specialty: "mathematician".to_string(),
        }),
        Arc::new(SessionMemory::new(Box::new(InMemoryStore::new()), 10)),
        AgentOptions::default(),
    )
    .with_tools(catalog);

    // Demonstrate coordination
    println!("📋 Coordinator Agent:");
    let coord_response = coordinator_agent
        .generate("coord_session", "Organize a research task")
        .await?;
    println!("   {}\n", coord_response);

    println!("🔬 Researcher Agent:");
    let research_response = researcher
        .generate("research_session", "Analyze AI trends")
        .await?;
    println!("   {}\n", research_response);

    println!("💻 Coder Agent:");
    let code_response = coder
        .generate("coder_session", "Write a Rust function")
        .await?;
    println!("   {}\n", code_response);

    println!("🧮 Math Agent with Calculator Tool:");
    let mut args = HashMap::new();
    args.insert("operation".to_string(), serde_json::json!("multiply"));
    args.insert("a".to_string(), serde_json::json!(42));
    args.insert("b".to_string(), serde_json::json!(2));

    let calc_result = math_agent
        .invoke_tool("math_session", "calculator", args)
        .await?;
    println!("   Calculator result: {}\n", calc_result);

    println!("✅ Multi-agent coordination complete!");
    println!("🎯 Each agent has its own memory and specialty");
    println!("🔧 Agents can use tools for specific tasks");

    Ok(())
}
