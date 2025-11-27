use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Tool specification describing how an agent presents a tool to the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub examples: Option<Vec<serde_json::Value>>,
}

/// Tool request captures an invocation request
#[derive(Debug, Clone)]
pub struct ToolRequest {
    pub session_id: String,
    pub arguments: HashMap<String, serde_json::Value>,
}

/// Tool response represents the structured response from a tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResponse {
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

/// Message role in a conversation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// Message in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

/// File attachment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct File {
    pub mime_type: String,
    pub data: Vec<u8>,
}

/// Generation response from a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResponse {
    pub content: String,
    pub metadata: Option<HashMap<String, String>>,
}

/// Configuration options for creating an agent
#[derive(Debug, Clone)]
pub struct AgentOptions {
    pub system_prompt: Option<String>,
    pub context_limit: Option<usize>,
}

impl Default for AgentOptions {
    fn default() -> Self {
        Self {
            system_prompt: None,
            context_limit: Some(8192),
        }
    }
}
