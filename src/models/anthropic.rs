use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::error::{AgentError, Result};
use crate::models::LLM;
use crate::types::{File, GenerationResponse, Message, Role};

/// Anthropic Claude LLM provider
pub struct AnthropicLLM {
    client: Client,
    api_key: String,
    model: String,
    max_tokens: u32,
}

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: ImageSource },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ImageSource {
    #[serde(rename = "type")]
    source_type: String,
    media_type: String,
    data: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<ContentBlock>,
}

impl AnthropicLLM {
    /// Creates a new Anthropic LLM with API key from environment
    pub fn new(model: impl Into<String>) -> Result<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
            AgentError::ConfigError("ANTHROPIC_API_KEY environment variable not set".to_string())
        })?;

        Ok(Self {
            client: Client::new(),
            api_key,
            model: model.into(),
            max_tokens: 4096,
        })
    }

    /// Creates with explicit API key
    pub fn with_api_key(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            model: model.into(),
            max_tokens: 4096,
        }
    }

    /// Sets max tokens for generation
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    fn convert_role(role: &Role) -> String {
        match role {
            Role::User => "user".to_string(),
            Role::Assistant => "assistant".to_string(),
            Role::System => "user".to_string(), // System handled separately
            Role::Tool => "user".to_string(),
        }
    }
}

#[async_trait]
impl LLM for AnthropicLLM {
    async fn generate(
        &self,
        messages: Vec<Message>,
        files: Option<Vec<File>>,
    ) -> Result<GenerationResponse> {
        // Extract system message if present
        let system_prompt = messages
            .iter()
            .find(|m| matches!(m.role, Role::System))
            .map(|m| m.content.clone());

        // Convert remaining messages
        let mut anthropic_messages: Vec<AnthropicMessage> = messages
            .into_iter()
            .filter(|m| !matches!(m.role, Role::System))
            .map(|m| AnthropicMessage {
                role: Self::convert_role(&m.role),
                content: AnthropicContent::Text(m.content),
            })
            .collect();

        // Add files to last user message if provided
        if let Some(files) = files {
            if let Some(last_msg) = anthropic_messages.last_mut() {
                let mut blocks = vec![ContentBlock::Text {
                    text: match &last_msg.content {
                        AnthropicContent::Text(t) => t.clone(),
                        _ => String::new(),
                    },
                }];

                for file in files {
                    if file.mime_type.starts_with("image/") {
                        blocks.push(ContentBlock::Image {
                            source: ImageSource {
                                source_type: "base64".to_string(),
                                media_type: file.mime_type,
                                data: base64::engine::general_purpose::STANDARD.encode(&file.data),
                            },
                        });
                    }
                }

                last_msg.content = AnthropicContent::Blocks(blocks);
            }
        }

        let request = AnthropicRequest {
            model: self.model.clone(),
            messages: anthropic_messages,
            max_tokens: self.max_tokens,
            system: system_prompt,
        };

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| AgentError::ModelError(format!("Anthropic request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(AgentError::ModelError(format!(
                "Anthropic API error {}: {}",
                status, text
            )));
        }

        let anthropic_response: AnthropicResponse = response
            .json()
            .await
            .map_err(|e| AgentError::ModelError(format!("Failed to parse response: {}", e)))?;

        let content = anthropic_response
            .content
            .into_iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        Ok(GenerationResponse {
            content,
            metadata: None,
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires API key
    async fn test_anthropic_generate() {
        let llm = AnthropicLLM::new("claude-3-5-sonnet-20241022").unwrap();
        let messages = vec![Message {
            role: Role::User,
            content: "Say 'Hello' and nothing else.".to_string(),
            metadata: None,
        }];

        let response = llm.generate(messages, None).await.unwrap();
        assert!(response.content.contains("Hello"));
    }
}
