use async_trait::async_trait;
use base64::Engine;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::error::{AgentError, Result};
use crate::models::LLM;
use crate::types::{File, GenerationResponse, Message, Role};

/// Gemini LLM provider
pub struct GeminiLLM {
    client: Client,
    api_key: String,
    model: String,
}

#[derive(Debug, Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Serialize)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum GeminiPart {
    Text { text: String },
    InlineData { inline_data: GeminiBlob },
}

#[derive(Debug, Serialize)]
struct GeminiBlob {
    mime_type: String,
    data: String,
}

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Option<Vec<GeminiCandidate>>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: Option<GeminiContentResponse>,
}

#[derive(Debug, Deserialize)]
struct GeminiContentResponse {
    parts: Option<Vec<GeminiPartResponse>>,
}

#[derive(Debug, Deserialize)]
struct GeminiPartResponse {
    text: Option<String>,
}

impl GeminiLLM {
    /// Creates a new Gemini LLM with API key from environment
    pub fn new(model: impl Into<String>) -> Result<Self> {
        let api_key = std::env::var("GOOGLE_API_KEY")
            .or_else(|_| std::env::var("GEMINI_API_KEY"))
            .map_err(|_| {
                AgentError::ConfigError(
                    "GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set".to_string(),
                )
            })?;

        Ok(Self {
            client: Client::new(),
            api_key,
            model: model.into(),
        })
    }

    /// Creates with explicit API key
    pub fn with_api_key(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            model: model.into(),
        }
    }

    fn convert_role(role: &Role) -> String {
        match role {
            Role::User => "user".to_string(),
            Role::Assistant => "model".to_string(),
            Role::System => "user".to_string(), // Gemini maps system to user or uses specific system instruction
            Role::Tool => "user".to_string(),
        }
    }
}

#[async_trait]
impl LLM for GeminiLLM {
    async fn generate(
        &self,
        messages: Vec<Message>,
        files: Option<Vec<File>>,
    ) -> Result<GenerationResponse> {
        let mut contents: Vec<GeminiContent> = messages
            .iter()
            .map(|m| GeminiContent {
                role: Self::convert_role(&m.role),
                parts: vec![GeminiPart::Text {
                    text: m.content.clone(),
                }],
            })
            .collect();

        // Add file attachments if present to the last message
        if let Some(files) = files {
            if let Some(last_content) = contents.last_mut() {
                for file in files {
                    last_content.parts.push(GeminiPart::InlineData {
                        inline_data: GeminiBlob {
                            mime_type: file.mime_type,
                            data: base64::engine::general_purpose::STANDARD.encode(&file.data),
                        },
                    });
                }
            }
        }

        let request = GeminiRequest {
            contents,
            tools: None,
        };

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            self.model, self.api_key
        );

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| AgentError::ModelError(format!("Gemini API error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(AgentError::ModelError(format!(
                "Gemini API error {}: {}",
                status, text
            )));
        }

        let gemini_response: GeminiResponse = response
            .json()
            .await
            .map_err(|e| AgentError::ModelError(format!("Failed to parse response: {}", e)))?;

        let content = gemini_response
            .candidates
            .as_ref()
            .and_then(|c| c.first())
            .and_then(|c| c.content.as_ref())
            .and_then(|c| c.parts.as_ref())
            .and_then(|p| p.first())
            .and_then(|p| p.text.clone())
            .ok_or_else(|| AgentError::ModelError("No content in response".to_string()))?;

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
    async fn test_gemini_generate() {
        let llm = GeminiLLM::new("gemini-2.0-flash").unwrap();
        let messages = vec![Message {
            role: Role::User,
            content: "Say 'Hello, World!' and nothing else.".to_string(),
            metadata: None,
        }];

        let response = llm.generate(messages, None).await.unwrap();
        assert!(response.content.contains("Hello"));
    }
}
