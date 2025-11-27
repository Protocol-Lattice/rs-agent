use async_trait::async_trait;
use ollama_rs::generation::chat::request::ChatMessageRequest;
use ollama_rs::generation::chat::ChatMessage;
use ollama_rs::Ollama;

use crate::error::{AgentError, Result};
use crate::models::LLM;
use crate::types::{File, GenerationResponse, Message, Role};

/// Ollama LLM provider using ollama-rs SDK
pub struct OllamaLLM {
    client: Ollama,
    model: String,
}

impl OllamaLLM {
    /// Creates a new Ollama LLM with default localhost connection
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            client: Ollama::default(),
            model: model.into(),
        }
    }

    /// Creates with custom host and port
    pub fn with_host(host: impl Into<String>, port: u16, model: impl Into<String>) -> Self {
        Self {
            client: Ollama::new(host.into(), port),
            model: model.into(),
        }
    }

    fn convert_role(role: &Role) -> String {
        match role {
            Role::System => "system".to_string(),
            Role::User => "user".to_string(),
            Role::Assistant => "assistant".to_string(),
            Role::Tool => "user".to_string(),
        }
    }

    fn convert_message(&self, msg: &Message) -> ChatMessage {
        ChatMessage {
            role: Self::convert_role(&msg.role),
            content: msg.content.clone(),
            images: None,
        }
    }
}

#[async_trait]
impl LLM for OllamaLLM {
    async fn generate(
        &self,
        messages: Vec<Message>,
        files: Option<Vec<File>>,
    ) -> Result<GenerationResponse> {
        let mut chat_messages: Vec<ChatMessage> =
            messages.iter().map(|m| self.convert_message(m)).collect();

        // Add images to the last user message if provided
        if let Some(files) = files {
            let images: Vec<String> = files
                .into_iter()
                .filter(|f| f.mime_type.starts_with("image/"))
                .map(|f| base64::engine::general_purpose::STANDARD.encode(&f.data))
                .collect();

            if !images.is_empty() {
                if let Some(last_msg) = chat_messages.last_mut() {
                    last_msg.images = Some(images);
                }
            }
        }

        let request = ChatMessageRequest::new(self.model.clone(), chat_messages);

        let response = self
            .client
            .send_chat_messages(request)
            .await
            .map_err(|e| AgentError::ModelError(format!("Ollama error: {}", e)))?;

        Ok(GenerationResponse {
            content: response.message.content,
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
    #[ignore] // Requires Ollama running locally
    async fn test_ollama_generate() {
        let llm = OllamaLLM::new("llama2");
        let messages = vec![Message {
            role: Role::User,
            content: "Say 'test' and nothing else.".to_string(),
            metadata: None,
        }];

        let response = llm.generate(messages, None).await;
        assert!(response.is_ok());
    }
}
