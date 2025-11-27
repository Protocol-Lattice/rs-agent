use async_openai::{
    types::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs, ChatCompletionRequestUserMessageContent,
        CreateChatCompletionRequestArgs, ImageUrl,
    },
    Client,
};
use async_trait::async_trait;

use crate::error::{AgentError, Result};
use crate::models::LLM;
use crate::types::{File, GenerationResponse, Message, Role};

/// OpenAI LLM provider
pub struct OpenAILLM {
    client: Client<async_openai::config::OpenAIConfig>,
    model: String,
}

impl OpenAILLM {
    /// Creates a new OpenAI LLM with API key from environment
    pub fn new(model: impl Into<String>) -> Result<Self> {
        let _ = std::env::var("OPENAI_API_KEY").map_err(|_| {
            AgentError::ConfigError("OPENAI_API_KEY environment variable not set".to_string())
        })?;

        Ok(Self {
            client: Client::new(),
            model: model.into(),
        })
    }

    /// Creates with explicit API key
    pub fn with_api_key(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let config = async_openai::config::OpenAIConfig::new().with_api_key(api_key);
        Self {
            client: Client::with_config(config),
            model: model.into(),
        }
    }
}

#[async_trait]
impl LLM for OpenAILLM {
    async fn generate(
        &self,
        messages: Vec<Message>,
        files: Option<Vec<File>>,
    ) -> Result<GenerationResponse> {
        let mut chat_messages = Vec::new();

        for msg in messages {
            match msg.role {
                Role::System => {
                    chat_messages.push(
                        ChatCompletionRequestSystemMessageArgs::default()
                            .content(msg.content)
                            .build()
                            .map_err(|e| {
                                AgentError::ModelError(format!(
                                    "Failed to build system message: {}",
                                    e
                                ))
                            })?
                            .into(),
                    );
                }
                Role::User => {
                    chat_messages.push(
                        ChatCompletionRequestUserMessageArgs::default()
                            .content(msg.content)
                            .build()
                            .map_err(|e| {
                                AgentError::ModelError(format!(
                                    "Failed to build user message: {}",
                                    e
                                ))
                            })?
                            .into(),
                    );
                }
                Role::Assistant => {
                    chat_messages.push(
                        ChatCompletionRequestAssistantMessageArgs::default()
                            .content(msg.content)
                            .build()
                            .map_err(|e| {
                                AgentError::ModelError(format!(
                                    "Failed to build assistant message: {}",
                                    e
                                ))
                            })?
                            .into(),
                    );
                }
                Role::Tool => {
                    // Handle tool messages if needed, treating as user for now or skipping
                    // OpenAI has specific tool message types, but for basic chat we might skip or adapt
                    chat_messages.push(
                        ChatCompletionRequestUserMessageArgs::default()
                            .content(format!("Tool output: {}", msg.content))
                            .build()
                            .map_err(|e| {
                                AgentError::ModelError(format!(
                                    "Failed to build tool message: {}",
                                    e
                                ))
                            })?
                            .into(),
                    );
                }
            }
        }

        // Handle files (images) by appending to the last user message
        if let Some(files) = files {
            if let Some(last_msg) = chat_messages.last_mut() {
                if let async_openai::types::ChatCompletionRequestMessage::User(user_msg) = last_msg
                {
                    let mut content_parts = Vec::new();

                    // Add existing text content
                    if let Some(content) = &user_msg.content {
                        match content {
                            ChatCompletionRequestUserMessageContent::Text(text) => {
                                content_parts.push(async_openai::types::ChatCompletionRequestMessageContentPart::Text(
                                    async_openai::types::ChatCompletionRequestMessageContentPartTextArgs::default()
                                        .text(text)
                                        .build()
                                        .unwrap()
                                ));
                            }
                            ChatCompletionRequestUserMessageContent::Array(parts) => {
                                content_parts.extend(parts.clone());
                            }
                        }
                    }

                    // Add images
                    for file in files {
                        if file.mime_type.starts_with("image/") {
                            let base64_image =
                                base64::engine::general_purpose::STANDARD.encode(&file.data);
                            let data_url =
                                format!("data:{};base64,{}", file.mime_type, base64_image);

                            content_parts.push(async_openai::types::ChatCompletionRequestMessageContentPart::ImageUrl(
                                async_openai::types::ChatCompletionRequestMessageContentPartImageArgs::default()
                                    .image_url(
                                        ImageUrl::default()
                                            .url(data_url)
                                            .detail(async_openai::types::ImageDetail::Auto)
                                    )
                                    .build()
                                    .unwrap()
                            ));
                        }
                    }

                    // Update the message content
                    // Note: This is a bit hacky as we're modifying the built message.
                    // Ideally we'd build it with images initially.
                    // For simplicity in this structure, we might need to rebuild or use a different approach if the types don't allow mutation easily.
                    // async-openai types are often builders.

                    // Re-create the user message with new content
                    *user_msg = ChatCompletionRequestUserMessageArgs::default()
                        .content(content_parts)
                        .build()
                        .map_err(|e| {
                            AgentError::ModelError(format!(
                                "Failed to rebuild user message with images: {}",
                                e
                            ))
                        })?;
                }
            }
        }

        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.model)
            .messages(chat_messages)
            .build()
            .map_err(|e| AgentError::ModelError(format!("Failed to build request: {}", e)))?;

        let response = self
            .client
            .chat()
            .create(request)
            .await
            .map_err(|e| AgentError::ModelError(format!("OpenAI API error: {}", e)))?;

        let content = response
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();

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
    async fn test_openai_generate() {
        let llm = OpenAILLM::new("gpt-3.5-turbo").unwrap();
        let messages = vec![Message {
            role: Role::User,
            content: "Say 'Hello' and nothing else.".to_string(),
            metadata: None,
        }];

        let response = llm.generate(messages, None).await.unwrap();
        assert!(response.content.contains("Hello"));
    }
}
