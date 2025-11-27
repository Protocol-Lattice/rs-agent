use async_trait::async_trait;

use crate::error::Result;
use crate::types::{File, GenerationResponse, Message};

/// LLM model interface
#[async_trait]
pub trait LLM: Send + Sync {
    /// Generates a response from the model
    async fn generate(
        &self,
        messages: Vec<Message>,
        files: Option<Vec<File>>,
    ) -> Result<GenerationResponse>;

    /// Returns the model name
    fn model_name(&self) -> &str;
}

// LLM provider implementations
#[cfg(feature = "gemini")]
pub mod gemini;

#[cfg(feature = "ollama")]
pub mod ollama;

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "openai")]
pub mod openai;

// Re-export providers
#[cfg(feature = "gemini")]
pub use gemini::GeminiLLM;

#[cfg(feature = "ollama")]
pub use ollama::OllamaLLM;

#[cfg(feature = "anthropic")]
pub use anthropic::AnthropicLLM;

#[cfg(feature = "openai")]
pub use openai::OpenAILLM;
