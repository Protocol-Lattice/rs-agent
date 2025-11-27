use thiserror::Error;

/// Error types for the agent framework
#[derive(Error, Debug)]
pub enum AgentError {
    #[error("Model error: {0}")]
    ModelError(String),

    #[error("Memory error: {0}")]
    MemoryError(String),

    #[error("Tool error: {0}")]
    ToolError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("UTCP error: {0}")]
    UtcpError(String),

    #[error("Agent not found: {0}")]
    AgentNotFound(String),

    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("Other error: {0}")]
    Other(String),

    #[error("TOON format error: {0}")]
    ToonFormatError(String),
}

pub type Result<T> = std::result::Result<T, AgentError>;
