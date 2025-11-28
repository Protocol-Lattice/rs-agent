//! Core Agent orchestrator
//!
//! This module provides the main Agent struct that coordinates LLM calls, memory,
//! tool invocations, and UTCP integration. Matches the structure from go-agent's agent.go.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::anyhow;
use chrono::Utc;
use futures::FutureExt;
use rs_utcp::plugins::codemode::{CodeModeUtcp, CodemodeOrchestrator};
use rs_utcp::providers::base::Provider as UtcpProvider;
use rs_utcp::providers::cli::CliProvider;
use rs_utcp::tools::Tool as UtcpTool;
use rs_utcp::tools::ToolInputOutputSchema;
use rs_utcp::UtcpClientInterface;
use serde_json::{json, Value};
use toon_format::encode_default;
use uuid::Uuid;

use crate::agent_orchestrators::{build_orchestrator, format_codemode_value, CodeModeTool};
use crate::agent_tool::{ensure_agent_cli_transport, InProcessTool};
use crate::error::{AgentError, Result};
use crate::memory::{MemoryRecord, SessionMemory};
use crate::models::LLM;
use crate::tools::ToolCatalog;
use crate::types::{AgentOptions, AgentState, File, GenerationResponse, Message, Role, ToolRequest};

const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful AI assistant. Provide concise, accurate answers and explain when you use tools.";

/// Main Agent orchestrator
///
/// The Agent coordinates model calls, memory, tools, and sub-agents. It matches
/// the structure from go-agent's Agent struct.
pub struct Agent {
    model: Arc<dyn LLM>,
    memory: Arc<SessionMemory>,
    system_prompt: String,
    context_limit: usize,
    tool_catalog: Arc<ToolCatalog>,
    codemode: Option<Arc<CodeModeUtcp>>,
    codemode_orchestrator: Option<Arc<CodemodeOrchestrator>>,
}

impl Agent {
    /// Creates a new Agent with the given configuration
    pub fn new(model: Arc<dyn LLM>, memory: Arc<SessionMemory>, options: AgentOptions) -> Self {
        Self {
            model,
            memory,
            system_prompt: options
                .system_prompt
                .unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.to_string()),
            context_limit: options.context_limit.unwrap_or(8192),
            tool_catalog: Arc::new(ToolCatalog::new()),
            codemode: None,
            codemode_orchestrator: None,
        }
    }

    /// Sets the system prompt
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Sets the tool catalog
    pub fn with_tools(mut self, catalog: Arc<ToolCatalog>) -> Self {
        self.tool_catalog = catalog;
        self
    }

    /// Enables CodeMode execution as a first-class tool (`codemode.run_code`).
    pub fn with_codemode(mut self, engine: Arc<CodeModeUtcp>) -> Self {
        self.set_codemode(engine);
        self
    }

    /// Enables CodeMode plus the Codemode orchestrator for automatic tool routing.
    /// If `orchestrator_model` is None, the primary agent model is reused.
    pub fn with_codemode_orchestrator(
        mut self,
        engine: Arc<CodeModeUtcp>,
        orchestrator_model: Option<Arc<dyn LLM>>,
    ) -> Self {
        self.set_codemode(engine.clone());

        let llm = orchestrator_model.unwrap_or_else(|| Arc::clone(&self.model));
        let orchestrator = build_orchestrator(engine, llm);
        self.codemode_orchestrator = Some(Arc::new(orchestrator));
        self
    }

    /// Registers a UTCP provider and loads its tools into the agent's catalog.
    pub async fn register_utcp_provider(
        &self,
        client: Arc<dyn UtcpClientInterface>,
        provider: Arc<dyn UtcpProvider>,
    ) -> Result<Vec<UtcpTool>> {
        let tools = client
            .register_tool_provider(provider)
            .await
            .map_err(|e| AgentError::UtcpError(e.to_string()))?;

        crate::utcp::register_utcp_tools(self.tool_catalog.as_ref(), client, tools.clone())?;
        Ok(tools)
    }

    /// Registers a UTCP provider using a predefined set of tools and adds them to the catalog.
    pub async fn register_utcp_provider_with_tools(
        &self,
        client: Arc<dyn UtcpClientInterface>,
        provider: Arc<dyn UtcpProvider>,
        tools: Vec<UtcpTool>,
    ) -> Result<Vec<UtcpTool>> {
        let registered_tools = client
            .register_tool_provider_with_tools(provider, tools)
            .await
            .map_err(|e| AgentError::UtcpError(e.to_string()))?;

        crate::utcp::register_utcp_tools(
            self.tool_catalog.as_ref(),
            client,
            registered_tools.clone(),
        )?;

        Ok(registered_tools)
    }

    /// Registers UTCP tools into the agent's catalog without re-registering the provider.
    pub fn register_utcp_tools(
        &self,
        client: Arc<dyn UtcpClientInterface>,
        tools: Vec<UtcpTool>,
    ) -> Result<()> {
        crate::utcp::register_utcp_tools(self.tool_catalog.as_ref(), client, tools)
    }

    /// Returns a UTCP tool specification representing this agent as an in-process tool.
    pub fn as_utcp_tool(
        &self,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> UtcpTool {
        let name = name.into();
        let description = description.into();
        let provider_name = name
            .split('.')
            .next()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or("agent")
            .to_string();

        let inputs = ToolInputOutputSchema {
            type_: "object".to_string(),
            properties: Some(HashMap::from([
                (
                    "instruction".to_string(),
                    json!({
                        "type": "string",
                        "description": "The instruction or query for the agent."
                    }),
                ),
                (
                    "session_id".to_string(),
                    json!({
                        "type": "string",
                        "description": "Optional session id; defaults to the provider-derived session."
                    }),
                ),
            ])),
            required: Some(vec!["instruction".to_string()]),
            description: Some("Call the agent with an instruction".to_string()),
            title: Some("AgentInvocation".to_string()),
            items: None,
            enum_: None,
            minimum: None,
            maximum: None,
            format: None,
        };

        let outputs = ToolInputOutputSchema {
            type_: "object".to_string(),
            properties: Some(HashMap::from([
                ("response".to_string(), json!({ "type": "string" })),
                ("session_id".to_string(), json!({ "type": "string" })),
            ])),
            required: None,
            description: Some("Agent response payload".to_string()),
            title: Some("AgentResponse".to_string()),
            items: None,
            enum_: None,
            minimum: None,
            maximum: None,
            format: None,
        };

        UtcpTool {
            name,
            description,
            inputs,
            outputs,
            tags: vec![
                "agent".to_string(),
                "rs-agent".to_string(),
                "inproc".to_string(),
            ],
            average_response_size: None,
            provider: Some(json!({
                "name": provider_name,
                "provider_type": "cli",
            })),
        }
    }

    /// Registers this agent as a UTCP provider using an in-process CLI shim.
    pub async fn register_as_utcp_provider(
        self: Arc<Self>,
        utcp_client: &dyn UtcpClientInterface,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Result<()> {
        let name = name.into();
        let description = description.into();

        let provider_name = name
            .split('.')
            .next()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or("agent")
            .to_string();

        let tool_spec = self.as_utcp_tool(&name, &description);
        let default_session = format!("{}.session", provider_name);
        let agent = Arc::clone(&self);
        let handler = Arc::new(move |args: HashMap<String, Value>| {
            let agent = Arc::clone(&agent);
            let default_session = default_session.clone();
            async move {
                let instruction = args
                    .get("instruction")
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
                    .filter(|s| !s.trim().is_empty())
                    .ok_or_else(|| anyhow!("missing or invalid 'instruction'"))?;

                let session_id = args
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
                    .filter(|s| !s.trim().is_empty())
                    .unwrap_or_else(|| default_session.clone());

                let content = agent
                    .generate(session_id, instruction)
                    .await
                    .map_err(|e| anyhow!(e.to_string()))?;

                Ok(Value::String(content))
            }
            .boxed()
        });

        let inproc_tool = InProcessTool {
            spec: tool_spec.clone(),
            handler,
        };

        let transport = ensure_agent_cli_transport();
        transport.register(&provider_name, inproc_tool);

        let provider = CliProvider::new(
            provider_name.clone(),
            format!("rs-agent-{}", provider_name),
            None,
        );

        utcp_client
            .register_tool_provider_with_tools(Arc::new(provider), vec![tool_spec])
            .await
            .map_err(|e| AgentError::UtcpError(e.to_string()))?;

        Ok(())
    }

    /// Generates a response for the given user input
    pub async fn generate(
        &self,
        session_id: impl Into<String>,
        user_input: impl Into<String>,
    ) -> Result<String> {
        let response = self
            .generate_internal(session_id.into(), user_input.into(), None)
            .await?;

        Ok(response.content)
    }

    /// Generates a response encoded as TOON for token-efficient downstream parsing
    pub async fn generate_toon(
        &self,
        session_id: impl Into<String>,
        user_input: impl Into<String>,
    ) -> Result<String> {
        let response = self
            .generate_internal(session_id.into(), user_input.into(), None)
            .await?;

        encode_default(&response).map_err(|e| AgentError::ToonFormatError(e.to_string()))
    }

    /// Generates a response with file attachments
    pub async fn generate_with_files(
        &self,
        session_id: impl Into<String>,
        user_input: impl Into<String>,
        files: Vec<File>,
    ) -> Result<String> {
        let response = self
            .generate_internal(session_id.into(), user_input.into(), Some(files))
            .await?;

        Ok(response.content)
    }

    /// Invokes a tool by name
    pub async fn invoke_tool(
        &self,
        session_id: impl Into<String>,
        tool_name: &str,
        arguments: HashMap<String, serde_json::Value>,
    ) -> Result<String> {
        let session_id = session_id.into();

        let request = ToolRequest {
            session_id: session_id.clone(),
            arguments,
        };

        let response = self.tool_catalog.invoke(tool_name, request).await?;

        // Store tool invocation in memory
        self.store_memory(
            &session_id,
            "tool",
            &format!("Called {}: {}", tool_name, response.content),
            response.metadata,
        )
        .await?;

        Ok(response.content)
    }

    /// Builds the prompt with system message and context
    async fn build_prompt(&self, session_id: &str, user_input: &str) -> Result<Vec<Message>> {
        let mut messages = Vec::new();

        // Add system prompt if set
        if !self.system_prompt.is_empty() {
            messages.push(Message {
                role: Role::System,
                content: self.system_prompt.clone(),
                metadata: None,
            });
        }

        // Retrieve recent conversation history
        let recent_memories = self.memory.retrieve_recent(session_id).await?;

        // Add context from memory (limited by context_limit)
        let mut token_count = 0;
        for record in recent_memories.iter().rev() {
            // Simple token estimation (4 chars ≈ 1 token)
            let estimated_tokens = record.content.len() / 4;
            if token_count + estimated_tokens > self.context_limit {
                break;
            }

            messages.push(Message {
                role: match record.role.as_str() {
                    "user" => Role::User,
                    "assistant" => Role::Assistant,
                    "tool" => Role::Tool,
                    _ => Role::User,
                },
                content: record.content.clone(),
                metadata: record.metadata.clone(),
            });

            token_count += estimated_tokens;
        }

        // Add current user input
        messages.push(Message {
            role: Role::User,
            content: user_input.to_string(),
            metadata: None,
        });

        Ok(messages)
    }

    async fn generate_internal(
        &self,
        session_id: String,
        user_input: String,
        files: Option<Vec<File>>,
    ) -> Result<GenerationResponse> {
        // Store user message in memory
        self.store_memory(&session_id, "user", &user_input, None)
            .await?;

        // Try CodeMode orchestration before invoking the primary model
        let has_files = files.as_ref().map(|f| !f.is_empty()).unwrap_or(false);
        if !has_files {
            if let Some((content, metadata)) = self
                .try_codemode_orchestration(&session_id, &user_input)
                .await?
            {
                self.store_memory(&session_id, "assistant", &content, metadata.clone())
                    .await?;

                return Ok(GenerationResponse { content, metadata });
            }
        }

        // Build prompt with context
        let messages = self.build_prompt(&session_id, &user_input).await?;

        // Generate response
        let response = self.model.generate(messages, files).await?;

        // Store assistant response in memory
        self.store_memory(&session_id, "assistant", &response.content, None)
            .await?;

        Ok(response)
    }

    fn set_codemode(&mut self, engine: Arc<CodeModeUtcp>) {
        self.codemode = Some(engine.clone());
        // Expose codemode.run_code as a tool; ignore duplicate registrations
        let _ = self
            .tool_catalog
            .register(Box::new(CodeModeTool::new(engine)));
    }

    async fn try_codemode_orchestration(
        &self,
        _session_id: &str,
        user_input: &str,
    ) -> Result<Option<(String, Option<HashMap<String, String>>)>> {
        let orchestrator = match self.codemode_orchestrator.as_ref() {
            Some(o) => o,
            None => return Ok(None),
        };

        let value = orchestrator
            .call_prompt(user_input)
            .await
            .map_err(|e| AgentError::Other(e.to_string()))?;

        if let Some(v) = value {
            let content = format_codemode_value(&v);
            let metadata = Some(HashMap::from([(
                "source".to_string(),
                "codemode_orchestrator".to_string(),
            )]));
            return Ok(Some((content, metadata)));
        }

        Ok(None)
    }

    /// Stores a memory record
    async fn store_memory(
        &self,
        session_id: &str,
        role: &str,
        content: &str,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<()> {
        let record = MemoryRecord {
            id: Uuid::new_v4(),
            session_id: session_id.to_string(),
            role: role.to_string(),
            content: content.to_string(),
            importance: 0.5, // Default importance
            timestamp: Utc::now(),
            metadata,
            embedding: None,
        };

        self.memory.store(record).await
    }

    /// Flushes memory to persistent store
    pub async fn flush(&self, _session_id: &str) -> Result<()> {
        self.memory.flush().await
    }

    /// Returns the tool catalog
    pub fn tools(&self) -> Arc<ToolCatalog> {
        Arc::clone(&self.tool_catalog)
    }

    /// Checkpoints the agent state for persistence
    pub async fn checkpoint(&self, session_id: &str) -> Result<Vec<u8>> {
        let recent = self.memory.retrieve_recent(session_id).await?;

        let state = AgentState {
            system_prompt: self.system_prompt.clone(),
            short_term: recent,
            joined_spaces: None,
            timestamp: Utc::now(),
        };

        serde_json::to_vec(&state).map_err(|e| AgentError::SerializationError(e))
    }

    /// Restores agent state from checkpoint
    pub async fn restore(&self, _session_id: &str, data: &[u8]) -> Result<()> {
        let state: AgentState =
            serde_json::from_slice(data).map_err(|e| AgentError::SerializationError(e))?;

        // Restore memories
        for record in state.short_term {
            self.memory.store(record).await?;
        }

        Ok(())
    }
}
