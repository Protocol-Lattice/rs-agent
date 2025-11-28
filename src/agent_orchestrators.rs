//! CodeMode orchestration and tool integration
//!
//! This module handles the integration of CodeMode with the agent system,
//! matching the structure from go-agent's agent_orchestrators.go.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::anyhow;
use async_trait::async_trait;
use rs_utcp::plugins::codemode::{
    CodeModeArgs, CodeModeResult, CodeModeUtcp, CodemodeOrchestrator, LlmModel,
};
use serde_json::Value;

use crate::error::AgentError;
use crate::models::LLM;
use crate::tools::Tool;
use crate::types::{Message, Role, ToolRequest, ToolResponse, ToolSpec};

/// Adapter that exposes the UTCP CodeMode runtime as a tool in the agent catalog.
///
/// This allows agents to execute code snippets via the `codemode.run_code` tool.
pub struct CodeModeTool {
    engine: Arc<CodeModeUtcp>,
}

impl CodeModeTool {
    pub fn new(engine: Arc<CodeModeUtcp>) -> Self {
        Self { engine }
    }

    fn spec_from_engine(&self) -> ToolSpec {
        let schema = self.engine.tool();
        let input_schema = serde_json::to_value(&schema.inputs)
            .unwrap_or_else(|_| serde_json::json!({"type": "object"}));

        ToolSpec {
            name: schema.name,
            description: schema.description,
            input_schema,
            examples: None,
        }
    }
}

#[async_trait]
impl Tool for CodeModeTool {
    fn spec(&self) -> ToolSpec {
        self.spec_from_engine()
    }

    async fn invoke(&self, req: ToolRequest) -> crate::Result<ToolResponse> {
        let code = req
            .arguments
            .get("code")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AgentError::ToolError("codemode.run_code requires `code`".into()))?;

        let timeout = req.arguments.get("timeout").and_then(|v| v.as_u64());

        let result = self
            .engine
            .execute(CodeModeArgs {
                code: code.to_string(),
                timeout,
            })
            .await
            .map_err(|e| AgentError::ToolError(e.to_string()))?;

        let content = serialize_result(&result);
        Ok(ToolResponse {
            content,
            metadata: Some(HashMap::from([(
                "provider".to_string(),
                "codemode".to_string(),
            )])),
        })
    }
}

/// Bridge that lets the CodeMode orchestrator reuse an `rs-agent` LLM.
///
/// This adapter allows the CodeMode orchestrator to call into any LLM provider
/// that implements the rs-agent LLM trait.
pub struct CodemodeLlmAdapter {
    llm: Arc<dyn LLM>,
}

impl CodemodeLlmAdapter {
    pub fn new(llm: Arc<dyn LLM>) -> Self {
        Self { llm }
    }
}

#[async_trait]
impl LlmModel for CodemodeLlmAdapter {
    async fn complete(&self, prompt: &str) -> anyhow::Result<Value> {
        let messages = vec![Message {
            role: Role::User,
            content: prompt.to_string(),
            metadata: None,
        }];

        let result = self
            .llm
            .generate(messages, None)
            .await
            .map_err(|e| anyhow!(e.to_string()))?;

        let cleaned = strip_code_fence(&result.content);
        Ok(Value::String(cleaned))
    }
}

/// Builds a CodeMode orchestrator with the given engine and LLM.
///
/// The orchestrator can automatically route natural language queries to tool chains
/// or executable code snippets.
pub fn build_orchestrator(engine: Arc<CodeModeUtcp>, llm: Arc<dyn LLM>) -> CodemodeOrchestrator {
    let adapter = CodemodeLlmAdapter::new(llm);
    CodemodeOrchestrator::new(engine, Arc::new(adapter))
}

/// Convenience function to format orchestrator output for agent responses.
pub fn format_codemode_value(value: &Value) -> String {
    if let Some(s) = value.as_str() {
        return s.to_string();
    }

    serde_json::to_string(value).unwrap_or_else(|_| format!("{value:?}"))
}

fn serialize_result(result: &CodeModeResult) -> String {
    serde_json::to_string(result).unwrap_or_else(|_| format!("{result:?}"))
}

fn strip_code_fence(s: &str) -> String {
    let trimmed = s.trim();
    if !trimmed.starts_with("```") {
        return trimmed.to_string();
    }

    // Remove opening ```lang if present
    let mut inner = trimmed.trim_start_matches("```");
    if let Some(pos) = inner.find('\n') {
        inner = &inner[pos + 1..];
    }

    // Remove closing ```
    if let Some(end) = inner.rfind("```") {
        inner = &inner[..end];
    }

    inner.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_code_fence_removes_markdown_fences() {
        assert_eq!(strip_code_fence("```rust\nlet x = 5;\n```"), "let x = 5;");
        assert_eq!(strip_code_fence("```\ncode\n```"), "code");
        assert_eq!(strip_code_fence("plain text"), "plain text");
    }

    #[test]
    fn format_codemode_value_handles_strings_and_json() {
        assert_eq!(format_codemode_value(&Value::String("test".into())), "test");
        assert_eq!(
            format_codemode_value(&Value::Number(42.into())),
            "42"
        );
    }
}
