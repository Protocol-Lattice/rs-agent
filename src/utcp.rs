use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use rs_utcp::tools::Tool as UtcpTool;
use rs_utcp::UtcpClientInterface;

use crate::error::{AgentError, Result};
use crate::tools::Tool;
use crate::types::{ToolRequest, ToolResponse, ToolSpec};

/// Adapter that exposes a UTCP tool through the rs-agent `Tool` trait.
pub struct UtcpToolAdapter {
    client: Arc<dyn UtcpClientInterface>,
    tool: UtcpTool,
}

impl UtcpToolAdapter {
    pub fn new(client: Arc<dyn UtcpClientInterface>, tool: UtcpTool) -> Self {
        Self { client, tool }
    }

    fn tool_spec(&self) -> ToolSpec {
        let input_schema = serde_json::to_value(&self.tool.inputs)
            .unwrap_or_else(|_| serde_json::json!({"type": "object"}));

        ToolSpec {
            name: self.tool.name.clone(),
            description: self.tool.description.clone(),
            input_schema,
            examples: None,
        }
    }
}

#[async_trait]
impl Tool for UtcpToolAdapter {
    fn spec(&self) -> ToolSpec {
        self.tool_spec()
    }

    async fn invoke(&self, req: ToolRequest) -> Result<ToolResponse> {
        // Forward invocation through the UTCP client
        let result = self
            .client
            .call_tool(&self.tool.name, req.arguments)
            .await
            .map_err(|e| AgentError::UtcpError(e.to_string()))?;

        // Preserve string outputs as-is; serialize other payloads to JSON text
        let content = match result {
            serde_json::Value::String(s) => s,
            other => serde_json::to_string(&other).unwrap_or_else(|_| format!("{other:?}")),
        };

        Ok(ToolResponse {
            content,
            metadata: Some(HashMap::from([(
                "provider".to_string(),
                "utcp".to_string(),
            )])),
        })
    }
}

/// Registers UTCP tools into the agent's tool catalog.
pub fn register_utcp_tools(
    catalog: &crate::tools::ToolCatalog,
    client: Arc<dyn UtcpClientInterface>,
    tools: Vec<UtcpTool>,
) -> Result<()> {
    for tool in tools {
        let adapter = UtcpToolAdapter::new(client.clone(), tool);
        catalog.register(Box::new(adapter))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Agent;
    use crate::memory::{InMemoryStore, SessionMemory};
    use crate::models::LLM;
    use crate::types::{AgentOptions, File, GenerationResponse, Message};
    use anyhow::anyhow;
    use rs_utcp::providers::base::Provider;
    use rs_utcp::tools::ToolInputOutputSchema;
    use rs_utcp::transports::stream::StreamResult;
    use rs_utcp::transports::CommunicationProtocol;
    use std::sync::Mutex;

    struct MockUtcpClient {
        calls: Mutex<Vec<(String, HashMap<String, serde_json::Value>)>>,
    }

    impl MockUtcpClient {
        fn new() -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl UtcpClientInterface for MockUtcpClient {
        async fn register_tool_provider(
            &self,
            _prov: Arc<dyn Provider>,
        ) -> anyhow::Result<Vec<UtcpTool>> {
            Ok(vec![])
        }

        async fn register_tool_provider_with_tools(
            &self,
            _prov: Arc<dyn Provider>,
            tools: Vec<UtcpTool>,
        ) -> anyhow::Result<Vec<UtcpTool>> {
            Ok(tools)
        }

        async fn deregister_tool_provider(&self, _provider_name: &str) -> anyhow::Result<()> {
            Ok(())
        }

        async fn call_tool(
            &self,
            tool_name: &str,
            args: HashMap<String, serde_json::Value>,
        ) -> anyhow::Result<serde_json::Value> {
            self.calls
                .lock()
                .unwrap()
                .push((tool_name.to_string(), args));
            Ok(serde_json::json!({"ok": true}))
        }

        async fn search_tools(&self, _query: &str, _limit: usize) -> anyhow::Result<Vec<UtcpTool>> {
            Ok(vec![])
        }

        fn get_transports(&self) -> HashMap<String, Arc<dyn CommunicationProtocol>> {
            HashMap::new()
        }

        async fn call_tool_stream(
            &self,
            _tool_name: &str,
            _args: HashMap<String, serde_json::Value>,
        ) -> anyhow::Result<Box<dyn StreamResult>> {
            Err(anyhow!("not implemented"))
        }
    }

    struct MockLLM;

    #[async_trait]
    impl LLM for MockLLM {
        async fn generate(
            &self,
            messages: Vec<Message>,
            _files: Option<Vec<File>>,
        ) -> Result<GenerationResponse> {
            let last = messages.last().unwrap();
            Ok(GenerationResponse {
                content: format!("Echo: {}", last.content),
                metadata: None,
            })
        }

        fn model_name(&self) -> &str {
            "mock"
        }
    }

    #[tokio::test]
    async fn registers_and_invokes_utcp_tool() {
        let client = Arc::new(MockUtcpClient::new());
        let memory = Arc::new(SessionMemory::new(Box::new(InMemoryStore::new()), 4));
        let agent = Agent::new(Arc::new(MockLLM), memory, AgentOptions::default());

        let tool = UtcpTool {
            name: "dummy.echo".to_string(),
            description: "Echo via UTCP".to_string(),
            inputs: ToolInputOutputSchema {
                type_: "object".to_string(),
                properties: Some(HashMap::from([(
                    "text".to_string(),
                    serde_json::json!({"type": "string"}),
                )])),
                required: Some(vec!["text".to_string()]),
                description: None,
                title: None,
                items: None,
                enum_: None,
                minimum: None,
                maximum: None,
                format: None,
            },
            outputs: ToolInputOutputSchema {
                type_: "object".to_string(),
                properties: None,
                required: None,
                description: None,
                title: None,
                items: None,
                enum_: None,
                minimum: None,
                maximum: None,
                format: None,
            },
            tags: vec![],
            average_response_size: None,
            provider: None,
        };

        register_utcp_tools(agent.tools().as_ref(), client.clone(), vec![tool]).unwrap();

        let mut args = HashMap::new();
        args.insert("text".to_string(), serde_json::json!("hello"));

        let result = agent.invoke_tool("s", "dummy.echo", args).await.unwrap();

        assert_eq!(result, r#"{"ok":true}"#);
        let calls = client.calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "dummy.echo");
    }
}
