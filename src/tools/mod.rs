use async_trait::async_trait;
use std::collections::HashMap;

use crate::error::Result;
use crate::types::{ToolRequest, ToolResponse, ToolSpec};

/// Tool trait for defining custom tools
#[async_trait]
pub trait Tool: Send + Sync {
    /// Returns the tool specification
    fn spec(&self) -> ToolSpec;

    /// Invokes the tool with the given request
    async fn invoke(&self, req: ToolRequest) -> Result<ToolResponse>;
}

/// Tool catalog manages registered tools
#[derive(Default)]
pub struct ToolCatalog {
    tools: parking_lot::RwLock<HashMap<String, Box<dyn Tool>>>,
}

impl ToolCatalog {
    /// Creates a new empty tool catalog
    pub fn new() -> Self {
        Self {
            tools: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    /// Registers a tool in the catalog
    pub fn register(&self, tool: Box<dyn Tool>) -> Result<()> {
        let spec = tool.spec();
        let mut tools = self.tools.write();
        tools.insert(spec.name.clone(), tool);
        Ok(())
    }

    /// Looks up a tool by name
    pub fn lookup(&self, name: &str) -> Option<ToolSpec> {
        let tools = self.tools.read();
        tools.get(name).map(|tool| tool.spec())
    }

    /// Returns all tool specifications
    pub fn specs(&self) -> Vec<ToolSpec> {
        let tools = self.tools.read();
        tools.values().map(|tool| tool.spec()).collect()
    }

    /// Invokes a tool by name
    pub async fn invoke(&self, name: &str, req: ToolRequest) -> Result<ToolResponse> {
        let tool = {
            let tools = self.tools.read();
            tools.get(name).map(|t| t.spec().name.clone())
        };

        if tool.is_none() {
            return Err(crate::error::AgentError::ToolNotFound(name.to_string()));
        }

        let tools = self.tools.read();
        let tool = tools.get(name).unwrap();
        tool.invoke(req).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: "echo".to_string(),
                description: "Echoes the input".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "Text to echo"
                        }
                    },
                    "required": ["input"]
                }),
                examples: None,
            }
        }

        async fn invoke(&self, req: ToolRequest) -> Result<ToolResponse> {
            let input = req
                .arguments
                .get("input")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            Ok(ToolResponse {
                content: input.to_string(),
                metadata: None,
            })
        }
    }

    #[tokio::test]
    async fn test_tool_catalog() {
        let catalog = ToolCatalog::new();
        catalog.register(Box::new(EchoTool)).unwrap();

        let spec = catalog.lookup("echo");
        assert!(spec.is_some());

        let mut args = HashMap::new();
        args.insert("input".to_string(), serde_json::json!("hello"));

        let response = catalog
            .invoke(
                "echo",
                ToolRequest {
                    session_id: "test".to_string(),
                    arguments: args,
                },
            )
            .await
            .unwrap();

        assert_eq!(response.content, "hello");
    }
}
