//! Tool and SubAgent catalog implementations
//!
//! This module provides the default in-memory catalog implementations for both
//! tools and sub-agents, matching the structure from go-agent's catalog.go.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::error::{AgentError, Result};
use crate::tools::Tool;
use crate::types::{SubAgent, SubAgentDirectory, ToolSpec};

/// StaticToolCatalog is the default in-memory implementation of a tool registry.
/// It maintains tools in registration order and provides thread-safe lookup.
pub struct StaticToolCatalog {
    tools: RwLock<HashMap<String, Arc<dyn Tool>>>,
    specs: RwLock<HashMap<String, ToolSpec>>,
    order: RwLock<Vec<String>>,
}

impl StaticToolCatalog {
    /// Creates a new empty catalog
    pub fn new() -> Self {
        Self {
            tools: RwLock::new(HashMap::new()),
            specs: RwLock::new(HashMap::new()),
            order: RwLock::new(Vec::new()),
        }
    }

    /// Register a tool in the catalog using a lower-cased key.
    /// Duplicate names return an error.
    pub fn register(&self, tool: Arc<dyn Tool>) -> Result<()> {
        let spec = tool.spec();
        let key = spec.name.to_lowercase().trim().to_string();

        if key.is_empty() {
            return Err(AgentError::ToolError("tool name is empty".into()));
        }

        let mut tools = self.tools.write().unwrap();
        let mut specs = self.specs.write().unwrap();
        let mut order = self.order.write().unwrap();

        if tools.contains_key(&key) {
            return Err(AgentError::ToolError(format!(
                "tool {} already registered",
                spec.name
            )));
        }

        tools.insert(key.clone(), tool);
        specs.insert(key.clone(), spec);
        order.push(key);

        Ok(())
    }

    /// Lookup a tool and its specification by name
    pub fn lookup(&self, name: &str) -> Option<(Arc<dyn Tool>, ToolSpec)> {
        let key = name.to_lowercase().trim().to_string();

        let tools = self.tools.read().unwrap();
        let specs = self.specs.read().unwrap();

        if let Some(tool) = tools.get(&key) {
            if let Some(spec) = specs.get(&key) {
                return Some((Arc::clone(tool), spec.clone()));
            }
        }

        None
    }

    /// Returns a snapshot of all tool specifications in registration order
    pub fn specs(&self) -> Vec<ToolSpec> {
        let order = self.order.read().unwrap();
        let specs = self.specs.read().unwrap();

        order
            .iter()
            .filter_map(|key| specs.get(key).cloned())
            .collect()
    }

    /// Returns all registered tools in order
    pub fn tools(&self) -> Vec<Arc<dyn Tool>> {
        let order = self.order.read().unwrap();
        let tools = self.tools.read().unwrap();

        order
            .iter()
            .filter_map(|key| tools.get(key).map(Arc::clone))
            .collect()
    }
}

impl Default for StaticToolCatalog {
    fn default() -> Self {
        Self::new()
    }
}

/// StaticSubAgentDirectory is the default SubAgentDirectory implementation.
/// It maintains sub-agents in registration order and provides thread-safe lookup.
pub struct StaticSubAgentDirectory {
    subagents: RwLock<HashMap<String, Arc<dyn SubAgent>>>,
    order: RwLock<Vec<String>>,
}

impl StaticSubAgentDirectory {
    /// Creates a new empty directory
    pub fn new() -> Self {
        Self {
            subagents: RwLock::new(HashMap::new()),
            order: RwLock::new(Vec::new()),
        }
    }
}

impl Default for StaticSubAgentDirectory {
    fn default() -> Self {
        Self::new()
    }
}

impl SubAgentDirectory for StaticSubAgentDirectory {
    /// Register a sub-agent. Duplicate names return an error.
    fn register(&self, subagent: Arc<dyn SubAgent>) -> Result<()> {
        let name = subagent.name();
        let key = name.to_lowercase().trim().to_string();

        if key.is_empty() {
            return Err(AgentError::Other("sub-agent name is empty".into()));
        }

        let mut subagents = self.subagents.write().unwrap();
        let mut order = self.order.write().unwrap();

        if subagents.contains_key(&key) {
            return Err(AgentError::Other(format!(
                "sub-agent {} already registered",
                name
            )));
        }

        subagents.insert(key.clone(), subagent);
        order.push(key);

        Ok(())
    }

    /// Lookup a sub-agent by name
    fn lookup(&self, name: &str) -> Option<Arc<dyn SubAgent>> {
        let key = name.to_lowercase().trim().to_string();
        let subagents = self.subagents.read().unwrap();
        subagents.get(&key).map(Arc::clone)
    }

    /// Returns all registered sub-agents in registration order
    fn all(&self) -> Vec<Arc<dyn SubAgent>> {
        let order = self.order.read().unwrap();
        let subagents = self.subagents.read().unwrap();

        order
            .iter()
            .filter_map(|key| subagents.get(key).map(Arc::clone))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use crate::types::{ToolRequest, ToolResponse};

    struct TestTool {
        name: String,
    }

    #[async_trait]
    impl Tool for TestTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: self.name.clone(),
                description: "Test tool".into(),
                input_schema: serde_json::json!({}),
                examples: None,
            }
        }

        async fn invoke(&self, _req: ToolRequest) -> Result<ToolResponse> {
            Ok(ToolResponse {
                content: "test".into(),
                metadata: None,
            })
        }
    }

    #[test]
    fn catalog_registers_and_lookups_tools() {
        let catalog = StaticToolCatalog::new();
        let tool = Arc::new(TestTool {
            name: "test.tool".into(),
        });

        catalog.register(tool).unwrap();
        assert!(catalog.lookup("test.tool").is_some());
        assert!(catalog.lookup("unknown").is_none());
    }

    #[test]
    fn catalog_prevents_duplicate_registration() {
        let catalog = StaticToolCatalog::new();
        let tool1 = Arc::new(TestTool {
            name: "test.tool".into(),
        });
        let tool2 = Arc::new(TestTool {
            name: "test.tool".into(),
        });

        catalog.register(tool1).unwrap();
        assert!(catalog.register(tool2).is_err());
    }

    struct TestSubAgent {
        name: String,
    }

    #[async_trait]
    impl SubAgent for TestSubAgent {
        fn name(&self) -> String {
            self.name.clone()
        }

        fn description(&self) -> String {
            "Test sub-agent".into()
        }

        async fn run(&self, _input: String) -> Result<String> {
            Ok("test output".into())
        }
    }

    #[test]
    fn directory_registers_and_lookups_subagents() {
        let dir = StaticSubAgentDirectory::new();
        let subagent = Arc::new(TestSubAgent {
            name: "test.agent".into(),
        });

        dir.register(subagent).unwrap();
        assert!(dir.lookup("test.agent").is_some());
        assert!(dir.lookup("unknown").is_none());
    }

    #[test]
    fn directory_prevents_duplicate_registration() {
        let dir = StaticSubAgentDirectory::new();
        let sa1 = Arc::new(TestSubAgent {
            name: "test.agent".into(),
        });
        let sa2 = Arc::new(TestSubAgent {
            name: "test.agent".into(),
        });

        dir.register(sa1).unwrap();
        assert!(dir.register(sa2).is_err());
    }
}
