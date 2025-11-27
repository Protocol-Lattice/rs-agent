use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use futures::future::BoxFuture;
use parking_lot::RwLock;
use rs_utcp::providers::base::Provider;
use rs_utcp::providers::cli::CliProvider;
use rs_utcp::tools::Tool as UtcpTool;
use rs_utcp::transports::stream::StreamResult;
use rs_utcp::transports::CommunicationProtocol;
use serde_json::Value;

/// Handler type for in-process UTCP tools.
pub(crate) type InProcessHandler =
    Arc<dyn Fn(HashMap<String, Value>) -> BoxFuture<'static, Result<Value>> + Send + Sync>;

/// UTCP tool paired with an in-process handler.
#[derive(Clone)]
pub(crate) struct InProcessTool {
    pub spec: UtcpTool,
    pub handler: InProcessHandler,
}

/// Transport shim that routes CLI providers to in-process handlers while
/// delegating everything else to the original transport.
pub(crate) struct AgentCliTransport {
    inner: Arc<dyn CommunicationProtocol>,
    tools: RwLock<HashMap<String, Vec<InProcessTool>>>,
}

impl AgentCliTransport {
    pub fn new(inner: Arc<dyn CommunicationProtocol>) -> Self {
        Self {
            inner,
            tools: RwLock::new(HashMap::new()),
        }
    }

    pub fn register(&self, provider: &str, tool: InProcessTool) {
        let mut guard = self.tools.write();
        guard.entry(provider.to_string()).or_default().push(tool);
    }

    fn lookup_handler(&self, provider: &str, tool_name: &str) -> Option<InProcessHandler> {
        let guard = self.tools.read();
        let list = guard.get(provider)?;
        let handler = list.iter().find(|t| {
            t.spec.name == tool_name
                || t.spec
                    .name
                    .rsplit('.')
                    .next()
                    .map(|suffix| suffix == tool_name)
                    .unwrap_or(false)
        })?;
        Some(handler.handler.clone())
    }

    fn specs_for(&self, provider: &str) -> Option<Vec<UtcpTool>> {
        let guard = self.tools.read();
        guard
            .get(provider)
            .map(|tools| tools.iter().map(|t| t.spec.clone()).collect())
    }
}

#[async_trait]
impl CommunicationProtocol for AgentCliTransport {
    async fn register_tool_provider(&self, prov: &dyn Provider) -> Result<Vec<UtcpTool>> {
        if let Some(cli) = prov.as_any().downcast_ref::<CliProvider>() {
            if let Some(specs) = self.specs_for(&cli.base.name) {
                return Ok(specs);
            }
        }
        self.inner.register_tool_provider(prov).await
    }

    async fn deregister_tool_provider(&self, prov: &dyn Provider) -> Result<()> {
        if let Some(cli) = prov.as_any().downcast_ref::<CliProvider>() {
            if self.tools.write().remove(&cli.base.name).is_some() {
                return Ok(());
            }
        }
        self.inner.deregister_tool_provider(prov).await
    }

    async fn call_tool(
        &self,
        tool_name: &str,
        args: HashMap<String, Value>,
        prov: &dyn Provider,
    ) -> Result<Value> {
        if let Some(cli) = prov.as_any().downcast_ref::<CliProvider>() {
            if let Some(handler) = self.lookup_handler(&cli.base.name, tool_name) {
                return handler(args).await;
            }
        }
        self.inner.call_tool(tool_name, args, prov).await
    }

    async fn call_tool_stream(
        &self,
        tool_name: &str,
        args: HashMap<String, Value>,
        prov: &dyn Provider,
    ) -> Result<Box<dyn StreamResult>> {
        if let Some(cli) = prov.as_any().downcast_ref::<CliProvider>() {
            if self.tools.read().contains_key(&cli.base.name) {
                return Err(anyhow!(
                    "Streaming not supported for in-process tool {}",
                    tool_name
                ));
            }
        }
        self.inner.call_tool_stream(tool_name, args, prov).await
    }
}

/// Register (or retrieve) the global agent CLI transport, ensuring it replaces the default CLI transport.
pub(crate) fn ensure_agent_cli_transport() -> Arc<AgentCliTransport> {
    use std::sync::OnceLock;

    static TRANSPORT: OnceLock<Arc<AgentCliTransport>> = OnceLock::new();

    TRANSPORT
        .get_or_init(|| {
            let snapshot = rs_utcp::transports::communication_protocols_snapshot();
            let fallback = snapshot
                .get("cli")
                .unwrap_or_else(|| Arc::new(rs_utcp::transports::cli::CliTransport::new()));

            let shim = Arc::new(AgentCliTransport::new(fallback));
            // Replace the global CLI transport so existing clients pick up the shim.
            rs_utcp::transports::register_communication_protocol("cli", shim.clone());
            shim
        })
        .clone()
}
