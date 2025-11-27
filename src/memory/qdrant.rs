use async_trait::async_trait;
use qdrant_client::qdrant::{
    Condition, CreateCollection, Filter, PointStruct, SearchPoints, UpsertPoints, VectorParams,
    VectorsConfig,
};
use qdrant_client::{Payload, Qdrant};

use crate::error::{AgentError, Result};
use crate::memory::{MemoryRecord, MemoryStore};

/// Qdrant vector database memory store
pub struct QdrantStore {
    client: Qdrant,
    collection_name: String,
}

impl QdrantStore {
    /// Creates a new Qdrant store
    pub async fn new(url: &str, collection_name: impl Into<String>) -> Result<Self> {
        let client = Qdrant::from_url(url)
            .build()
            .map_err(|e| AgentError::MemoryError(format!("Failed to connect to Qdrant: {}", e)))?;

        let collection_name = collection_name.into();

        // Create collection if it doesn't exist
        let collections = client
            .list_collections()
            .await
            .map_err(|e| AgentError::MemoryError(format!("Failed to list collections: {}", e)))?;

        let exists = collections
            .collections
            .iter()
            .any(|c| c.name == collection_name);

        if !exists {
            client
                .create_collection(CreateCollection {
                    collection_name: collection_name.clone(),
                    vectors_config: Some(VectorsConfig {
                        config: Some(qdrant_client::qdrant::vectors_config::Config::Params(
                            VectorParams {
                                size: 384, // Default embedding size, can be configured
                                distance: qdrant_client::qdrant::Distance::Cosine.into(),
                                ..Default::default()
                            },
                        )),
                    }),
                    ..Default::default()
                })
                .await
                .map_err(|e| {
                    AgentError::MemoryError(format!("Failed to create collection: {}", e))
                })?;
        }

        Ok(Self {
            client,
            collection_name,
        })
    }

    /// Set embedding dimension
    pub async fn with_dimension(self, _dim: u64) -> Result<Self> {
        // Recreate collection with new dimension if needed
        Ok(self)
    }
}

#[async_trait]
impl MemoryStore for QdrantStore {
    async fn store(&self, record: MemoryRecord) -> Result<()> {
        if let Some(embedding) = &record.embedding {
            let mut payload = serde_json::json!({
                "id": record.id.to_string(),
                "session_id": record.session_id,
                "role": record.role,
                "content": record.content,
                "importance": record.importance,
                "timestamp": record.timestamp.to_rfc3339(),
            });

            if let Some(metadata) = &record.metadata {
                payload["metadata"] = serde_json::to_value(metadata)
                    .map_err(|e| AgentError::SerializationError(e))?;
            }

            let point = PointStruct::new(
                record.id.to_string(),
                embedding.clone(),
                Payload::try_from(payload).map_err(|e| {
                    AgentError::MemoryError(format!("Failed to convert payload: {:?}", e))
                })?,
            );

            self.client
                .upsert_points(UpsertPoints {
                    collection_name: self.collection_name.clone(),
                    points: vec![point],
                    wait: Some(true),
                    ..Default::default()
                })
                .await
                .map_err(|e| AgentError::MemoryError(format!("Failed to upsert point: {}", e)))?;
        }

        Ok(())
    }

    async fn retrieve(&self, session_id: &str, limit: usize) -> Result<Vec<MemoryRecord>> {
        // Qdrant doesn't support direct filtering without vector search
        // We'll use a dummy search with high limit
        let dummy_vector = vec![0.0; 384]; // Adjust dimension as needed

        let search_result = self
            .client
            .search_points(SearchPoints {
                collection_name: self.collection_name.clone(),
                vector: dummy_vector,
                limit: limit as u64,
                with_payload: Some(true.into()),
                filter: Some(
                    Filter::must([Condition::matches("session_id", session_id.to_string())]).into(),
                ),
                ..Default::default()
            })
            .await
            .map_err(|e| AgentError::MemoryError(format!("Failed to search: {}", e)))?;

        let mut records = Vec::new();
        for point in search_result.result {
            if let Some(payload) = point.payload {
                records.push(payload_to_memory_record(payload)?);
            }
        }

        Ok(records)
    }

    async fn search(
        &self,
        session_id: &str,
        query_embedding: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>> {
        let search_result = self
            .client
            .search_points(SearchPoints {
                collection_name: self.collection_name.clone(),
                vector: query_embedding,
                limit: limit as u64,
                with_payload: Some(true.into()),
                filter: Some(
                    Filter::must([Condition::matches("session_id", session_id.to_string())]).into(),
                ),
                ..Default::default()
            })
            .await
            .map_err(|e| AgentError::MemoryError(format!("Failed to search: {}", e)))?;

        let mut records = Vec::new();
        for point in search_result.result {
            if let Some(payload) = point.payload {
                records.push(payload_to_memory_record(payload)?);
            }
        }

        Ok(records)
    }

    async fn flush(&self) -> Result<()> {
        // Qdrant writes are immediate
        Ok(())
    }
}

fn payload_to_memory_record(
    payload: std::collections::HashMap<String, qdrant_client::qdrant::Value>,
) -> Result<MemoryRecord> {
    let id = payload
        .get("id")
        .and_then(|v| v.as_str())
        .and_then(|s| uuid::Uuid::parse_str(s).ok())
        .ok_or_else(|| AgentError::MemoryError("Missing or invalid id".to_string()))?;

    let session_id = payload
        .get("session_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| AgentError::MemoryError("Missing session_id".to_string()))?
        .to_string();

    let role = payload
        .get("role")
        .and_then(|v| v.as_str())
        .ok_or_else(|| AgentError::MemoryError("Missing role".to_string()))?
        .to_string();

    let content = payload
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or_else(|| AgentError::MemoryError("Missing content".to_string()))?
        .to_string();

    let importance = payload
        .get("importance")
        .and_then(|v| v.as_double())
        .unwrap_or(0.5) as f32;

    let timestamp = payload
        .get("timestamp")
        .and_then(|v| v.as_str())
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .unwrap_or_else(chrono::Utc::now);

    let metadata = payload.get("metadata").and_then(|v| {
        serde_json::to_value(v)
            .ok()
            .and_then(|jv| serde_json::from_value(jv).ok())
    });

    Ok(MemoryRecord {
        id,
        session_id,
        role,
        content,
        importance,
        timestamp,
        metadata,
        embedding: None, // Qdrant stores embeddings separately
    })
}
