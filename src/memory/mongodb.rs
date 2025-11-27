use async_trait::async_trait;
use mongodb::bson::{doc, Document};
use mongodb::{Client, Collection};

use crate::error::{AgentError, Result};
use crate::memory::{MemoryRecord, MemoryStore};

/// MongoDB memory store
pub struct MongoStore {
    collection: Collection<Document>,
}

impl MongoStore {
    /// Creates a new MongoDB store
    pub async fn new(connection_string: &str, database: &str, collection: &str) -> Result<Self> {
        let client = Client::with_uri_str(connection_string)
            .await
            .map_err(|e| AgentError::MemoryError(format!("Failed to connect to MongoDB: {}", e)))?;

        let db = client.database(database);
        let collection = db.collection::<Document>(collection);

        // Create indexes
        collection
            .create_index(
                mongodb::IndexModel::builder()
                    .keys(doc! { "session_id": 1, "timestamp": -1 })
                    .build(),
            )
            .await
            .map_err(|e| AgentError::MemoryError(format!("Failed to create index: {}", e)))?;

        Ok(Self { collection })
    }

    /// Create vector search index (MongoDB Atlas Search required)
    pub async fn create_vector_index(&self, index_name: &str) -> Result<()> {
        // Note: This requires MongoDB Atlas with Vector Search enabled
        // The actual index creation is done through Atlas UI or CLI
        tracing::info!(
            "Vector search index {} would be created through Atlas",
            index_name
        );
        Ok(())
    }
}

#[async_trait]
impl MemoryStore for MongoStore {
    async fn store(&self, record: MemoryRecord) -> Result<()> {
        let mut doc = doc! {
            "_id": record.id.to_string(),
            "session_id": &record.session_id,
            "role": &record.role,
            "content": &record.content,
            "importance": record.importance,
            "timestamp": mongodb::bson::DateTime::from_chrono(record.timestamp),
        };

        if let Some(metadata) = &record.metadata {
            let metadata_doc = serde_json::to_value(metadata)
                .map_err(|e| AgentError::SerializationError(e))
                .and_then(|v| {
                    mongodb::bson::to_bson(&v).map_err(|e| {
                        AgentError::MemoryError(format!("Failed to convert metadata: {}", e))
                    })
                })?;
            doc.insert("metadata", metadata_doc);
        }

        if let Some(embedding) = &record.embedding {
            doc.insert("embedding", embedding);
        }

        self.collection
            .replace_one(doc! { "_id": record.id.to_string() }, doc.clone())
            .upsert(true)
            .await
            .map_err(|e| AgentError::MemoryError(format!("Failed to store memory: {}", e)))?;

        Ok(())
    }

    async fn retrieve(&self, session_id: &str, limit: usize) -> Result<Vec<MemoryRecord>> {
        let filter = doc! { "session_id": session_id };
        let options = mongodb::options::FindOptions::builder()
            .sort(doc! { "timestamp": -1 })
            .limit(limit as i64)
            .build();

        let mut cursor = self
            .collection
            .find(filter)
            .with_options(options)
            .await
            .map_err(|e| AgentError::MemoryError(format!("Failed to retrieve memories: {}", e)))?;

        let mut records = Vec::new();
        while cursor
            .advance()
            .await
            .map_err(|e| AgentError::MemoryError(format!("Failed to advance cursor: {}", e)))?
        {
            let doc = cursor.current();
            records.push(document_to_memory_record(doc)?);
        }

        Ok(records)
    }

    async fn search(
        &self,
        session_id: &str,
        query_embedding: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>> {
        // For basic MongoDB, we'll do client-side vector search
        // For MongoDB Atlas, you'd use $vectorSearch aggregation
        let all_records = self.retrieve(session_id, 1000).await?; // Get larger set

        let mut scored: Vec<(f32, MemoryRecord)> = all_records
            .into_iter()
            .filter(|r| r.embedding.is_some())
            .map(|r| {
                let embedding = r.embedding.as_ref().unwrap();
                let similarity = super::cosine_similarity(&query_embedding, embedding);
                (similarity, r)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        Ok(scored.into_iter().take(limit).map(|(_, r)| r).collect())
    }

    async fn flush(&self) -> Result<()> {
        // MongoDB commits automatically
        Ok(())
    }
}

fn document_to_memory_record(doc: &Document) -> Result<MemoryRecord> {
    let id = doc
        .get_str("_id")
        .ok()
        .and_then(|s| uuid::Uuid::parse_str(s).ok())
        .ok_or_else(|| AgentError::MemoryError("Missing or invalid _id".to_string()))?;

    let session_id = doc
        .get_str("session_id")
        .map_err(|_| AgentError::MemoryError("Missing session_id".to_string()))?
        .to_string();

    let role = doc
        .get_str("role")
        .map_err(|_| AgentError::MemoryError("Missing role".to_string()))?
        .to_string();

    let content = doc
        .get_str("content")
        .map_err(|_| AgentError::MemoryError("Missing content".to_string()))?
        .to_string();

    let importance = doc.get_f64("importance").unwrap_or(0.5) as f32;

    let timestamp = doc
        .get_datetime("timestamp")
        .ok()
        .map(|dt| dt.to_chrono())
        .unwrap_or_else(chrono::Utc::now);

    let metadata = doc
        .get_document("metadata")
        .ok()
        .and_then(|d| mongodb::bson::from_bson(mongodb::bson::Bson::Document(d.clone())).ok());

    let embedding = doc.get_array("embedding").ok().and_then(|arr| {
        arr.iter()
            .map(|v| v.as_f64().map(|f| f as f32))
            .collect::<Option<Vec<f32>>>()
    });

    Ok(MemoryRecord {
        id,
        session_id,
        role,
        content,
        importance,
        timestamp,
        metadata,
        embedding,
    })
}
