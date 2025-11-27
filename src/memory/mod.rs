use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::error::Result;

// Memory backend implementations
#[cfg(feature = "postgres")]
pub mod postgres;

#[cfg(feature = "qdrant")]
pub mod qdrant;

#[cfg(feature = "mongodb")]
pub mod mongodb;

// Re-export backends
#[cfg(feature = "postgres")]
pub use postgres::PostgresStore;

#[cfg(feature = "qdrant")]
pub use qdrant::QdrantStore;

#[cfg(feature = "mongodb")]
pub use mongodb::MongoStore;

/// Memory record storing a piece of information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: Uuid,
    pub session_id: String,
    pub role: String,
    pub content: String,
    pub importance: f32,
    pub timestamp: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

/// Memory store trait for different backends
#[async_trait::async_trait]
pub trait MemoryStore: Send + Sync {
    /// Stores a memory record
    async fn store(&self, record: MemoryRecord) -> Result<()>;

    /// Retrieves memories for a session
    async fn retrieve(&self, session_id: &str, limit: usize) -> Result<Vec<MemoryRecord>>;

    /// Searches for similar memories using embeddings
    async fn search(
        &self,
        session_id: &str,
        query_embedding: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>>;

    /// Flushes all pending writes
    async fn flush(&self) -> Result<()>;
}

/// In-memory store implementation
pub struct InMemoryStore {
    records: parking_lot::RwLock<Vec<MemoryRecord>>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self {
            records: parking_lot::RwLock::new(Vec::new()),
        }
    }
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl MemoryStore for InMemoryStore {
    async fn store(&self, record: MemoryRecord) -> Result<()> {
        let mut records = self.records.write();
        records.push(record);
        Ok(())
    }

    async fn retrieve(&self, session_id: &str, limit: usize) -> Result<Vec<MemoryRecord>> {
        let records = self.records.read();
        let filtered: Vec<MemoryRecord> = records
            .iter()
            .filter(|r| r.session_id == session_id)
            .rev()
            .take(limit)
            .cloned()
            .collect();
        Ok(filtered)
    }

    async fn search(
        &self,
        session_id: &str,
        query_embedding: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>> {
        let records = self.records.read();
        let mut scored: Vec<(f32, MemoryRecord)> = records
            .iter()
            .filter(|r| r.session_id == session_id && r.embedding.is_some())
            .map(|r| {
                let embedding = r.embedding.as_ref().unwrap();
                let similarity = cosine_similarity(&query_embedding, embedding);
                (similarity, r.clone())
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        Ok(scored.into_iter().take(limit).map(|(_, r)| r).collect())
    }

    async fn flush(&self) -> Result<()> {
        Ok(())
    }
}

/// Calculates cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        0.0
    } else {
        dot / (mag_a * mag_b)
    }
}

/// Maximal Marginal Relevance (MMR) for diverse retrieval
///
/// Balances relevance to query with diversity in results.
/// Lambda controls the trade-off: 1.0 = pure relevance, 0.0 = pure diversity
pub fn mmr_rerank(
    query_embedding: &[f32],
    candidates: Vec<MemoryRecord>,
    k: usize,
    lambda: f32,
) -> Vec<MemoryRecord> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let k = k.min(candidates.len());
    let mut selected = Vec::with_capacity(k);
    let mut remaining = candidates;

    // Select first item with highest similarity to query
    if let Some((idx, _)) = remaining
        .iter()
        .enumerate()
        .filter_map(|(i, r)| {
            r.embedding
                .as_ref()
                .map(|emb| (i, cosine_similarity(query_embedding, emb)))
        })
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    {
        selected.push(remaining.swap_remove(idx));
    }

    // Iteratively select items that maximize MMR score
    while selected.len() < k && !remaining.is_empty() {
        let next_idx = remaining
            .iter()
            .enumerate()
            .filter_map(|(i, r)| {
                let emb = r.embedding.as_ref()?;

                // Relevance: similarity to query
                let relevance = cosine_similarity(query_embedding, emb);

                // Diversity: max similarity to already selected items
                let max_sim_selected = selected
                    .iter()
                    .filter_map(|s| s.embedding.as_ref())
                    .map(|s_emb| cosine_similarity(emb, s_emb))
                    .fold(f32::NEG_INFINITY, f32::max);

                // MMR score: λ * relevance - (1-λ) * max_similarity_to_selected
                let mmr_score = lambda * relevance - (1.0 - lambda) * max_sim_selected;

                Some((i, mmr_score))
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i);

        if let Some(idx) = next_idx {
            selected.push(remaining.swap_remove(idx));
        } else {
            break;
        }
    }

    selected
}

/// Session memory manages short-term and long-term memory for a session
pub struct SessionMemory {
    store: Box<dyn MemoryStore>,
    // Short-term cache of recent messages
    short_term: parking_lot::RwLock<HashMap<String, Vec<MemoryRecord>>>,
    context_window: usize,
}

impl SessionMemory {
    /// Creates a new session memory with the given store
    pub fn new(store: Box<dyn MemoryStore>, context_window: usize) -> Self {
        Self {
            store,
            short_term: parking_lot::RwLock::new(HashMap::new()),
            context_window,
        }
    }

    /// Stores a memory record
    pub async fn store(&self, record: MemoryRecord) -> Result<()> {
        let session_id = record.session_id.clone();

        // Add to short-term cache
        {
            let mut short_term = self.short_term.write();
            let session_records = short_term.entry(session_id).or_insert_with(Vec::new);
            session_records.push(record.clone());

            // Trim to context window
            if session_records.len() > self.context_window {
                session_records.drain(0..session_records.len() - self.context_window);
            }
        }

        // Store in long-term
        self.store.store(record).await
    }

    /// Retrieves recent memories from short-term cache
    pub async fn retrieve_recent(&self, session_id: &str) -> Result<Vec<MemoryRecord>> {
        let short_term = self.short_term.read();
        Ok(short_term.get(session_id).cloned().unwrap_or_default())
    }

    /// Searches for relevant memories
    pub async fn search(
        &self,
        session_id: &str,
        query_embedding: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>> {
        self.store.search(session_id, query_embedding, limit).await
    }

    /// Flushes all pending writes
    pub async fn flush(&self) -> Result<()> {
        self.store.flush().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_in_memory_store() {
        let store = InMemoryStore::new();
        let record = MemoryRecord {
            id: Uuid::new_v4(),
            session_id: "test".to_string(),
            role: "user".to_string(),
            content: "Hello".to_string(),
            importance: 0.8,
            timestamp: Utc::now(),
            metadata: None,
            embedding: None,
        };

        store.store(record.clone()).await.unwrap();
        let retrieved = store.retrieve("test", 10).await.unwrap();
        assert_eq!(retrieved.len(), 1);
        assert_eq!(retrieved[0].content, "Hello");
    }

    #[tokio::test]
    async fn test_session_memory() {
        let store = Box::new(InMemoryStore::new());
        let memory = SessionMemory::new(store, 5);

        let record = MemoryRecord {
            id: Uuid::new_v4(),
            session_id: "test".to_string(),
            role: "user".to_string(),
            content: "Test message".to_string(),
            importance: 0.9,
            timestamp: Utc::now(),
            metadata: None,
            embedding: None,
        };

        memory.store(record).await.unwrap();
        let recent = memory.retrieve_recent("test").await.unwrap();
        assert_eq!(recent.len(), 1);
    }
}
