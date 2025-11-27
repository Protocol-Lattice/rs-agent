use async_trait::async_trait;
use sqlx::PgPool;

use crate::error::{AgentError, Result};
use crate::memory::{MemoryRecord, MemoryStore};

/// PostgreSQL memory store with pgvector support
pub struct PostgresStore {
    pool: PgPool,
}

impl PostgresStore {
    /// Creates a new PostgreSQL store
    pub async fn new(database_url: &str) -> Result<Self> {
        let pool = PgPool::connect(database_url).await.map_err(|e| {
            AgentError::MemoryError(format!("Failed to connect to PostgreSQL: {}", e))
        })?;

        // Create table if not exists
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS memories (
                id UUID PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                importance REAL NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                metadata JSONB,
                embedding vector(384)
            );
            
            CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
            CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp DESC);
            "#,
        )
        .execute(&pool)
        .await
        .map_err(|e| AgentError::MemoryError(format!("Failed to create table: {}", e)))?;

        Ok(Self { pool })
    }

    /// Create embedding index for faster searches
    pub async fn create_embedding_index(&self) -> Result<()> {
        sqlx::query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_memories_embedding 
            ON memories USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| AgentError::MemoryError(format!("Failed to create index: {}", e)))?;

        Ok(())
    }
}

#[async_trait]
impl MemoryStore for PostgresStore {
    async fn store(&self, record: MemoryRecord) -> Result<()> {
        let embedding_vec: Option<Vec<f32>> = record.embedding;
        let metadata_json = record
            .metadata
            .as_ref()
            .map(|m| serde_json::to_value(m).ok())
            .flatten();

        sqlx::query(
            r#"
            INSERT INTO memories (id, session_id, role, content, importance, timestamp, metadata, embedding)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (id) DO UPDATE SET
                content = EXCLUDED.content,
                importance = EXCLUDED.importance,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding
            "#,
        )
        .bind(record.id)
        .bind(&record.session_id)
        .bind(&record.role)
        .bind(&record.content)
        .bind(record.importance)
        .bind(record.timestamp)
        .bind(metadata_json)
        .bind(embedding_vec)
        .execute(&self.pool)
        .await
        .map_err(|e| AgentError::MemoryError(format!("Failed to store memory: {}", e)))?;

        Ok(())
    }

    async fn retrieve(&self, session_id: &str, limit: usize) -> Result<Vec<MemoryRecord>> {
        let records = sqlx::query_as::<
            _,
            (
                uuid::Uuid,
                String,
                String,
                String,
                f32,
                chrono::DateTime<chrono::Utc>,
                Option<serde_json::Value>,
                Option<Vec<f32>>,
            ),
        >(
            r#"SELECT id, session_id, role, content, importance, timestamp, metadata, embedding
               FROM memories
               WHERE session_id = $1
               ORDER BY timestamp DESC
               LIMIT $2"#,
        )
        .bind(session_id)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| AgentError::MemoryError(format!("Failed to retrieve memories: {}", e)))?;

        Ok(records
            .into_iter()
            .map(
                |(id, session_id, role, content, importance, timestamp, metadata, embedding)| {
                    MemoryRecord {
                        id,
                        session_id,
                        role,
                        content,
                        importance,
                        timestamp,
                        metadata: metadata.and_then(|v| serde_json::from_value(v).ok()),
                        embedding,
                    }
                },
            )
            .collect())
    }

    async fn search(
        &self,
        session_id: &str,
        query_embedding: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>> {
        let records = sqlx::query_as::<
            _,
            (
                uuid::Uuid,
                String,
                String,
                String,
                f32,
                chrono::DateTime<chrono::Utc>,
                Option<serde_json::Value>,
                Option<Vec<f32>>,
            ),
        >(
            r#"SELECT id, session_id, role, content, importance, timestamp, metadata, embedding
               FROM memories
               WHERE session_id = $1 AND embedding IS NOT NULL
               ORDER BY embedding <=> $2
               LIMIT $3"#,
        )
        .bind(session_id)
        .bind(&query_embedding)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| AgentError::MemoryError(format!("Failed to search memories: {}", e)))?;

        Ok(records
            .into_iter()
            .map(
                |(id, session_id, role, content, importance, timestamp, metadata, embedding)| {
                    MemoryRecord {
                        id,
                        session_id,
                        role,
                        content,
                        importance,
                        timestamp,
                        metadata: metadata.and_then(|v| serde_json::from_value(v).ok()),
                        embedding,
                    }
                },
            )
            .collect())
    }

    async fn flush(&self) -> Result<()> {
        // PostgreSQL commits automatically
        Ok(())
    }
}
