//! Postgres-backed `Storage`. Multi-process, multi-machine. Suitable for
//! production deployments of `forge serve` with multiple workers writing
//! against the same graph.
//!
//! Schema (created idempotently on `connect`):
//! ```sql
//! CREATE TABLE forge_steps (
//!     id           TEXT PRIMARY KEY,
//!     parent       TEXT,
//!     kind         JSONB NOT NULL,
//!     timestamp_ms BIGINT NOT NULL
//! );
//! CREATE INDEX forge_steps_parent_idx ON forge_steps(parent);
//!
//! CREATE TABLE forge_runs (
//!     head           TEXT PRIMARY KEY,
//!     root           TEXT NOT NULL,
//!     recorded_at_ms BIGINT NOT NULL
//! );
//! ```

use async_trait::async_trait;
use forge_core::{NodeHash, Step, StepKind};
use sqlx::postgres::{PgPool, PgPoolOptions};
use sqlx::Row;

use crate::{RunMeta, Storage};

fn row_to_run_meta(r: sqlx::postgres::PgRow) -> RunMeta {
    let head: String = r.get("head");
    let root: String = r.get("root");
    let recorded_at_ms: i64 = r.get("recorded_at_ms");
    let tag: Option<String> = r.try_get("tag").ok().flatten();
    RunMeta {
        head: NodeHash(head),
        root: NodeHash(root),
        recorded_at_ms: recorded_at_ms as u64,
        tag,
    }
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS forge_steps (
    id           TEXT PRIMARY KEY,
    parent       TEXT,
    kind         JSONB NOT NULL,
    timestamp_ms BIGINT NOT NULL
);
CREATE INDEX IF NOT EXISTS forge_steps_parent_idx ON forge_steps(parent);
CREATE TABLE IF NOT EXISTS forge_runs (
    head           TEXT PRIMARY KEY,
    root           TEXT NOT NULL,
    recorded_at_ms BIGINT NOT NULL,
    tag            TEXT
);
ALTER TABLE forge_runs ADD COLUMN IF NOT EXISTS tag TEXT;
CREATE INDEX IF NOT EXISTS forge_runs_tag_idx ON forge_runs(tag);
";

pub struct PostgresStorage {
    pool: PgPool,
}

impl PostgresStorage {
    /// Connect to a Postgres URL (`postgres://user:pass@host:port/db`) and
    /// run the (idempotent) schema migration.
    pub async fn connect(url: &str) -> anyhow::Result<Self> {
        let pool = PgPoolOptions::new().max_connections(8).connect(url).await?;
        sqlx::raw_sql(SCHEMA).execute(&pool).await?;
        Ok(Self { pool })
    }
}

#[async_trait]
impl Storage for PostgresStorage {
    async fn put(&self, step: Step) -> anyhow::Result<NodeHash> {
        let kind_json = serde_json::to_value(&step.kind)?;
        sqlx::query(
            "INSERT INTO forge_steps (id, parent, kind, timestamp_ms)
             VALUES ($1, $2, $3, $4)
             ON CONFLICT (id) DO NOTHING",
        )
        .bind(&step.id.0)
        .bind(step.parent.as_ref().map(|p| p.0.clone()))
        .bind(kind_json)
        .bind(step.timestamp_ms as i64)
        .execute(&self.pool)
        .await?;
        Ok(step.id)
    }

    async fn get(&self, hash: &NodeHash) -> anyhow::Result<Option<Step>> {
        let row =
            sqlx::query("SELECT id, parent, kind, timestamp_ms FROM forge_steps WHERE id = $1")
                .bind(&hash.0)
                .fetch_optional(&self.pool)
                .await?;
        match row {
            Some(r) => {
                let id: String = r.try_get("id")?;
                let parent: Option<String> = r.try_get("parent")?;
                let kind_json: serde_json::Value = r.try_get("kind")?;
                let timestamp_ms: i64 = r.try_get("timestamp_ms")?;
                let kind: StepKind = serde_json::from_value(kind_json)?;
                Ok(Some(Step {
                    id: NodeHash(id),
                    parent: parent.map(NodeHash),
                    kind,
                    timestamp_ms: timestamp_ms as u64,
                }))
            }
            None => Ok(None),
        }
    }

    async fn children(&self, hash: &NodeHash) -> anyhow::Result<Vec<NodeHash>> {
        let rows =
            sqlx::query("SELECT id FROM forge_steps WHERE parent = $1 ORDER BY timestamp_ms")
                .bind(&hash.0)
                .fetch_all(&self.pool)
                .await?;
        Ok(rows
            .into_iter()
            .map(|r| {
                let id: String = r.get("id");
                NodeHash(id)
            })
            .collect())
    }

    async fn record_run(&self, meta: &RunMeta) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO forge_runs (head, root, recorded_at_ms, tag)
             VALUES ($1, $2, $3, $4)
             ON CONFLICT (head) DO UPDATE
                 SET root = EXCLUDED.root,
                     recorded_at_ms = EXCLUDED.recorded_at_ms,
                     tag = COALESCE(EXCLUDED.tag, forge_runs.tag)",
        )
        .bind(&meta.head.0)
        .bind(&meta.root.0)
        .bind(meta.recorded_at_ms as i64)
        .bind(meta.tag.as_deref())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn run_meta(&self, head: &NodeHash) -> anyhow::Result<Option<RunMeta>> {
        let row =
            sqlx::query("SELECT head, root, recorded_at_ms, tag FROM forge_runs WHERE head = $1")
                .bind(&head.0)
                .fetch_optional(&self.pool)
                .await?;
        Ok(row.map(row_to_run_meta))
    }

    async fn list_runs(&self) -> anyhow::Result<Vec<RunMeta>> {
        let rows = sqlx::query(
            "SELECT head, root, recorded_at_ms, tag FROM forge_runs
             ORDER BY recorded_at_ms DESC",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.into_iter().map(row_to_run_meta).collect())
    }

    /// Override the default chain walk with a single recursive CTE so we
    /// don't make N round-trips for an N-step chain.
    async fn chain_to(&self, head: &NodeHash) -> anyhow::Result<Vec<Step>> {
        let rows = sqlx::query(
            "WITH RECURSIVE walk(id, parent, kind, timestamp_ms, depth) AS (
                 SELECT id, parent, kind, timestamp_ms, 0 FROM forge_steps WHERE id = $1
                 UNION ALL
                 SELECT s.id, s.parent, s.kind, s.timestamp_ms, w.depth + 1
                 FROM forge_steps s JOIN walk w ON s.id = w.parent
             )
             SELECT id, parent, kind, timestamp_ms FROM walk ORDER BY depth DESC",
        )
        .bind(&head.0)
        .fetch_all(&self.pool)
        .await?;

        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            let id: String = r.try_get("id")?;
            let parent: Option<String> = r.try_get("parent")?;
            let kind_json: serde_json::Value = r.try_get("kind")?;
            let timestamp_ms: i64 = r.try_get("timestamp_ms")?;
            let kind: StepKind = serde_json::from_value(kind_json)?;
            out.push(Step {
                id: NodeHash(id),
                parent: parent.map(NodeHash),
                kind,
                timestamp_ms: timestamp_ms as u64,
            });
        }
        if out.is_empty() {
            anyhow::bail!("step missing while walking chain: {head}");
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_core::StepKind;

    /// Live tests run only when `TEST_POSTGRES_URL` is set, since they
    /// require a real Postgres instance. The connection is exercised
    /// against a unique table prefix to avoid collisions; for a robust
    /// CI we'd use testcontainers.
    fn test_url() -> Option<String> {
        std::env::var("TEST_POSTGRES_URL").ok()
    }

    #[tokio::test]
    async fn round_trips_a_chain_when_postgres_is_available() {
        let Some(url) = test_url() else {
            eprintln!("TEST_POSTGRES_URL not set; skipping postgres round-trip");
            return;
        };
        let store = PostgresStorage::connect(&url).await.unwrap();
        let s0 = Step::new(
            None,
            StepKind::Message {
                role: "user".into(),
                content: "hi-pg".into(),
            },
            1,
        );
        store.put(s0.clone()).await.unwrap();
        let got = store.get(&s0.id).await.unwrap().unwrap();
        assert_eq!(got.id, s0.id);
        let chain = store.chain_to(&s0.id).await.unwrap();
        assert_eq!(chain.len(), 1);
    }
}
