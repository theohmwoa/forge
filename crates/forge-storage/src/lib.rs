//! Storage backends for Forge.
//!
//! [`Storage`] is the persistence trait. Three implementations ship in this
//! crate:
//! - [`MemoryStorage`]   — in-process, ephemeral; for tests.
//! - [`SledStorage`]     — single-process durable embedded DB.
//! - [`PostgresStorage`] — multi-process, multi-machine; for production.
//!
//! All run-graph operations (`record_run`, `list_runs`, `chain_to`) live on
//! the trait so backends can express them efficiently. `chain_to` has a
//! default implementation walking parents via `get`, which any backend can
//! override with a recursive CTE or similar.

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use forge_core::{NodeHash, Step};
use serde::{Deserialize, Serialize};

mod postgres_store;
mod sled_store;
pub use postgres_store::PostgresStorage;
pub use sled_store::SledStorage;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMeta {
    pub head: NodeHash,
    pub root: NodeHash,
    pub recorded_at_ms: u64,
    /// Optional free-form label, set via `x-forge-tag` header on the
    /// recorder or `--tag` on `forge run`. Useful for filtering runs by
    /// experiment / dataset / user.
    #[serde(default)]
    pub tag: Option<String>,
}

#[async_trait]
pub trait Storage: Send + Sync {
    async fn put(&self, step: Step) -> anyhow::Result<NodeHash>;
    async fn get(&self, hash: &NodeHash) -> anyhow::Result<Option<Step>>;
    async fn children(&self, hash: &NodeHash) -> anyhow::Result<Vec<NodeHash>>;

    async fn record_run(&self, meta: &RunMeta) -> anyhow::Result<()>;
    async fn run_meta(&self, head: &NodeHash) -> anyhow::Result<Option<RunMeta>>;
    async fn list_runs(&self) -> anyhow::Result<Vec<RunMeta>>;

    /// Walk parents from `head` and return the chain in root-to-head order.
    /// Default implementation uses `get`; backends may override with a
    /// single-query implementation (recursive CTE on Postgres, etc.).
    async fn chain_to(&self, head: &NodeHash) -> anyhow::Result<Vec<Step>> {
        let mut out = Vec::new();
        let mut cursor = Some(head.clone());
        while let Some(h) = cursor {
            let step = self
                .get(&h)
                .await?
                .ok_or_else(|| anyhow::anyhow!("step missing while walking chain: {h}"))?;
            cursor = step.parent.clone();
            out.push(step);
        }
        out.reverse();
        Ok(out)
    }
}

#[derive(Default)]
pub struct MemoryStorage {
    steps: Mutex<HashMap<NodeHash, Step>>,
    children: Mutex<HashMap<NodeHash, Vec<NodeHash>>>,
    runs: Mutex<HashMap<NodeHash, RunMeta>>,
}

impl MemoryStorage {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl Storage for MemoryStorage {
    async fn put(&self, step: Step) -> anyhow::Result<NodeHash> {
        let hash = step.id.clone();
        if let Some(parent) = step.parent.clone() {
            let mut guard = self.children.lock().unwrap();
            let entry = guard.entry(parent).or_default();
            if !entry.contains(&hash) {
                entry.push(hash.clone());
            }
        }
        self.steps.lock().unwrap().insert(hash.clone(), step);
        Ok(hash)
    }

    async fn get(&self, hash: &NodeHash) -> anyhow::Result<Option<Step>> {
        Ok(self.steps.lock().unwrap().get(hash).cloned())
    }

    async fn children(&self, hash: &NodeHash) -> anyhow::Result<Vec<NodeHash>> {
        Ok(self
            .children
            .lock()
            .unwrap()
            .get(hash)
            .cloned()
            .unwrap_or_default())
    }

    async fn record_run(&self, meta: &RunMeta) -> anyhow::Result<()> {
        self.runs
            .lock()
            .unwrap()
            .insert(meta.head.clone(), meta.clone());
        Ok(())
    }

    async fn run_meta(&self, head: &NodeHash) -> anyhow::Result<Option<RunMeta>> {
        Ok(self.runs.lock().unwrap().get(head).cloned())
    }

    async fn list_runs(&self) -> anyhow::Result<Vec<RunMeta>> {
        let mut out: Vec<RunMeta> = self.runs.lock().unwrap().values().cloned().collect();
        out.sort_by_key(|r| std::cmp::Reverse(r.recorded_at_ms));
        Ok(out)
    }
}
