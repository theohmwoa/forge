//! Storage backend trait for Forge.
//!
//! A `Storage` is the persistence layer for the content-addressed step graph.
//! In-memory backend ships first; sled and Postgres come next.

use std::collections::HashMap;
use std::sync::Mutex;

use forge_core::{NodeHash, Step};

#[async_trait::async_trait]
pub trait Storage: Send + Sync {
    async fn put(&self, step: Step) -> anyhow::Result<NodeHash>;
    async fn get(&self, hash: &NodeHash) -> anyhow::Result<Option<Step>>;
    async fn children(&self, hash: &NodeHash) -> anyhow::Result<Vec<NodeHash>>;
}

#[derive(Default)]
pub struct MemoryStorage {
    steps: Mutex<HashMap<NodeHash, Step>>,
    children: Mutex<HashMap<NodeHash, Vec<NodeHash>>>,
}

impl MemoryStorage {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait::async_trait]
impl Storage for MemoryStorage {
    async fn put(&self, step: Step) -> anyhow::Result<NodeHash> {
        let hash = step.id.clone();
        if let Some(parent) = step.parent.clone() {
            self.children
                .lock()
                .unwrap()
                .entry(parent)
                .or_default()
                .push(hash.clone());
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
}
