//! Sled-backed `Storage`. Single-process, persistent.
//!
//! Layout:
//! - `steps`     : `hash` -> JSON-encoded `Step`
//! - `children`  : `hash` -> JSON-encoded `Vec<NodeHash>`
//! - `runs`      : `head_hash` -> JSON-encoded `RunMeta`
//!
//! Concurrency note: read-modify-write of the children index is not safe
//! under multi-process writers. Use `PostgresStorage` for that.

use std::path::Path;

use async_trait::async_trait;
use forge_core::{NodeHash, Step};

use crate::{RunMeta, Storage};

const TREE_STEPS: &str = "steps";
const TREE_CHILDREN: &str = "children";
const TREE_RUNS: &str = "runs";

pub struct SledStorage {
    _db: sled::Db,
    steps: sled::Tree,
    children: sled::Tree,
    runs: sled::Tree,
}

impl SledStorage {
    pub fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let db = sled::open(path)?;
        let steps = db.open_tree(TREE_STEPS)?;
        let children = db.open_tree(TREE_CHILDREN)?;
        let runs = db.open_tree(TREE_RUNS)?;
        Ok(Self {
            _db: db,
            steps,
            children,
            runs,
        })
    }
}

#[async_trait]
impl Storage for SledStorage {
    async fn put(&self, step: Step) -> anyhow::Result<NodeHash> {
        let hash = step.id.clone();
        let bytes = serde_json::to_vec(&step)?;
        self.steps.insert(hash.0.as_bytes(), bytes)?;

        if let Some(parent) = &step.parent {
            let key = parent.0.as_bytes();
            let mut kids: Vec<NodeHash> = match self.children.get(key)? {
                Some(b) => serde_json::from_slice(&b).unwrap_or_default(),
                None => Vec::new(),
            };
            if !kids.contains(&hash) {
                kids.push(hash.clone());
                self.children.insert(key, serde_json::to_vec(&kids)?)?;
            }
        }
        Ok(hash)
    }

    async fn get(&self, hash: &NodeHash) -> anyhow::Result<Option<Step>> {
        match self.steps.get(hash.0.as_bytes())? {
            Some(b) => Ok(Some(serde_json::from_slice(&b)?)),
            None => Ok(None),
        }
    }

    async fn children(&self, hash: &NodeHash) -> anyhow::Result<Vec<NodeHash>> {
        match self.children.get(hash.0.as_bytes())? {
            Some(b) => Ok(serde_json::from_slice(&b)?),
            None => Ok(Vec::new()),
        }
    }

    async fn record_run(&self, meta: &RunMeta) -> anyhow::Result<()> {
        let bytes = serde_json::to_vec(meta)?;
        self.runs.insert(meta.head.0.as_bytes(), bytes)?;
        Ok(())
    }

    async fn run_meta(&self, head: &NodeHash) -> anyhow::Result<Option<RunMeta>> {
        match self.runs.get(head.0.as_bytes())? {
            Some(b) => Ok(Some(serde_json::from_slice(&b)?)),
            None => Ok(None),
        }
    }

    async fn list_runs(&self) -> anyhow::Result<Vec<RunMeta>> {
        let mut out = Vec::new();
        for kv in self.runs.iter() {
            let (_, v) = kv?;
            out.push(serde_json::from_slice::<RunMeta>(&v)?);
        }
        out.sort_by_key(|m| std::cmp::Reverse(m.recorded_at_ms));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_core::StepKind;

    #[tokio::test]
    async fn round_trips_a_chain() {
        let dir = tempfile::tempdir().unwrap();
        let store = SledStorage::open(dir.path()).unwrap();

        let s0 = Step::new(
            None,
            StepKind::Message {
                role: "user".into(),
                content: "hi".into(),
            },
            1,
        );
        let s1 = Step::new(
            Some(s0.id.clone()),
            StepKind::Message {
                role: "assistant".into(),
                content: "hello".into(),
            },
            2,
        );

        store.put(s0.clone()).await.unwrap();
        store.put(s1.clone()).await.unwrap();

        let got = store.get(&s1.id).await.unwrap().unwrap();
        assert_eq!(got.parent, Some(s0.id.clone()));

        let kids = store.children(&s0.id).await.unwrap();
        assert_eq!(kids, vec![s1.id.clone()]);

        let chain = store.chain_to(&s1.id).await.unwrap();
        assert_eq!(chain.len(), 2);
        assert_eq!(chain[0].id, s0.id);
        assert_eq!(chain[1].id, s1.id);
    }

    #[tokio::test]
    async fn put_is_idempotent_in_children_index() {
        let dir = tempfile::tempdir().unwrap();
        let store = SledStorage::open(dir.path()).unwrap();
        let s0 = Step::new(
            None,
            StepKind::Message {
                role: "user".into(),
                content: "x".into(),
            },
            0,
        );
        let s1 = Step::new(
            Some(s0.id.clone()),
            StepKind::Message {
                role: "assistant".into(),
                content: "y".into(),
            },
            0,
        );
        store.put(s0.clone()).await.unwrap();
        store.put(s1.clone()).await.unwrap();
        store.put(s1.clone()).await.unwrap();
        let kids = store.children(&s0.id).await.unwrap();
        assert_eq!(kids.len(), 1, "duplicate put should not duplicate child");
    }
}
