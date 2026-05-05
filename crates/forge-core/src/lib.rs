//! Core types for Forge.
//!
//! Every step in an agent run is a content-addressed node. Runs are DAGs over
//! these nodes. This crate defines the shapes and the agent trait; storage and
//! execution live elsewhere.

use serde::{Deserialize, Serialize};

pub mod agent;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeHash(pub String);

impl NodeHash {
    pub fn of_bytes(bytes: &[u8]) -> Self {
        Self(blake3::hash(bytes).to_hex().to_string())
    }
}

impl std::fmt::Display for NodeHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunId(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    pub id: NodeHash,
    pub parent: Option<NodeHash>,
    pub kind: StepKind,
    pub timestamp_ms: u64,
}

impl Step {
    /// Compute the content-addressed id for a step, derived from its parent
    /// and kind. Timestamp is intentionally excluded so logically-identical
    /// continuations of the same prefix collapse to the same node.
    pub fn compute_id(parent: &Option<NodeHash>, kind: &StepKind) -> NodeHash {
        let payload = serde_json::to_vec(&(parent, kind))
            .expect("step kinds and node hashes are always serializable");
        NodeHash::of_bytes(&payload)
    }

    pub fn new(parent: Option<NodeHash>, kind: StepKind, timestamp_ms: u64) -> Self {
        let id = Self::compute_id(&parent, &kind);
        Self {
            id,
            parent,
            kind,
            timestamp_ms,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StepKind {
    Prompt {
        model: String,
        content: String,
    },
    Message {
        role: String,
        content: String,
    },
    ToolCall {
        call_id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        call_id: String,
        output: serde_json::Value,
    },
}

#[derive(thiserror::Error, Debug)]
pub enum CoreError {
    #[error("step not found: {0}")]
    StepNotFound(NodeHash),
    #[error("hash mismatch: computed {computed}, declared {declared}")]
    HashMismatch {
        computed: NodeHash,
        declared: NodeHash,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_stable() {
        let a = NodeHash::of_bytes(b"hello");
        let b = NodeHash::of_bytes(b"hello");
        assert_eq!(a, b);
    }

    #[test]
    fn hash_differs_on_input() {
        assert_ne!(NodeHash::of_bytes(b"a"), NodeHash::of_bytes(b"b"));
    }

    #[test]
    fn step_id_changes_with_parent() {
        let kind = StepKind::Message {
            role: "assistant".into(),
            content: "hi".into(),
        };
        let a = Step::new(None, kind.clone(), 0);
        let b = Step::new(Some(NodeHash::of_bytes(b"prev")), kind, 0);
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn step_id_stable_across_timestamps() {
        let kind = StepKind::Message {
            role: "assistant".into(),
            content: "hi".into(),
        };
        let a = Step::new(None, kind.clone(), 100);
        let b = Step::new(None, kind, 999);
        assert_eq!(a.id, b.id);
    }
}
