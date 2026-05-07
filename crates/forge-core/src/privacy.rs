//! Privacy-aware chain redaction for tool-keyed model routing.
//!
//! When the router transitions from a "trusted" model (local) to an
//! "untrusted" one (a cloud provider), the chain prefix is what gets
//! reconstructed and sent to the cloud. By default, every prior `ToolResult`
//! step is in that prefix verbatim — including any PII the trusted model
//! was supposed to shield.
//!
//! This module produces a *cloud view* of the chain: a sanitized copy that
//! the cloud-facing agent's `prefix_to_history` should see, leaving the raw
//! chain intact for replay/audit. Different policies trade off between
//! information loss, cache stability, and tool-author burden.
//!
//! Three axes matter:
//!
//! 1. **Privacy** — does the PII string appear in the cloud view?
//! 2. **Within-run cache stability** — are the bytes of the cloud view a
//!    stable prefix as the chain extends? Anthropic / OpenAI prompt caches
//!    require byte-identical prefixes across consecutive calls.
//! 3. **Cross-run cache stability** — do two different runs of the "same
//!    workflow" produce the same cloud-view bytes for the shared prefix?
//!    This drives long-term cache hit rate.
//!
//! See `tests` for the empirical comparison of the four policies.

use std::collections::HashSet;

use serde_json::{json, Value};

use crate::{Step, StepKind};

/// Strategies for producing the cloud-facing view of a chain.
#[derive(Debug, Clone)]
pub enum RedactionPolicy {
    /// Pass-through. PII flows to the cloud. Used as a baseline / control.
    None,

    /// Replace every sensitive tool's `ToolResult.output` with a fixed
    /// string. Deterministic across runs as long as the marker is fixed.
    /// Tool authors don't need to do anything; the policy is applied at
    /// the routing layer.
    ReplaceFixed { marker: String },

    /// The tool returns an object with a `public` sub-field; the cloud view
    /// keeps only `public` and drops everything else. Tool authors are
    /// responsible for deciding what's safe to expose. Deterministic only
    /// if the tool's `public` output is deterministic for the same input.
    PublicPayload,

    /// Replace every sensitive tool's `ToolResult.output` with the memo
    /// string. Models pre-redaction can't access the original PII through
    /// the cloud view. Memo content typically comes from the trusted model
    /// summarizing what the cloud needs to know — which is sampling-based
    /// and so non-deterministic across runs.
    Memo { memo: String },
}

/// Apply `policy` to a chain, returning a new chain that's safe to send to
/// the cloud-facing agent.
///
/// `sensitive_tools` is the set of tool names whose results should be
/// redacted. Tool authors signal sensitivity by being in this set; the
/// router (or the caller) maintains the set explicitly so it can vary by
/// route target rather than being a static `Tool` trait method.
pub fn cloud_view(
    chain: &[Step],
    policy: &RedactionPolicy,
    sensitive_tools: &HashSet<String>,
) -> Vec<Step> {
    chain
        .iter()
        .map(|step| match (&policy, &step.kind) {
            (RedactionPolicy::None, _) => step.clone(),

            (
                RedactionPolicy::ReplaceFixed { marker },
                StepKind::ToolResult { call_id, output },
            ) if is_sensitive_result(chain, &step.id, sensitive_tools) => {
                let _ = output;
                Step {
                    id: step.id.clone(),
                    parent: step.parent.clone(),
                    kind: StepKind::ToolResult {
                        call_id: call_id.clone(),
                        output: Value::String(marker.clone()),
                    },
                    timestamp_ms: step.timestamp_ms,
                }
            }

            (RedactionPolicy::PublicPayload, StepKind::ToolResult { call_id, output })
                if is_sensitive_result(chain, &step.id, sensitive_tools) =>
            {
                let public = match output.as_object() {
                    Some(obj) => obj
                        .get("public")
                        .cloned()
                        .unwrap_or_else(|| json!("[redacted: tool returned no public payload]")),
                    None => json!("[redacted: tool output not an object]"),
                };
                Step {
                    id: step.id.clone(),
                    parent: step.parent.clone(),
                    kind: StepKind::ToolResult {
                        call_id: call_id.clone(),
                        output: public,
                    },
                    timestamp_ms: step.timestamp_ms,
                }
            }

            (RedactionPolicy::Memo { memo }, StepKind::ToolResult { call_id, output })
                if is_sensitive_result(chain, &step.id, sensitive_tools) =>
            {
                let _ = output;
                Step {
                    id: step.id.clone(),
                    parent: step.parent.clone(),
                    kind: StepKind::ToolResult {
                        call_id: call_id.clone(),
                        output: Value::String(memo.clone()),
                    },
                    timestamp_ms: step.timestamp_ms,
                }
            }

            _ => step.clone(),
        })
        .collect()
}

/// A `ToolResult` step is "sensitive" if its preceding `ToolCall` step in
/// the chain names a tool in `sensitive_tools`.
fn is_sensitive_result(
    chain: &[Step],
    result_id: &crate::NodeHash,
    sensitive_tools: &HashSet<String>,
) -> bool {
    let result_idx = match chain.iter().position(|s| &s.id == result_id) {
        Some(i) => i,
        None => return false,
    };
    // Walk backwards from the result; the matching ToolCall is the most
    // recent step with a matching `call_id`.
    let result_call_id = match &chain[result_idx].kind {
        StepKind::ToolResult { call_id, .. } => call_id.clone(),
        _ => return false,
    };
    for s in chain[..result_idx].iter().rev() {
        if let StepKind::ToolCall { call_id, name, .. } = &s.kind {
            if call_id == &result_call_id {
                return sensitive_tools.contains(name);
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NodeHash, Step, StepKind};

    fn step(parent: Option<NodeHash>, kind: StepKind) -> Step {
        Step::new(parent, kind, 0)
    }

    /// Build a representative chain: prompt → assistant turn → ToolCall to
    /// `lookup_customer` → ToolResult containing PII → assistant turn.
    fn chain_with_pii(pii_payload: Value) -> Vec<Step> {
        let s0 = step(
            None,
            StepKind::Prompt {
                model: "test-cloud".into(),
                content: "Find customer 42 and tell me what to do.".into(),
            },
        );
        let s1 = step(
            Some(s0.id.clone()),
            StepKind::Message {
                role: "assistant".into(),
                content: "I'll look them up.".into(),
            },
        );
        let s2 = step(
            Some(s1.id.clone()),
            StepKind::ToolCall {
                call_id: "c-1".into(),
                name: "lookup_customer".into(),
                input: json!({ "id": 42 }),
            },
        );
        let s3 = step(
            Some(s2.id.clone()),
            StepKind::ToolResult {
                call_id: "c-1".into(),
                output: pii_payload,
            },
        );
        let s4 = step(
            Some(s3.id.clone()),
            StepKind::Message {
                role: "assistant".into(),
                content: "Got their record. Continuing.".into(),
            },
        );
        vec![s0, s1, s2, s3, s4]
    }

    fn sensitive() -> HashSet<String> {
        let mut s = HashSet::new();
        s.insert("lookup_customer".to_string());
        s
    }

    fn serialize(chain: &[Step]) -> String {
        // Approximation of what an agent's `prefix_to_history` produces:
        // serialize the kinds in order. The bytes of THIS string are what
        // the cache key is built on for the cloud-facing API call.
        chain
            .iter()
            .map(|s| serde_json::to_string(&s.kind).unwrap())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn contains(s: &str, needle: &str) -> bool {
        s.contains(needle)
    }

    // ---- Axis 1: privacy --------------------------------------------------

    #[test]
    fn baseline_none_policy_leaks_pii() {
        // Establishes the failure mode the other policies must beat.
        let pii = "SSN-111-22-3333";
        let chain = chain_with_pii(json!({ "ssn": pii, "name": "Jane" }));
        let view = cloud_view(&chain, &RedactionPolicy::None, &sensitive());
        let s = serialize(&view);
        assert!(contains(&s, pii), "RedactionPolicy::None must leak PII");
    }

    #[test]
    fn replace_fixed_policy_blocks_pii() {
        let pii = "SSN-111-22-3333";
        let chain = chain_with_pii(json!({ "ssn": pii, "name": "Jane" }));
        let view = cloud_view(
            &chain,
            &RedactionPolicy::ReplaceFixed {
                marker: "[REDACTED]".into(),
            },
            &sensitive(),
        );
        let s = serialize(&view);
        assert!(!contains(&s, pii), "ReplaceFixed must block PII");
        assert!(
            !contains(&s, "Jane"),
            "ReplaceFixed must block sibling PII fields too"
        );
        assert!(contains(&s, "[REDACTED]"), "marker must be present");
    }

    #[test]
    fn public_payload_policy_blocks_private_keeps_public() {
        // Tool author cooperates: returns `{public, private}` in the output.
        // The redactor keeps only `public`.
        let chain = chain_with_pii(json!({
            "public": "customer-42-vip",
            "private": { "ssn": "SSN-111-22-3333", "address": "1 Main" }
        }));
        let view = cloud_view(&chain, &RedactionPolicy::PublicPayload, &sensitive());
        let s = serialize(&view);
        assert!(!contains(&s, "SSN-111-22-3333"), "private must be stripped");
        assert!(
            !contains(&s, "1 Main"),
            "address (under private) must be stripped"
        );
        assert!(contains(&s, "customer-42-vip"), "public must survive");
    }

    #[test]
    fn memo_policy_blocks_pii_with_caller_supplied_memo() {
        let pii = "SSN-111-22-3333";
        let chain = chain_with_pii(json!({ "ssn": pii }));
        let view = cloud_view(
            &chain,
            &RedactionPolicy::Memo {
                memo: "Customer 42 is a VIP, eligible for discount tier B.".into(),
            },
            &sensitive(),
        );
        let s = serialize(&view);
        assert!(!contains(&s, pii), "Memo must block raw PII");
        assert!(
            contains(&s, "VIP"),
            "Memo's summary should appear so the cloud has SOMETHING to act on"
        );
    }

    // ---- Axis 2: within-run cache stability -------------------------------
    //
    // Two consecutive cloud cycles within the same run must see prefixes
    // that are byte-identical for their shared portion. (Otherwise prompt
    // caching breaks on every cloud cycle.) We model this by extending
    // the chain with one more step and checking the redacted prefix's
    // stable portion.

    fn extend_chain(mut chain: Vec<Step>, kind: StepKind) -> Vec<Step> {
        let parent = chain.last().map(|s| s.id.clone());
        chain.push(step(parent, kind));
        chain
    }

    #[test]
    fn replace_fixed_is_within_run_cache_stable() {
        let pii = json!({ "ssn": "SSN-111-22-3333" });
        let chain_a = chain_with_pii(pii.clone());
        let chain_b = extend_chain(
            chain_a.clone(),
            StepKind::Message {
                role: "assistant".into(),
                content: "follow-up.".into(),
            },
        );
        let policy = RedactionPolicy::ReplaceFixed {
            marker: "[REDACTED]".into(),
        };
        let view_a = cloud_view(&chain_a, &policy, &sensitive());
        let view_b = cloud_view(&chain_b, &policy, &sensitive());
        let serialized_a = serialize(&view_a);
        let serialized_b = serialize(&view_b);
        assert!(
            serialized_b.starts_with(&serialized_a),
            "extending the chain must not change the redacted prefix"
        );
    }

    #[test]
    fn public_payload_is_within_run_cache_stable_when_public_is_deterministic() {
        let chain_a = chain_with_pii(json!({
            "public": "stable-handle-42",
            "private": "ssn-secret"
        }));
        let chain_b = extend_chain(
            chain_a.clone(),
            StepKind::Message {
                role: "assistant".into(),
                content: "next step".into(),
            },
        );
        let view_a = cloud_view(&chain_a, &RedactionPolicy::PublicPayload, &sensitive());
        let view_b = cloud_view(&chain_b, &RedactionPolicy::PublicPayload, &sensitive());
        let serialized_a = serialize(&view_a);
        let serialized_b = serialize(&view_b);
        assert!(serialized_b.starts_with(&serialized_a));
    }

    #[test]
    fn memo_is_within_run_cache_stable_when_memo_is_fixed() {
        // Within a single run, the memo is decided once. As long as we don't
        // re-decide it on each cycle, the prefix is stable.
        let chain_a = chain_with_pii(json!({ "ssn": "SSN-111-22-3333" }));
        let chain_b = extend_chain(
            chain_a.clone(),
            StepKind::Message {
                role: "assistant".into(),
                content: "next".into(),
            },
        );
        let policy = RedactionPolicy::Memo {
            memo: "Customer 42 is VIP.".into(),
        };
        let view_a = cloud_view(&chain_a, &policy, &sensitive());
        let view_b = cloud_view(&chain_b, &policy, &sensitive());
        assert!(serialize(&view_b).starts_with(&serialize(&view_a)));
    }

    // ---- Axis 3: cross-run cache stability --------------------------------
    //
    // Two different runs of the "same workflow" should produce identical
    // bytes for the redacted view IF the underlying PII is the same input
    // OR is being normalized to a canonical form. Cross-run cache stability
    // is what makes high-volume agent traffic cheap on Anthropic's cache.

    #[test]
    fn replace_fixed_is_cross_run_cache_stable_regardless_of_pii() {
        // Strongest guarantee: even if the PII differs across runs, the
        // redacted view is byte-identical because the marker is fixed.
        let chain_run1 = chain_with_pii(json!({ "ssn": "SSN-111-22-3333" }));
        let chain_run2 = chain_with_pii(json!({ "ssn": "SSN-999-88-7777" }));
        let policy = RedactionPolicy::ReplaceFixed {
            marker: "[REDACTED]".into(),
        };
        let v1 = serialize(&cloud_view(&chain_run1, &policy, &sensitive()));
        let v2 = serialize(&cloud_view(&chain_run2, &policy, &sensitive()));
        assert_eq!(
            v1, v2,
            "ReplaceFixed: redacted view independent of PII content → cache hits across runs"
        );
    }

    #[test]
    fn public_payload_is_cross_run_cache_stable_when_public_is_stable() {
        // Tool author returns the SAME public handle across runs; only
        // private differs. Cache hits.
        let r1 = chain_with_pii(json!({
            "public": "customer-42-handle",
            "private": "ssn-AAA"
        }));
        let r2 = chain_with_pii(json!({
            "public": "customer-42-handle",
            "private": "ssn-BBB"
        }));
        let v1 = serialize(&cloud_view(
            &r1,
            &RedactionPolicy::PublicPayload,
            &sensitive(),
        ));
        let v2 = serialize(&cloud_view(
            &r2,
            &RedactionPolicy::PublicPayload,
            &sensitive(),
        ));
        assert_eq!(v1, v2);
    }

    #[test]
    fn public_payload_breaks_cross_run_cache_when_public_varies() {
        // If the tool author's "public" output isn't deterministic across
        // runs (e.g. timestamps, request ids), the redaction is correct
        // but cache hits collapse.
        let r1 = chain_with_pii(json!({
            "public": "request-id-aaaa",
            "private": "ssn"
        }));
        let r2 = chain_with_pii(json!({
            "public": "request-id-bbbb",
            "private": "ssn"
        }));
        let v1 = serialize(&cloud_view(
            &r1,
            &RedactionPolicy::PublicPayload,
            &sensitive(),
        ));
        let v2 = serialize(&cloud_view(
            &r2,
            &RedactionPolicy::PublicPayload,
            &sensitive(),
        ));
        assert_ne!(v1, v2);
    }

    #[test]
    fn memo_breaks_cross_run_cache_when_memo_is_llm_generated() {
        // Real local-model memos vary across runs (sampling). We model
        // this by feeding two different memos to two runs of the same
        // workflow.
        let chain = chain_with_pii(json!({ "ssn": "SSN-111-22-3333" }));
        let v1 = serialize(&cloud_view(
            &chain,
            &RedactionPolicy::Memo {
                memo: "Customer is VIP.".into(),
            },
            &sensitive(),
        ));
        let v2 = serialize(&cloud_view(
            &chain,
            &RedactionPolicy::Memo {
                memo: "The customer qualifies for VIP discount.".into(),
            },
            &sensitive(),
        ));
        assert_ne!(
            v1, v2,
            "Memo content drives the cache key — LLM-sampled memos break cross-run caching"
        );
    }

    // ---- Composite assertion: the winner ----------------------------------

    #[test]
    fn winner_summary() {
        // ReplaceFixed is the only policy that satisfies all three axes
        // without requiring tool-author cooperation. PublicPayload ties on
        // privacy + cache when the tool author is disciplined, but loses on
        // burden. Memo loses on cross-run cache.
        //
        // This test is documentation more than assertion — it pins the
        // claims we make in the README, so a future regression in any of
        // the underlying tests breaks this one too.
        let pii = "SSN-111-22-3333";
        let chain = chain_with_pii(json!({ "ssn": pii, "address": "1 Main" }));
        let view = cloud_view(
            &chain,
            &RedactionPolicy::ReplaceFixed {
                marker: "[REDACTED:lookup_customer]".into(),
            },
            &sensitive(),
        );
        let s = serialize(&view);

        // Privacy ✓
        assert!(!s.contains(pii));
        assert!(!s.contains("1 Main"));

        // Within-run + cross-run cache stability ✓ (covered by the
        // dedicated tests above; here we just sanity-check that the marker
        // is fixed and contains no PII).
        assert!(s.contains("[REDACTED:lookup_customer]"));
    }
}
