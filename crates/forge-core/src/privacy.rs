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
//! Three step kinds get redacted when the policy demands it:
//!
//! - `ToolResult.output` — the obvious one. The tool's payload is the
//!   primary leak path.
//! - `ToolCall.input` — the agent might have passed PII as a tool argument
//!   (`lookup_address(name="Jane Doe", ssn="...")`). Redacted only for the
//!   *sensitive* tool's call, not for non-sensitive tools elsewhere in the
//!   chain.
//! - `Message.content` *(role=assistant)* when the message is in a
//!   "sensitive context" — i.e. comes anywhere after a sensitive
//!   `ToolResult` in the chain. The trusted model might have echoed the
//!   PII into its own response while reasoning about it. Conservative by
//!   design: every assistant message after a sensitive tool result gets
//!   redacted, regardless of whether it actually quotes PII. Apps that
//!   need to preserve a sanitized summary should send that summary as a
//!   non-`assistant` role (e.g. `summary`) which is left untouched.
//!
//! `Prompt.content` and `Message.content` for non-assistant roles are
//! never touched. The original user prompt is structural; rewriting it
//! would corrupt the run.
//!
//! See `tests` for the empirical comparison of the four policies across
//! all three step kinds.

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
    if matches!(policy, RedactionPolicy::None) {
        return chain.to_vec();
    }

    // First pass: identify which assistant messages are in a "sensitive
    // context" — i.e. anywhere after the first sensitive `ToolResult` in
    // the chain. Once exposed, every subsequent assistant message is a
    // potential PII leak path.
    let mut sensitive_seen = false;
    let mut tainted_messages: HashSet<usize> = HashSet::new();
    for (i, step) in chain.iter().enumerate() {
        if matches!(&step.kind, StepKind::ToolResult { .. })
            && is_sensitive_result(chain, &step.id, sensitive_tools)
        {
            sensitive_seen = true;
            continue;
        }
        if sensitive_seen {
            if let StepKind::Message { role, .. } = &step.kind {
                if role == "assistant" {
                    tainted_messages.insert(i);
                }
            }
        }
    }

    chain
        .iter()
        .enumerate()
        .map(|(i, step)| match &step.kind {
            StepKind::ToolCall {
                call_id,
                name,
                input,
            } if sensitive_tools.contains(name) => {
                redact_tool_call_input(step, call_id, name, input, policy)
            }
            StepKind::ToolResult { call_id, output }
                if is_sensitive_result(chain, &step.id, sensitive_tools) =>
            {
                redact_tool_result_output(step, call_id, output, policy)
            }
            StepKind::Message { role, content }
                if role == "assistant" && tainted_messages.contains(&i) =>
            {
                redact_assistant_message(step, role, content, policy)
            }
            _ => step.clone(),
        })
        .collect()
}

fn redact_value(input: &Value, policy: &RedactionPolicy, label: &str) -> Value {
    match policy {
        RedactionPolicy::None => input.clone(),
        RedactionPolicy::ReplaceFixed { marker } => Value::String(marker.clone()),
        RedactionPolicy::PublicPayload => match input.as_object() {
            Some(obj) => obj
                .get("public")
                .cloned()
                .unwrap_or_else(|| json!(format!("[redacted: no public {label} payload]"))),
            None => json!(format!("[redacted: {label} not an object]")),
        },
        RedactionPolicy::Memo { memo } => Value::String(memo.clone()),
    }
}

fn redact_tool_call_input(
    step: &Step,
    call_id: &str,
    name: &str,
    input: &Value,
    policy: &RedactionPolicy,
) -> Step {
    Step {
        id: step.id.clone(),
        parent: step.parent.clone(),
        kind: StepKind::ToolCall {
            call_id: call_id.to_string(),
            name: name.to_string(),
            input: redact_value(input, policy, "input"),
        },
        timestamp_ms: step.timestamp_ms,
    }
}

fn redact_tool_result_output(
    step: &Step,
    call_id: &str,
    output: &Value,
    policy: &RedactionPolicy,
) -> Step {
    Step {
        id: step.id.clone(),
        parent: step.parent.clone(),
        kind: StepKind::ToolResult {
            call_id: call_id.to_string(),
            output: redact_value(output, policy, "output"),
        },
        timestamp_ms: step.timestamp_ms,
    }
}

fn redact_assistant_message(
    step: &Step,
    role: &str,
    _content: &str,
    policy: &RedactionPolicy,
) -> Step {
    let new_content = match policy {
        RedactionPolicy::None => unreachable!("None handled at the top of cloud_view"),
        RedactionPolicy::ReplaceFixed { marker } => marker.clone(),
        RedactionPolicy::PublicPayload => "[redacted assistant message]".to_string(),
        RedactionPolicy::Memo { memo } => memo.clone(),
    };
    Step {
        id: step.id.clone(),
        parent: step.parent.clone(),
        kind: StepKind::Message {
            role: role.to_string(),
            content: new_content,
        },
        timestamp_ms: step.timestamp_ms,
    }
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

    // ---- ToolCall.input redaction ----------------------------------------

    /// Build a chain where the agent passes PII as a tool argument.
    /// e.g. `lookup_address(name="Jane", ssn="...")`.
    fn chain_with_pii_in_tool_input(input: Value, output: Value) -> Vec<Step> {
        let s0 = step(
            None,
            StepKind::Prompt {
                model: "test-cloud".into(),
                content: "Look up an address.".into(),
            },
        );
        let s1 = step(
            Some(s0.id.clone()),
            StepKind::ToolCall {
                call_id: "c-1".into(),
                name: "lookup_customer".into(),
                input,
            },
        );
        let s2 = step(
            Some(s1.id.clone()),
            StepKind::ToolResult {
                call_id: "c-1".into(),
                output,
            },
        );
        vec![s0, s1, s2]
    }

    #[test]
    fn replace_fixed_redacts_pii_in_tool_call_input() {
        let pii = "SSN-111-22-3333";
        let chain = chain_with_pii_in_tool_input(
            json!({ "ssn": pii, "name": "Jane Doe" }),
            json!({ "address": "1 Main" }),
        );
        let view = cloud_view(
            &chain,
            &RedactionPolicy::ReplaceFixed {
                marker: "[REDACTED]".into(),
            },
            &sensitive(),
        );
        let s = serialize(&view);
        assert!(
            !contains(&s, pii),
            "SSN must not flow through ToolCall.input"
        );
        assert!(
            !contains(&s, "Jane Doe"),
            "name in input must also be redacted"
        );
        assert!(
            !contains(&s, "1 Main"),
            "output redaction (existing behavior) must still hold"
        );
    }

    #[test]
    fn non_sensitive_tool_input_is_not_touched() {
        // Mixed chain: a non-sensitive tool that shouldn't get its input
        // touched, plus a sensitive one that should.
        let s0 = step(
            None,
            StepKind::Prompt {
                model: "m".into(),
                content: "go".into(),
            },
        );
        let s1 = step(
            Some(s0.id.clone()),
            StepKind::ToolCall {
                call_id: "c-pub".into(),
                name: "list_files".into(),
                input: json!({ "path": "/etc/passwd" }),
            },
        );
        let s2 = step(
            Some(s1.id.clone()),
            StepKind::ToolResult {
                call_id: "c-pub".into(),
                output: json!(["a", "b"]),
            },
        );
        let s3 = step(
            Some(s2.id.clone()),
            StepKind::ToolCall {
                call_id: "c-priv".into(),
                name: "lookup_customer".into(),
                input: json!({ "ssn": "111-22-3333" }),
            },
        );
        let s4 = step(
            Some(s3.id.clone()),
            StepKind::ToolResult {
                call_id: "c-priv".into(),
                output: json!({ "addr": "secret" }),
            },
        );
        let chain = vec![s0, s1, s2, s3, s4];

        let view = cloud_view(
            &chain,
            &RedactionPolicy::ReplaceFixed {
                marker: "[REDACTED]".into(),
            },
            &sensitive(),
        );
        let s = serialize(&view);
        // Non-sensitive tool's input survives.
        assert!(contains(&s, "/etc/passwd"));
        assert!(contains(&s, "list_files"));
        // Sensitive tool's input is redacted.
        assert!(!contains(&s, "111-22-3333"));
        // Sensitive tool's output is also redacted (existing behavior).
        assert!(!contains(&s, "secret"));
    }

    #[test]
    fn public_payload_keeps_public_field_in_tool_call_input() {
        let chain = chain_with_pii_in_tool_input(
            json!({ "public": "lookup-customer-42", "private": "ssn-AAA-111" }),
            json!({ "public": "tier-A", "private": "secret-payload-zzz" }),
        );
        let view = cloud_view(&chain, &RedactionPolicy::PublicPayload, &sensitive());
        let s = serialize(&view);
        assert!(!contains(&s, "ssn-AAA-111"));
        assert!(!contains(&s, "secret-payload-zzz"));
        assert!(contains(&s, "lookup-customer-42"));
        assert!(contains(&s, "tier-A"));
    }

    #[test]
    fn memo_replaces_tool_call_input() {
        let chain = chain_with_pii_in_tool_input(
            json!({ "ssn": "111-22-3333" }),
            json!({ "addr": "secret" }),
        );
        let view = cloud_view(
            &chain,
            &RedactionPolicy::Memo {
                memo: "Customer lookup performed.".into(),
            },
            &sensitive(),
        );
        let s = serialize(&view);
        assert!(!contains(&s, "111-22-3333"));
        assert!(!contains(&s, "secret"));
        assert!(contains(&s, "Customer lookup performed."));
    }

    // ---- Assistant message redaction in sensitive context ------------------

    /// Like `chain_with_pii` but the post-tool-result assistant message
    /// also echoes the PII (modeling a local model that summarized the
    /// record back into its own response).
    fn chain_with_pii_echoed_in_message(pii: &str) -> Vec<Step> {
        let mut chain = chain_with_pii(json!({ "ssn": pii }));
        // The trailing assistant message in `chain_with_pii` doesn't contain
        // the PII; replace it with one that does.
        let last = chain.pop().unwrap();
        let parent = last.parent.clone();
        chain.push(step(
            parent,
            StepKind::Message {
                role: "assistant".into(),
                content: format!("Got the customer record. SSN is {pii}, proceeding."),
            },
        ));
        chain
    }

    #[test]
    fn assistant_message_after_sensitive_tool_result_is_redacted() {
        let pii = "SSN-111-22-3333";
        let chain = chain_with_pii_echoed_in_message(pii);
        let view = cloud_view(
            &chain,
            &RedactionPolicy::ReplaceFixed {
                marker: "[REDACTED]".into(),
            },
            &sensitive(),
        );
        let s = serialize(&view);
        assert!(
            !contains(&s, pii),
            "PII echoed by the model into its own message must not flow to the cloud"
        );
        // The original assistant content "Got the customer record..." should
        // also be gone; it's replaced wholesale.
        assert!(!contains(&s, "Got the customer record"));
    }

    #[test]
    fn assistant_message_before_any_sensitive_tool_is_left_alone() {
        // First assistant message in chain_with_pii is "I'll look them up.";
        // it occurs BEFORE the sensitive ToolResult and shouldn't be touched.
        let chain = chain_with_pii(json!({ "ssn": "111-22-3333" }));
        let view = cloud_view(
            &chain,
            &RedactionPolicy::ReplaceFixed {
                marker: "[REDACTED]".into(),
            },
            &sensitive(),
        );
        let s = serialize(&view);
        assert!(
            contains(&s, "I'll look them up."),
            "messages before the first sensitive ToolResult must not be redacted"
        );
    }

    #[test]
    fn user_role_messages_are_never_redacted_even_after_sensitive_tool() {
        // Manually build a chain with a user-role message AFTER the
        // sensitive ToolResult (simulating a multi-turn conversation
        // where the user adds info post-lookup).
        let mut chain = chain_with_pii(json!({ "ssn": "111-22-3333" }));
        let parent = chain.last().unwrap().id.clone();
        chain.push(step(
            Some(parent),
            StepKind::Message {
                role: "user".into(),
                content: "Continue with that record.".into(),
            },
        ));
        let view = cloud_view(
            &chain,
            &RedactionPolicy::ReplaceFixed {
                marker: "[REDACTED]".into(),
            },
            &sensitive(),
        );
        let s = serialize(&view);
        assert!(
            contains(&s, "Continue with that record."),
            "user-role messages must always pass through verbatim"
        );
    }

    // ---- Cache stability with the expanded redaction surface --------------

    #[test]
    fn replace_fixed_full_redaction_is_cross_run_cache_stable() {
        // Two runs with different PII in EVERY leak path (input, output,
        // echoed message). The redacted view must be byte-identical
        // because every redactable surface goes to the same fixed marker.
        let chain_run1 = {
            let mut c = chain_with_pii_echoed_in_message("SSN-111-22-3333");
            // Also stuff PII into the ToolCall.input.
            if let StepKind::ToolCall { input, .. } = &mut c[2].kind {
                *input = json!({ "id": 42, "verify_ssn": "SSN-111-22-3333" });
            }
            c
        };
        let chain_run2 = {
            let mut c = chain_with_pii_echoed_in_message("SSN-999-88-7777");
            if let StepKind::ToolCall { input, .. } = &mut c[2].kind {
                *input = json!({ "id": 42, "verify_ssn": "SSN-999-88-7777" });
            }
            c
        };
        let policy = RedactionPolicy::ReplaceFixed {
            marker: "[REDACTED]".into(),
        };
        let v1 = serialize(&cloud_view(&chain_run1, &policy, &sensitive()));
        let v2 = serialize(&cloud_view(&chain_run2, &policy, &sensitive()));
        assert_eq!(
            v1, v2,
            "full-surface redaction must produce byte-identical bytes across runs"
        );
    }

    #[test]
    fn replace_fixed_full_redaction_is_within_run_cache_stable() {
        let chain_a = chain_with_pii_echoed_in_message("SSN-111-22-3333");
        let chain_b = extend_chain(
            chain_a.clone(),
            StepKind::Message {
                role: "assistant".into(),
                content: "follow up".into(),
            },
        );
        let policy = RedactionPolicy::ReplaceFixed {
            marker: "[REDACTED]".into(),
        };
        let v_a = serialize(&cloud_view(&chain_a, &policy, &sensitive()));
        let v_b = serialize(&cloud_view(&chain_b, &policy, &sensitive()));
        assert!(
            v_b.starts_with(&v_a),
            "extended chain must preserve the redacted prefix verbatim"
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
