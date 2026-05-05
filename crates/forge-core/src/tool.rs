//! Tools the agent can call. The trait is provider-agnostic; adapter crates
//! map between provider tool schemas (Anthropic / OpenAI / Gemini) and this
//! shape.

use async_trait::async_trait;
use serde_json::{json, Value};

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    /// JSON Schema describing the tool's input.
    fn schema(&self) -> Value;
    async fn run(&self, input: &Value) -> anyhow::Result<Value>;
}

/// A built-in arithmetic tool. Useful for examples and tests.
pub struct Calculator;

#[async_trait]
impl Tool for Calculator {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Performs arithmetic on two numbers. Input: { op, a, b } where op is one of add, sub, mul, div."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "op": { "type": "string", "enum": ["add", "sub", "mul", "div"] },
                "a": { "type": "number" },
                "b": { "type": "number" }
            },
            "required": ["op", "a", "b"]
        })
    }

    async fn run(&self, input: &Value) -> anyhow::Result<Value> {
        let op = input["op"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("calculator: missing op"))?;
        let a = input["a"]
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("calculator: a must be a number"))?;
        let b = input["b"]
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("calculator: b must be a number"))?;
        let result = match op {
            "add" => a + b,
            "sub" => a - b,
            "mul" => a * b,
            "div" if b == 0.0 => anyhow::bail!("calculator: division by zero"),
            "div" => a / b,
            other => anyhow::bail!("calculator: unknown op {other}"),
        };
        Ok(json!(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn calculator_basics() {
        let c = Calculator;
        assert_eq!(
            c.run(&json!({"op": "add", "a": 2, "b": 3})).await.unwrap(),
            json!(5.0)
        );
        assert_eq!(
            c.run(&json!({"op": "mul", "a": 7, "b": 8})).await.unwrap(),
            json!(56.0)
        );
    }

    #[tokio::test]
    async fn calculator_div_by_zero_errors() {
        let err = Calculator
            .run(&json!({"op": "div", "a": 1, "b": 0}))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("division"));
    }
}
