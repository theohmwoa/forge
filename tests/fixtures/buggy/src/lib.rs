//! Fixture used by the agentic-debugging demo.
//!
//! There is exactly ONE planted bug in this file. The unit tests below
//! exercise it. A correct fix is detectable by `cargo test` succeeding;
//! a wrong fix either breaks compilation or leaves at least one test
//! failing. The agent's job is to find and fix it.

/// Returns the integer product of `a` and `b`.
pub fn multiply(a: i64, b: i64) -> i64 {
    // PLANTED BUG: this should compute the product but does the sum.
    // A correct fix changes `a + b` to `a * b`. Any other change either
    // breaks `divide`/`sum` (the other tests below) or doesn't restore
    // multiply's semantics.
    a + b
}

/// Returns the integer sum of all elements in `xs`.
pub fn sum(xs: &[i64]) -> i64 {
    let mut total = 0;
    for x in xs {
        total += x;
    }
    total
}

/// Integer division. Panics on division by zero.
pub fn divide(a: i64, b: i64) -> i64 {
    if b == 0 {
        panic!("divide: division by zero");
    }
    a / b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiply_basics() {
        // These three assertions can ONLY all hold if multiply does
        // multiplication. A naive fix that replaces the body with `a * b`
        // satisfies all three; any other guess (a-b, a+b, a, b, a*2, etc.)
        // fails at least one.
        assert_eq!(multiply(2, 3), 6);
        assert_eq!(multiply(7, 8), 56);
        assert_eq!(multiply(0, 5), 0);
        assert_eq!(multiply(-4, 3), -12);
    }

    #[test]
    fn sum_basics() {
        assert_eq!(sum(&[1, 2, 3]), 6);
        assert_eq!(sum(&[]), 0);
        assert_eq!(sum(&[-5, 5]), 0);
    }

    #[test]
    fn divide_basics() {
        assert_eq!(divide(10, 2), 5);
        assert_eq!(divide(7, 3), 2);
    }
}
