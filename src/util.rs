/// Asserts that a numerical value is in the provided interval `[a,b]` and panics
/// with a helpful message if not
///
/// ### Example
/// ```
/// let value = 2.0;
/// assert_interval!(value, 0.0, 1.0);
/// ```
/// This will panic with the message "Invalid value for \`value\`. Must be in the interval \[0.0, 1.0\]."
#[macro_export]
macro_rules! assert_interval {
    ($var:expr, $a:expr, $b:expr) => {
        assert!(
            $var >= $a && $var <= $b,
            "Invalid value for `{}`. Must be in the interval [{}, {}].",
            stringify!($var),
            $a,
            $b,
        );
    };
}
