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

pub fn transpose_iter<T>(v: Vec<Vec<T>>) -> impl Iterator<Item = Vec<T>> {
    assert!(!v.is_empty());
    let len = v[0].len();
    let mut iters = v.into_iter().map(|n| n.into_iter()).collect::<Vec<_>>();
    (0..len).map(move |_| {
        iters
            .iter_mut()
            .map(|n| n.next().unwrap())
            .collect::<Vec<T>>()
    })
}

pub fn transpose<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
    transpose_iter(v).collect()
}
