use std::collections::BTreeMap;

/// Asserts that a numerical value is in the provided interval `[a,b]` and panics
/// with a helpful message if not
///
/// This will panic with the message "Invalid value for \`value\`. Must be in the interval \[0.0, 1.0\]."
#[macro_export]
macro_rules! assert_interval {
    ($var:expr, $a:expr, $b:expr) => {
        assert!(
            ($a..=$b).contains(&$var),
            "Invalid value for `{}`. Must be in the interval [{}, {}].",
            stringify!($var),
            $a,
            $b,
        );
    };
}

/// Format a float with the given precision. Will use scientific notation if necessary.
pub(crate) fn _format_float(float: f64, precision: usize) -> String {
    let scientific_notation_threshold = 0.1_f64.powf(precision as f64 - 1.0);

    match scientific_notation_threshold >= float {
        true => format!("{float:.precision$e}"),
        false => format!("{float:.precision$}"),
    }
}

pub(crate) fn summary_from_keys(keys: &[&'static str]) -> BTreeMap<&'static str, f64> {
    keys.iter().map(|k| (*k, 0.0)).collect()
}
