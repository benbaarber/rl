#[macro_export]
macro_rules! assert_probability {
    ($var:expr) => {
        assert!(
            $var >= 0.0 && $var <= 1.0,
            "Value of `{}` is not a valid probability. Must be in the interval [0, 1].",
            stringify!($var)
        );
    };
}
