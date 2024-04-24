use rand::{thread_rng, Rng};

use crate::assert_interval;

/// Marker trait for exploration policies
pub trait Exploration {}

/// Exploration policy result
pub enum Choice {
    Explore,
    Exploit,
}

/// Epsilon greedy exploration policy with time-decaying epsilon threshold
pub struct EpsilonGreedy {
    start: f64,
    end: f64,
    decay: f64,
}

impl Exploration for EpsilonGreedy {}

impl EpsilonGreedy {
    /// Initialize epsilon greedy policy from start, end, and decay rate
    ///
    /// **Panics** if `start`, `end`, or `decay` is not in the interval `[0,1]`, or if `start` is less than `end`
    pub fn new(start: f64, end: f64, decay: f64) -> Self {
        assert_interval!(start, 0.0, 1.0);
        assert_interval!(end, 0.0, 1.0);
        assert_interval!(decay, 0.0, 1.0);
        assert!(
            start > end,
            "Epsilon start value must be greater than end value."
        );
        Self { start, end, decay }
    }

    /// Invoke epsilon greedy policy at time `t` where `t >= 0`
    ///
    /// **Panics** if `t < 0`
    pub fn choose(&self, t: f64) -> Choice {
        assert!(t >= 0.0, "t must be a positive number.");
        let epsilon = self.end + (self.start - self.end) / f64::exp(t / self.decay);
        if thread_rng().gen::<f64>() > epsilon {
            Choice::Exploit
        } else {
            Choice::Explore
        }
    }
}
