use std::{
    collections::BTreeMap,
    fmt::Debug,
    ops::{Deref, DerefMut},
};

use burn::tensor::{backend::Backend, Tensor, TensorKind};

use crate::util::summary_from_keys;

/// Represents a Markov decision process, defining the dynamics of an environment
/// in which an agent can operate.
///
/// This base trait represents the common case of a discrete-time MDP with one agent.
pub trait Environment {
    /// A representation of the state of the environment to be passed to an agent
    ///
    /// This should be a relatively simple data type
    ///
    /// ### Trait bounds
    /// - `Clone` - When sampling batches of experiences, cloning is necessary
    type State: Clone + Debug;

    /// A representation of an action that an agent can take to affect the environment
    ///
    /// This should be a relatively simple data type
    ///
    /// ### Trait bounds
    /// - `Clone` - When sampling batches of experiences, cloning is necessary
    type Action: Clone + Debug;

    /// Update the environment in response to a an action taken by an agent, producing a new state and associated reward
    ///
    /// **Returns** `(next_state, reward)`
    fn step(&mut self, action: Self::Action) -> (Option<Self::State>, f32);

    /// Reset the environment to an initial state
    ///
    /// **Returns** the state
    fn reset(&mut self) -> Self::State;

    /// Select a random action from the action space
    fn random_action(&self) -> Self::Action;

    /// Determine if the environment is in an active or terminal state
    fn is_active(&self) -> bool {
        true
    }
}

/// An [Environment] with a discrete action space
pub trait DiscreteActionSpace: Environment {
    /// Get the available actions for the current state
    ///
    /// The returned slice should never be empty, instead specify an action that represents doing nothing if necessary.
    fn actions(&self) -> Vec<Self::Action>;
}

/// A trait for converting items to tensors
///
/// Commonly implemented for `Vec<T>` to convert batches of `T` to a tensor of dimension `D`
///
/// See implementations of this for [`CartPole`](crate::gym::CartPole) as an example of how to implement this trait
pub trait ToTensor<B: Backend, const D: usize, K: TensorKind<B>> {
    fn to_tensor(self, device: &B::Device) -> Tensor<B, D, K>;
}

/// A format for reporting training results to [viz](crate::viz)
///
/// Functionally a wrapper around a [BTreeMap] such that values are always returned in the same order.
/// Meant to be initialized once and used for the lifetime of an [Environment].
///
/// See examples for implementation
#[derive(Debug)]
pub struct Report {
    keys: Vec<&'static str>,
    map: BTreeMap<&'static str, f64>,
}

impl Report {
    /// Create a new report format
    pub fn new(mut keys: Vec<&'static str>) -> Self {
        keys.sort_unstable();
        let map = summary_from_keys(&keys);
        Self { keys, map }
    }

    /// Get keys as a slice
    pub fn keys(&self) -> &[&'static str] {
        &self.keys
    }

    /// Take the report by extracting the inner map and leaving a default
    pub fn take(&mut self) -> BTreeMap<&'static str, f64> {
        std::mem::replace(&mut self.map, summary_from_keys(&self.keys))
    }
}

impl Deref for Report {
    type Target = BTreeMap<&'static str, f64>;

    fn deref(&self) -> &Self::Target {
        &self.map
    }
}

impl DerefMut for Report {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.map
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) struct MockEnv;

    impl Environment for MockEnv {
        type State = i32;
        type Action = i32;

        fn step(&mut self, _action: Self::Action) -> (Option<Self::State>, f32) {
            (None, 0.0)
        }

        fn reset(&mut self) -> Self::State {
            0
        }

        fn random_action(&self) -> Self::Action {
            0
        }
    }

    #[test]
    fn report_functional() {
        let mut report = Report::new(vec!["c", "a", "b"]);
        assert_eq!(
            *report.keys(),
            ["a", "b", "c"],
            "Keys were sorted on initialization"
        );

        report.entry("a").and_modify(|x| *x += 1.0);
        assert_eq!(
            *report.get("a").unwrap(),
            1.0,
            "Mutations on entries work and report derefs into inner map"
        );

        let inner_map = report.take();
        assert!(
            inner_map.values().eq([1.0, 0.0, 0.0].iter()),
            "Inner map can be taken with correct values"
        );
        assert!(
            report.values().eq([0.0, 0.0, 0.0].iter()),
            "Taking inner map leaves default values in report"
        );
    }
}
