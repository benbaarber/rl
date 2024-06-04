use std::{
    collections::BTreeMap,
    ops::{Deref, DerefMut},
};

use crate::util::summary_from_keys;

/// Represents a Markov decision process, defining the dynamics of an environment
/// in which an agent can operate.
///
/// This base trait represents the common case of a discrete-time MDP with one agent
/// and a finite state space and action space.
pub trait Environment {
    /// A representation of the state of the environment to be passed to an agent
    ///
    /// This should be a relatively simple data type
    ///
    /// ### Trait bounds
    /// - `Clone`: When sampling batches of experiences, cloning is necessary
    type State: Clone;

    /// A representation of an action that an agent can take to affect the environment
    ///
    /// This should be a relatively simple data type
    ///
    /// ### Trait bounds
    /// - `Clone`: When sampling batches of experiences, cloning is necessary
    type Action: Clone;

    // /// Relevant data to be returned after an episode summarizing the agent's performance
    // type Report;

    /// Get the available actions for the current state
    ///
    /// The returned slice should never be empty, instead specify an action that represents doing nothing if necessary.
    fn actions(&self) -> Vec<Self::Action>;

    /// Update the environment in response to a an action taken by an agent, producing a new state and associated reward
    ///
    /// **Returns** `(next_state, reward)`
    fn step(&mut self, action: Self::Action) -> (Option<Self::State>, f32);

    /// Reset the environment to an initial state
    ///
    /// **Returns** the state
    fn reset(&mut self) -> Self::State;

    /// Determine if the environment is in an active or terminal state
    fn is_active(&self) -> bool {
        true
    }
}

/// A format for reporting training results to [viz](crate::viz)
///
/// Functionally a wrapper around a [BTreeMap] such that values are always returned in the same order.
/// Meant to be initialized once and used for the lifetime of an [Environment].
///
/// See examples for implementation
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
