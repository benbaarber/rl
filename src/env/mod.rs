use std::hash::Hash;

/// Represents a Markov decision process, defining the dynamics of an environment
/// in which an agent can operate.
///
/// This base trait represents the common case of a discrete-time MDP with one agent
/// and a finite state space and action space.
pub trait Environment {
    /// A representation of the state of the environment to be passed to an agent
    type State: Clone;

    /// A representation of an action that an agent can take to affect the environment
    type Action: Clone;

    /// Get the available actions for the current state
    ///
    /// The returned slice should never be empty, instead specify an action that represents doing nothing if necessary.
    fn actions(&self) -> &[Self::Action];

    /// Determine if the state is active or terminal
    fn is_active(&self) -> bool;

    /// Compute the reward associated with transitioning from one state to another due to an action
    fn reward(&self, state: Self::State, action: Self::Action, next_state: Self::State) -> f64;

    /// Update the environment in response to a an action taken by an agent, producing a new state and associated reward
    ///
    /// **Returns** `(next_state, reward)`
    fn step(&mut self, action: Self::Action) -> (Self::State, f64);

    /// Reset the environment to an initial state
    ///
    /// **Returns** the state
    fn reset(&mut self) -> Self::State;
}

// Tried an approach with additional associated type bounds using `Environment` as a
// supertrait, but this feature is not implemented in the language yet, waiting on
// RFC 2289.
/// Represents a Markov decision process where every state-action pair can be a key in a hashmap, defining
/// the dynamics of a simple environment in which an agent can operate.
///
/// This environment trait should be used with state spaces and action spaces that
/// are *small* and *primitive*. It is designed for use with a simple Q table, so the
/// state and action types must be `Copy + Eq + Hash`.
pub trait TableEnvironment {
    /// A representation of the state of the environment to be passed to an agent
    type State: Copy + Eq + Hash;

    /// A representation of an action that an agent can take to affect the environment
    type Action: Copy + Eq + Hash;

    /// Get the available actions for the current state
    ///
    /// The returned slice should never be empty, instead specify an action that represents doing nothing if necessary.
    fn actions(&self) -> &[Self::Action];

    /// Determine if the state is active or terminal
    fn is_active(&self) -> bool;

    /// Compute the reward associated with transitioning from one state to another due to an action
    fn reward(&self, state: Self::State, action: Self::Action, next_state: Self::State) -> f64;

    /// Update the environment in response to a an action taken by an agent, producing a new state and associated reward
    ///
    /// **Returns** `(next_state, reward)`
    fn step(&mut self, action: Self::Action) -> (Self::State, f64);

    /// Reset the environment to an initial state
    ///
    /// **Returns** the state
    fn reset(&mut self) -> Self::State;
}
