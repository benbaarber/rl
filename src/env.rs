/// Represents a Markov decision process, defining the dynamics of an environment
/// in which an agent can operate.
///
/// This base trait represents the common case of a discrete-time MDP with one agent
/// and a finite state space and action space.
pub trait Environment {
    /// A representation of the state of the environment to be passed to an agent
    type State;

    /// A representation of an action that an agent can take to affect the environment
    type Action;

    /// Get the available actions for the current state
    ///
    /// The returned slice should never be empty, instead specify an action that represents doing nothing if necessary.
    fn actions(&self) -> Vec<Self::Action>;

    /// Determine if the state is active or terminal
    fn is_active(&self) -> bool;

    /// Update the environment in response to a an action taken by an agent, producing a new state and associated reward
    ///
    /// **Returns** `(next_state, reward)`
    fn step(&mut self, action: Self::Action) -> (Option<Self::State>, f64);

    /// Reset the environment to an initial state
    ///
    /// **Returns** the state
    fn reset(&mut self) -> Self::State;
}
