/// Represents a Markov decision process, defining the dynamics of an environment
/// in which an agent can operate.
///
/// This base trait represents the simplest case of a discrete-time MDP with one agent
/// and a finite state space and action space.
pub trait Environment {
    /// A representation of the state of the environment to be passed to an agent
    type State: Clone;

    /// A representation of an action that an agent can take to affect the environment
    type Action: Clone;

    /// Get the available actions for the current state
    ///
    /// **Returns**
    /// - `Some(actions)`: Active case - a slice of available actions
    /// - `None`: Terminal case
    fn actions(&self) -> Option<&[Self::Action]>;

    /// Compute the reward associated with transitioning from one state to another due to an action
    fn reward(&self, state: Self::State, action: Self::Action, next_state: Self::State) -> f64;

    /// Update the environment in response to a an action taken by an agent,
    /// producing a new state, associated reward, and available actions
    ///
    /// **Returns** `(next_state, reward, actions)`
    ///
    /// The state is terminal if `actions` is `None`
    fn step(&mut self, action: Self::Action) -> (Self::State, f64, Option<&[Self::Action]>);

    /// Reset the environment to an initial state
    ///
    /// **Returns** `(state, actions)`
    fn reset(&mut self) -> (Self::State, &[Self::Action]);
}
