use rand::{seq::SliceRandom, thread_rng};

use crate::{ds::RingBuffer, env::Environment};

/// Represents a single experience or transition in the environment
///
/// **Fields:**
/// - `.0` (state): The state of the environment before taking the action
/// - `.1` (action): The action taken in the given state
/// - `.2` (next state): The state of the environment after the action is taken
/// - `.3` (reward): The reward received after taking the action
pub struct Experience<E: Environment>(pub E::State, pub E::Action, pub E::State, pub f64);

impl<E: Environment> Clone for Experience<E> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), self.1.clone(), self.2.clone(), self.3)
    }
}

/// Experience replay structs
pub trait Memory {
    type Env: Environment;

    /// Add a new experience to the memory
    fn push(&mut self, exp: Experience<Self::Env>);
    /// Sample a random batch of experiences from the memory
    ///
    /// **Panics** if `batch_size` is greater than the memory capacity.
    fn sample(&self, batch_size: usize) -> Vec<Experience<Self::Env>>;
}

/// A fixed-size memory storage for reinforcement learning experiences
///
/// This structure uses a ring buffer to store experiences, which are tuples of (state, action, next state, reward).
/// It automatically overwrites the oldest experiences once it reaches its capacity.
///
/// **Type Parameters:**
/// - `S`: Represents the type of the states in the environment
/// - `A`: Represents the type of the actions
/// - `CAP`: The maximum number of experiences the memory can hold, specified at compile time
///
/// **Fields:**
/// - `memory`: A `RingBuffer` that stores the experiences
pub struct ReplayMemory<E: Environment> {
    memory: RingBuffer<Experience<E>>,
}

impl<E: Environment> Memory for ReplayMemory<E> {
    type Env = E;

    fn push(&mut self, exp: Experience<Self::Env>) {
        self.memory.push(exp);
    }

    fn sample(&self, batch_size: usize) -> Vec<Experience<Self::Env>> {
        assert!(
            batch_size <= self.memory.len(),
            "`batch_size` must be less than buffer capacity"
        );
        let mut rng = thread_rng();
        self.memory
            .view()
            .choose_multiple(&mut rng, batch_size)
            .cloned()
            .collect()
    }
}

impl<E: Environment> ReplayMemory<E> {
    /// Construct a new `ReplayMemory` from a provided array of experiences
    pub fn from(data: Vec<Experience<E>>) -> Self {
        Self {
            memory: RingBuffer::from(data),
        }
    }
}
