use rand::{seq::SliceRandom, thread_rng};

use crate::{ds::RingBuffer, env::Environment};

/// Represents a single experience or transition in the environment
pub struct Exp<E: Environment> {
    /// The state of the environment before taking the action
    pub state: E::State,
    /// The action taken in the given state
    pub action: E::Action,
    /// The state of the environment after the action is taken, or if terminal, `None`
    pub next_state: Option<E::State>,
    /// The reward received after taking the action
    pub reward: f64,
}

/// A zipped batch of [experiences](Exp)
pub struct ExpBatch<'a, E: Environment> {
    /// The state of the environment before taking the action
    pub states: Vec<&'a E::State>,
    /// The action taken in the given state
    pub actions: Vec<&'a E::Action>,
    /// The state of the environment after the action is taken, or if terminal, `None`
    pub next_states: Vec<Option<&'a E::State>>,
    /// The reward received after taking the action
    pub rewards: Vec<f64>,
}

impl<'a, E: Environment + 'a> ExpBatch<'a, E> {
    /// Construct an `ExpBatch` from an iterator of [experience](Exp) references and a specified batch size
    pub fn from_iter(iter: impl Iterator<Item = &'a Exp<E>>, batch_size: usize) -> Self {
        let batch = Self {
            states: Vec::with_capacity(batch_size),
            actions: Vec::with_capacity(batch_size),
            next_states: Vec::with_capacity(batch_size),
            rewards: Vec::with_capacity(batch_size),
        };

        iter.fold(batch, |mut b, e| {
            b.states.push(&e.state);
            b.actions.push(&e.action);
            b.next_states.push(e.next_state.as_ref());
            b.rewards.push(e.reward);
            b
        })
    }
}

/// Experience replay structs
pub trait Memory {
    type Env: Environment;

    /// Add a new experience to the memory
    fn push(&mut self, exp: Exp<Self::Env>);
    /// Sample a random batch of experiences from the memory
    ///
    /// ### Returns
    /// - `Some(experiences)` if `batch_size` is less than or equal to the buffer length
    /// - `None` otherwise
    ///
    /// **Panics** if `batch_size` is greater than the memory capacity.
    fn sample(&self, batch_size: usize) -> Option<Vec<&Exp<Self::Env>>>;
    /// Sample a random batch of experiences from the memory and zip the vector of tuples into a tuple of vectors
    ///
    /// ### Returns
    /// - `Some(batch)` if `batch_size` is less than or equal to the buffer length
    /// - `None` otherwise
    ///
    /// **Panics** if `batch_size` is greater than the memory capacity.
    fn sample_zipped(&self, batch_size: usize) -> Option<ExpBatch<Self::Env>>;
}

/// A fixed-size memory storage for reinforcement learning experiences
///
/// This structure uses a ring buffer to store experiences, which are tuples of (state, action, next state, reward).
/// It automatically overwrites the oldest experiences once it reaches its capacity.
///
/// ### Type Parameters:
/// - `E`: Environment
///
/// ### Fields:
/// - `memory`: A `RingBuffer` that stores the experiences
pub struct ReplayMemory<E: Environment> {
    memory: RingBuffer<Exp<E>>,
}

impl<E: Environment> Memory for ReplayMemory<E> {
    type Env = E;

    fn push(&mut self, exp: Exp<Self::Env>) {
        self.memory.push(exp);
    }

    fn sample(&self, batch_size: usize) -> Option<Vec<&Exp<Self::Env>>> {
        assert!(
            batch_size <= self.memory.capacity(),
            "`batch_size` must be less than buffer capacity"
        );

        if batch_size <= self.memory.len() {
            Some(
                self.memory
                    .view()
                    .choose_multiple(&mut thread_rng(), batch_size)
                    .collect(),
            )
        } else {
            None
        }
    }

    fn sample_zipped(&self, batch_size: usize) -> Option<ExpBatch<Self::Env>> {
        assert!(
            batch_size <= self.memory.capacity(),
            "`batch_size` must be less than buffer capacity"
        );

        if batch_size <= self.memory.len() {
            let experiences = self
                .memory
                .view()
                .choose_multiple(&mut thread_rng(), batch_size);
            let batch = ExpBatch::from_iter(experiences, batch_size);
            Some(batch)
        } else {
            None
        }
    }
}

impl<E: Environment> ReplayMemory<E> {
    pub fn new(capacity: usize) -> Self {
        Self {
            memory: RingBuffer::<Exp<E>>::new(capacity),
        }
    }

    /// Construct a new `ReplayMemory` from a provided array of experiences
    pub fn from(data: Vec<Exp<E>>) -> Self {
        Self {
            memory: RingBuffer::from(data),
        }
    }
}
