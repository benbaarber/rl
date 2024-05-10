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
    pub reward: f32,
}

impl<E: Environment> Clone for Exp<E> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            action: self.action.clone(),
            next_state: self.next_state.clone(),
            reward: self.reward,
        }
    }
}

/// A zipped batch of [experiences](Exp) where the batch size is known at compile time
///
/// The batch size must be passed to the const generic `S`
#[derive(Clone)]
pub struct ExpBatch<E: Environment, const S: usize> {
    /// The state of the environment before taking the action
    pub states: [E::State; S],
    /// The action taken in the given state
    pub actions: [E::Action; S],
    /// The state of the environment after the action is taken, or if terminal, `None`
    pub next_states: [Option<E::State>; S],
    /// The reward received after taking the action
    pub rewards: [f32; S],
}

impl<E: Environment, const S: usize> ExpBatch<E, S> {
    // TODO: try to avoid temporary heap allocation
    /// Construct an `ExpBatch` from an iterator of [experiences](Exp)
    pub fn from_iter(iter: impl Iterator<Item = Exp<E>>) -> Self {
        let batch = DynamicExpBatch::from_iter(iter, S);
        Self {
            states: batch.states.try_into().ok().unwrap(),
            actions: batch.actions.try_into().ok().unwrap(),
            next_states: batch.next_states.try_into().ok().unwrap(),
            rewards: batch.rewards.try_into().ok().unwrap(),
        }
    }
}

/// A zipped batch of [experiences](Exp) where the batch size is not known at compile time
#[derive(Clone)]
pub struct DynamicExpBatch<E: Environment> {
    /// The state of the environment before taking the action
    pub states: Vec<E::State>,
    /// The action taken in the given state
    pub actions: Vec<E::Action>,
    /// The state of the environment after the action is taken, or if terminal, `None`
    pub next_states: Vec<Option<E::State>>,
    /// The reward received after taking the action
    pub rewards: Vec<f32>,
}

impl<E: Environment> DynamicExpBatch<E> {
    /// Construct an `ExpBatch` from an iterator of [experience](Exp) references and a specified batch size
    pub fn from_iter(iter: impl Iterator<Item = Exp<E>>, batch_size: usize) -> Self {
        let batch = Self {
            states: Vec::with_capacity(batch_size),
            actions: Vec::with_capacity(batch_size),
            next_states: Vec::with_capacity(batch_size),
            rewards: Vec::with_capacity(batch_size),
        };

        iter.fold(batch, |mut b, e| {
            b.states.push(e.state.clone());
            b.actions.push(e.action.clone());
            b.next_states.push(e.next_state.clone());
            b.rewards.push(e.reward);
            b
        })
    }
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

    /// Add a new experience to the memory
    pub fn push(&mut self, exp: Exp<E>) {
        self.memory.push(exp);
    }

    /// Sample a random batch of experiences from the memory
    ///
    /// ### Returns
    /// - `Some(experiences)` if `S` is less than or equal to the buffer length
    /// - `None` otherwise
    ///
    /// **Panics** if `S` is greater than the memory capacity.
    pub fn sample<const S: usize>(&self) -> Option<[Exp<E>; S]> {
        assert!(
            S <= self.memory.capacity(),
            "`batch_size` must be less than buffer capacity"
        );

        if S <= self.memory.len() {
            Some(
                self.memory
                    .view()
                    .choose_multiple(&mut thread_rng(), S)
                    .cloned()
                    .collect::<Vec<_>>()
                    .try_into()
                    .ok()
                    .unwrap(),
            )
        } else {
            None
        }
    }

    /// Sample a random batch of experiences from the memory
    ///
    /// ### Returns
    /// - `Some(experiences)` if `batch_size` is less than or equal to the buffer length
    /// - `None` otherwise
    ///
    /// **Panics** if `batch_size` is greater than the memory capacity.
    pub fn sample_dyn(&self, batch_size: usize) -> Option<Vec<&Exp<E>>> {
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

    /// Sample a random batch of experiences from the memory and zip the vector of tuples into a tuple of vectors
    ///
    /// ### Returns
    /// - `Some(batch)` if `S` is less than or equal to the buffer length
    /// - `None` otherwise
    ///
    /// **Panics** if `S` is greater than the memory capacity.
    pub fn sample_zipped<const S: usize>(&self) -> Option<ExpBatch<E, S>> {
        assert!(
            S <= self.memory.capacity(),
            "`batch_size` must be less than buffer capacity"
        );

        if S <= self.memory.len() {
            let experiences = self
                .memory
                .view()
                .choose_multiple(&mut thread_rng(), S)
                .cloned();
            let batch = ExpBatch::from_iter(experiences);
            Some(batch)
        } else {
            None
        }
    }

    /// Sample a random batch of experiences from the memory and zip the vector of tuples into a tuple of vectors
    ///
    /// ### Returns
    /// - `Some(batch)` if `batch_size` is less than or equal to the buffer length
    /// - `None` otherwise
    ///
    /// **Panics** if `batch_size` is greater than the memory capacity.
    pub fn sample_zipped_dyn(&self, batch_size: usize) -> Option<DynamicExpBatch<E>> {
        assert!(
            batch_size <= self.memory.capacity(),
            "`batch_size` must be less than buffer capacity"
        );

        if batch_size <= self.memory.len() {
            let experiences = self
                .memory
                .view()
                .choose_multiple(&mut thread_rng(), batch_size)
                .cloned();
            let batch = DynamicExpBatch::from_iter(experiences, batch_size);
            Some(batch)
        } else {
            None
        }
    }
}
