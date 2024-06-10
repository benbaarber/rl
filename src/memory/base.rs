use log::warn;
use rand::{seq::SliceRandom, thread_rng};

use crate::{ds::RingBuffer, env::Environment};

use super::{DynamicExpBatch, Exp, ExpBatch};

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
    pub fn sample<const S: usize>(&self) -> Option<[Exp<E>; S]> {
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
    pub fn sample_dyn(&self, batch_size: usize) -> Option<Vec<&Exp<E>>> {
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
    pub fn sample_zipped<const S: usize>(&self) -> Option<ExpBatch<E, S>> {
        if S <= self.memory.len() {
            let experiences = self
                .memory
                .view()
                .choose_multiple(&mut thread_rng(), S)
                .cloned();
            let batch = ExpBatch::from_iter(experiences);
            Some(batch)
        } else {
            warn!("Memory length: {}", self.memory.len());
            None
        }
    }

    /// Sample a random batch of experiences from the memory and zip the vector of tuples into a tuple of vectors
    ///
    /// ### Returns
    /// - `Some(batch)` if `batch_size` is less than or equal to the buffer length
    /// - `None` otherwise
    pub fn sample_zipped_dyn(&self, batch_size: usize) -> Option<DynamicExpBatch<E>> {
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
