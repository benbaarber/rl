use rand::{seq::SliceRandom, thread_rng};

use crate::{ds::RingBuffer, env::Environment};

use super::{Exp, ExpBatch};

/// A fixed-size memory storage for reinforcement learning experiences
///
/// This structure uses a ring buffer to store experiences, which are tuples of (state, action, next state, reward).
/// It automatically overwrites the oldest experiences once it reaches its capacity.
///
/// ### Type Parameters:
/// - `E` - Environment
///
/// ### Fields:
/// - `memory` - A `RingBuffer` that stores the experiences
#[derive(Debug, Clone)]
pub struct ReplayMemory<E: Environment> {
    memory: RingBuffer<Exp<E>>,
    pub batch_size: usize,
}

impl<E: Environment> ReplayMemory<E> {
    /// Construct a new `ReplayMemory` with a given capacity and batch size
    pub fn new(capacity: usize, batch_size: usize) -> Self {
        Self {
            memory: RingBuffer::<Exp<E>>::new(capacity),
            batch_size,
        }
    }

    /// Add a new experience to the memory
    pub fn push(&mut self, exp: Exp<E>) {
        self.memory.push(exp);
    }

    /// Sample a random batch of experiences from the memory
    ///
    /// ### Returns
    /// - `None` if there are less experiences stored than can fill a batch
    /// - `Some(experiences)` otherwise
    pub fn sample(&self) -> Option<Vec<&Exp<E>>> {
        if self.batch_size <= self.memory.len() {
            Some(
                self.memory
                    .view()
                    .choose_multiple(&mut thread_rng(), self.batch_size)
                    .collect(),
            )
        } else {
            None
        }
    }

    /// Sample a random batch of experiences from the memory and zip the vector of tuples into a tuple of vectors
    ///
    /// ### Returns
    /// - `None` if there are less experiences stored than can fill a batch
    /// - `Some(experiences)` otherwise
    pub fn sample_zipped(&self) -> Option<ExpBatch<E>> {
        if self.batch_size <= self.memory.len() {
            let experiences = self
                .memory
                .view()
                .choose_multiple(&mut thread_rng(), self.batch_size)
                .cloned();
            let batch = ExpBatch::from_iter(experiences, self.batch_size);
            Some(batch)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::memory::tests::create_mock_exp_vec;

    use super::*;

    #[test]
    fn replay_memory_functional() {
        let experiences = create_mock_exp_vec(4);
        let mut memory = ReplayMemory::new(4, 2);

        assert!(
            memory.sample().is_none(),
            "sample none when too few experiences"
        );
        assert!(
            memory.sample_zipped().is_none(),
            "sample_zipped none when too few experiences"
        );

        for exp in experiences {
            memory.push(exp);
        }

        assert!(
            memory.sample().is_some_and(|b| b.len() == 2),
            "sample works"
        );
        assert!(
            memory.sample_zipped().is_some_and(|b| b.states.len() == 2),
            "sample_zipped works"
        );
    }
}
