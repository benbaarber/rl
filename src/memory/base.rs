use rand::{seq::SliceRandom, thread_rng};

use crate::{ds::RingBuffer, env::Environment};

use super::{DynamicExpBatch, Exp, ExpBatch};

// TODO: Split replay memory into two structs: ReplayMemory and DynamicReplayMemory, where the former has a const generic batch size
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
pub struct ReplayMemory<const B: usize, E: Environment> {
    memory: RingBuffer<Exp<E>>,
}

impl<const B: usize, E: Environment> ReplayMemory<B, E> {
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
    pub fn sample(&self) -> Option<[Exp<E>; B]> {
        if B <= self.memory.len() {
            Some(
                self.memory
                    .view()
                    .choose_multiple(&mut thread_rng(), B)
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
    pub fn sample_zipped(&self) -> Option<ExpBatch<E, B>> {
        if B <= self.memory.len() {
            let experiences = self
                .memory
                .view()
                .choose_multiple(&mut thread_rng(), B)
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

#[cfg(test)]
mod tests {
    use super::*;

    struct MockEnv;

    impl Environment for MockEnv {
        type State = i32;
        type Action = i32;

        fn step(&mut self, _action: Self::Action) -> (Option<Self::State>, f32) {
            (None, 0.0)
        }

        fn reset(&mut self) -> Self::State {
            0
        }

        fn random_action() -> Self::Action {
            0
        }
    }

    const MEMORY_CAP: usize = 4;
    const BATCH_SIZE: usize = 2;

    fn create_mock_exp_vec() -> Vec<Exp<MockEnv>> {
        (0..4)
            .into_iter()
            .map(|i| Exp {
                state: i,
                action: i + 1,
                next_state: Some(i + 1),
                reward: 1.0,
            })
            .collect()
    }

    #[test]
    fn replay_memory_functional() {
        let experiences = create_mock_exp_vec();
        let mut memory = ReplayMemory::<BATCH_SIZE, _>::new(MEMORY_CAP);

        assert!(
            memory.sample().is_none(),
            "sample none when too few experiences"
        );
        assert!(
            memory.sample_dyn(BATCH_SIZE).is_none(),
            "sample_dyn none when too few experiences"
        );
        assert!(
            memory.sample_zipped().is_none(),
            "sample_zipped none when too few experiences"
        );
        assert!(
            memory.sample_zipped_dyn(BATCH_SIZE).is_none(),
            "sample_zipped_dyn none when too few experiences"
        );

        for exp in experiences {
            memory.push(exp);
        }

        assert!(
            memory.sample().is_some_and(|b| b.len() == 2),
            "sample works"
        );
        assert!(
            memory.sample_dyn(BATCH_SIZE).is_some_and(|b| b.len() == 2),
            "sample_dyn works"
        );
        assert!(
            memory.sample_zipped().is_some_and(|b| b.states.len() == 2),
            "sample_zipped works"
        );
        assert!(
            memory
                .sample_zipped_dyn(BATCH_SIZE)
                .is_some_and(|b| b.states.len() == 2),
            "sample_zipped_dyn works"
        );
    }
}
