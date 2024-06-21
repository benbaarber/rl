use rand::{
    distributions::{Distribution, Uniform},
    thread_rng, Rng,
};

use crate::{
    decay::{self, Decay},
    ds::{RingBuffer, SumTree},
    env::Environment,
};

use super::{Exp, ExpBatch};

/// A prioritized replay memory, as described in [this paper](https://arxiv.org/abs/1511.05952)
///
/// An improvement over the base replay memory, this implementation prioritizes "surprising" or "valuable" experiences,
/// where the amount of surprise is approximated by the temporal difference error
pub struct PrioritizedReplayMemory<E: Environment> {
    memory: RingBuffer<Exp<E>>,
    priorities: SumTree,
    batch_size: usize,
    alpha: f32,
    beta: decay::Linear,
}

impl<E: Environment> PrioritizedReplayMemory<E> {
    pub fn new(
        capacity: usize,
        batch_size: usize,
        alpha: f32,
        beta_0: f32,
        num_episodes: usize,
    ) -> Self {
        Self {
            memory: RingBuffer::new(capacity),
            priorities: SumTree::new(capacity),
            batch_size,
            alpha,
            beta: decay::Linear::new((1.0 - beta_0) / num_episodes as f32, beta_0, 1.0).unwrap(),
        }
    }
    /// Add a new experience to the memory
    pub fn push(&mut self, exp: Exp<E>) {
        let ix = self.memory.push(exp);
        let max_priority = self.priorities.max();
        self.priorities.update(ix, max_priority);
    }

    /// Compute the importance sampling weights for each experience's probability
    fn compute_weights(&self, episode: usize, probs: Vec<f32>) -> Vec<f32> {
        let beta = self.beta.evaluate(episode as f32);
        let n = self.memory.len() as f32;

        let weights = probs.into_iter().map(|p| (n * p).powf(-beta));
        let w_max = weights.clone().reduce(f32::max).unwrap();
        weights.map(|w| w / w_max).collect()
    }

    /// Sample a random batch of prioritized experiences from the memory and compute the IS weights for each
    ///
    /// ### Returns
    /// - `None` if there are less experiences stored than can fill a batch
    /// - `Some((batch, weights, indices))` otherwise
    ///   - `batch`: the sampled experiences
    ///   - `weights`: the importance sampling weights
    ///   - `indices`: the indices of the sampled experiences - hold on to this and pass it back to the
    ///     [`update_priorities`](PrioritizedReplayMemory::update_priorities) function along with the computed
    ///     TD errors
    pub fn sample(&self, episode: usize) -> Option<(Vec<Exp<E>>, Vec<f32>, Vec<usize>)> {
        if self.batch_size > self.memory.len() {
            return None;
        }

        let total_priority = self.priorities.sum();

        let mut rng = thread_rng();
        let dist = Uniform::new(0.0, total_priority);

        let mut batch = Vec::with_capacity(self.batch_size);
        let mut probs = Vec::with_capacity(self.batch_size);
        let mut indices = Vec::with_capacity(self.batch_size);
        for _ in 0..self.batch_size {
            let priority = dist.sample(&mut rng);
            let (ix, val) = self.priorities.find(priority);
            batch.push(self.memory[ix].clone());
            probs.push(val / total_priority);
            indices.push(ix);
        }

        let weights = self.compute_weights(episode, probs);

        Some((batch, weights, indices))
    }

    /// Sample a random batch of prioritized experiences,
    /// zip the vector of tuples into a tuple of vectors,
    /// and compute the IS weights for each
    ///
    /// ### Returns
    /// - `None` if there are less experiences stored than can fill a batch
    /// - `Some((batch, weights, indices))` otherwise
    ///   - `batch`: the sampled experiences
    ///   - `weights`: the importance sampling weights
    ///   - `indices`: the indices of the sampled experiences - hold on to this and pass it back to the
    ///     [`update_priorities`](PrioritizedReplayMemory::update_priorities) function along with the computed
    ///     TD errors
    pub fn sample_zipped(&self, episode: usize) -> Option<(ExpBatch<E>, Vec<f32>, Vec<usize>)> {
        let (experiences, weights, indices) = self.sample(episode)?;
        let batch = ExpBatch::from_iter(experiences, self.batch_size);
        Some((batch, weights, indices))
    }

    /// Update the priorities of the sampled experiences
    ///
    /// **Panics** if `indices` and `td_errors` do not have the same length
    pub fn update_priorities(&mut self, indices: &[usize], td_errors: &[f32]) {
        assert_eq!(
            indices.len(),
            td_errors.len(),
            "`incides` and `td_errors` are the same length"
        );

        for (ix, tde) in indices.iter().zip(td_errors.iter()) {
            self.priorities.update(*ix, tde.abs().powf(self.alpha))
        }
    }
}
