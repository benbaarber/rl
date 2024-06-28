use rand::{
    distributions::{Distribution, Uniform},
    thread_rng,
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
/// where the amount of surprise is approximated by the temporal difference error.
///
/// ### Hyperparameters
/// - `alpha` - the prioritization exponent, which affects degree of prioritization used in the stochastic sampling of experiences
///   - A value of `0.0` means no prioritization, as it makes all priorities have a value of 1, yielding a uniform distribution
///   - Higher values mean higher prioritization, and `1.0` is a sensible maximum here, though higher values can be used
/// - `beta_0` - the initial value for beta, the importance sampling exponent, which is annealed from β<sub>0</sub> to 1 to apply
///   IS weights to the temporal difference errors
pub struct PrioritizedReplayMemory<E: Environment> {
    memory: RingBuffer<Exp<E>>,
    priorities: SumTree,
    alpha: f32,
    beta: decay::Linear,
    pub batch_size: usize,
}

impl<E: Environment> PrioritizedReplayMemory<E> {
    /// Initialize a `PrioritizedReplayMemory`
    ///
    /// ### Arguments
    /// - `capacity` - the number of experiences the replay memory can hold before overwriting the oldest ones
    /// - `batch_size` - the number of experiences in a sampled batch
    /// - `alpha` - the prioritization exponent
    ///   - A sensible default is `0.7`
    /// - `beta_0` - the initial value for beta, the importance sampling exponent
    ///   - A sensible default is `0.5`
    /// - `num_episodes` - the number of episodes the associated agent will train for
    ///   - Needed to set up annealing of the beta hyperparameter
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
            alpha,
            beta: decay::Linear::new((beta_0 - 1.0) / num_episodes as f32, beta_0, 1.0).unwrap(),
            batch_size,
        }
    }

    /// Add a new experience to the memory
    pub fn push(&mut self, exp: Exp<E>) {
        let ix = self.memory.push(exp);
        let max_priority = f32::max(self.priorities.max(), 1e-5);
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
    /// ### Arguments
    /// - `episode` - the current episode, used to calculate the current beta value
    ///
    /// ### Returns
    /// - `None` if there are less experiences stored than can fill a batch
    /// - `Some((batch, weights, indices))` otherwise
    ///   - `batch` - the sampled experiences
    ///   - `weights` - the importance sampling weights
    ///   - `indices` - the indices of the sampled experiences - hold on to this and pass it back to the
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
            let ix = self.priorities.find(priority).min(self.memory.len() - 1);
            let val = self.priorities[ix];

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
    /// ### Arguments
    /// - `episode` - the current episode, used to calculate the current beta value
    ///
    /// ### Returns
    /// - `None` if there are less experiences stored than can fill a batch
    /// - `Some((batch, weights, indices))` otherwise
    ///   - `batch` - the sampled experiences
    ///   - `weights` - the importance sampling weights
    ///   - `indices` - the indices of the sampled experiences - hold on to this and pass it back to the
    ///     [`update_priorities`](PrioritizedReplayMemory::update_priorities) function along with the computed
    ///     TD errors
    pub fn sample_zipped(&self, episode: usize) -> Option<(ExpBatch<E>, Vec<f32>, Vec<usize>)> {
        let (experiences, weights, indices) = self.sample(episode)?;
        let batch = ExpBatch::from_iter(experiences, self.batch_size);
        Some((batch, weights, indices))
    }

    /// Update the priorities of the sampled experiences after computing their temporal difference errors
    ///
    /// **Panics** if `indices` and `td_errors` do not have the same length
    ///
    /// ### Arguments
    /// - `indices` - the list of indices to update, returned from calling one of the sample methods
    /// - `td_errors` - the computed temporal difference errors which the new priorities are derived from
    pub fn update_priorities(&mut self, indices: &[usize], td_errors: &[f32]) {
        assert_eq!(
            indices.len(),
            td_errors.len(),
            "`indices` and `td_errors` are the same length"
        );

        for (ix, tde) in indices.iter().zip(td_errors.iter()) {
            self.priorities.update(*ix, tde.abs().powf(self.alpha))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::memory::tests::create_mock_exp_vec;

    use super::*;

    #[test]
    fn prioritized_replay_memory_functional() {
        let experiences = create_mock_exp_vec(8);
        let mut memory = PrioritizedReplayMemory::new(8, 4, 1.0, 0.5, 16);

        assert!(
            memory.sample(0).is_none(),
            "sample none when too few experiences"
        );
        assert!(
            memory.sample_zipped(0).is_none(),
            "sample_zipped none when too few experiences"
        );

        for exp in experiences {
            memory.push(exp);
        }

        assert_eq!(
            memory.priorities.max(),
            1e-5,
            "max priority is minimum value before updates"
        );
        assert_eq!(
            memory.priorities.sum(),
            8e-5,
            "sum is correct after pushing elements"
        );

        let (batch, weights, indices) = memory
            .sample(0)
            .expect("sample some when enough experiences");

        assert_eq!(batch.len(), 4, "batch length correct");
        assert_eq!(weights.len(), 4, "weights length correct");
        assert_eq!(indices.len(), 4, "indices length correct");

        memory.update_priorities(&indices, &[0.1, 0.2, 0.3, 0.4]);

        assert_eq!(
            memory.priorities.max(),
            0.4,
            "max priority is correct after updates"
        );
        assert!(
            memory.priorities.sum() > 0.4,
            "sum is correct after updates"
        );
    }
}
