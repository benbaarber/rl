use rand::{thread_rng, Rng};

use crate::{assert_interval, decay::Decay};

use super::Choice;

/// Epsilon greedy exploration policy with time-decaying epsilon threshold
pub struct EpsilonGreedy<D: Decay> {
    epsilon: D,
}

impl<D: Decay> EpsilonGreedy<D> {
    /// Initialize epsilon greedy policy with a decay strategy
    pub fn new(decay: D) -> Self {
        Self { epsilon: decay }
    }

    /// Invoke epsilon greedy policy for current episode
    pub fn choose(&self, episode: u32) -> Choice {
        let epsilon = self.epsilon.evaluate(episode as f32);
        if thread_rng().gen::<f32>() > epsilon {
            Choice::Exploit
        } else {
            Choice::Explore
        }
    }
}
