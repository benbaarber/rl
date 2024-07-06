use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::env::{DiscreteActionSpace, Environment, Report};

/// The K-armed bandit problem is a simple environment with 1 state and K actions. Each action has a reward
/// that is sampled from a normal distribution with a standard deviation of 1. The means of the reward distributions are sampled
/// from another normal distribution upon initialization. The goal is to find the action with the highest expected reward.
///
/// The environment can be stationary or non-stationary. In the stationary case, the reward
/// distribution of each action remains constant throughout the episode. In the non-stationary
/// case, the reward distribution of each action changes slightly after each step.
pub struct KArmedBandit<const K: usize> {
    arms: [Normal<f32>; K],
    steps: usize,
    step_limit: usize,
    is_stationary: bool,
    rewards: Vec<f32>,
    pub report: Report,
}

impl<const K: usize> KArmedBandit<K> {
    /// Initialize a new K-armed bandit environment
    ///
    /// ### Arguments
    /// - `step_limit` - The number of steps before the episode is terminated
    /// - `stationary` - Whether the environment is stationary or not
    pub fn new(step_limit: usize, stationary: bool) -> Self {
        Self {
            arms: generate_arms(),
            steps: 0,
            step_limit,
            is_stationary: stationary,
            rewards: Vec::with_capacity(step_limit),
            report: Report::new(vec!["reward"]),
        }
    }

    pub fn take_rewards(&mut self) -> Vec<f32> {
        std::mem::replace(&mut self.rewards, Vec::with_capacity(1000))
    }
}

impl<const K: usize> Environment for KArmedBandit<K> {
    type State = ();
    type Action = usize;

    fn step(&mut self, action: Self::Action) -> (Option<Self::State>, f32) {
        assert!(action < K, "Invalid action: {}", action);
        let reward = self.arms[action].sample(&mut rand::thread_rng());
        self.report
            .entry("reward")
            .and_modify(|x| *x += reward as f64);
        self.steps += 1;
        self.rewards.push(reward);

        if !self.is_stationary {
            let mut rng = rand::thread_rng();
            let dist = Normal::<f32>::new(0.0, 0.01).unwrap();
            self.arms = self.arms.map(|arm| {
                let mean = arm.mean() + dist.sample(&mut rng);
                Normal::new(mean, 1.0).unwrap()
            });
        }

        let next_state = if self.steps < self.step_limit {
            Some(())
        } else {
            None
        };

        (next_state, reward)
    }

    fn reset(&mut self) -> Self::State {
        self.steps = 0;
        self.arms = generate_arms();
        self.rewards.clear();

        ()
    }

    fn random_action(&self) -> Self::Action {
        rand::thread_rng().gen_range(0..K)
    }
}

impl<const K: usize> DiscreteActionSpace for KArmedBandit<K> {
    fn actions(&self) -> Vec<Self::Action> {
        (0..K).collect()
    }
}

fn generate_arms<const K: usize>() -> [Normal<f32>; K] {
    let mut rng = rand::thread_rng();
    let dist = Normal::<f32>::new(0.0, 1.0).unwrap();

    std::array::from_fn(|_| {
        let mean = dist.sample(&mut rng);
        Normal::new(mean, 1.0).unwrap()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k_armed_bandit_functional() {
        let mut env = KArmedBandit::<3>::new(10, true);
        assert_eq!(env.actions(), vec![0, 1, 2], "Actions are correct");

        let action = env.random_action();
        assert!(action < 3, "Random action is valid");

        let reward = env.step(action).1;
        assert!(reward.is_finite(), "Reward is finite");

        for _ in 0..9 {
            env.step(env.random_action());
        }

        assert_eq!(
            env.step(env.random_action()).0,
            None,
            "Step limit is reached"
        );

        let state = env.reset();
        assert_eq!(state, (), "Reset returns unit");
    }
}
