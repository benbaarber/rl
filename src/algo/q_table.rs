use std::{collections::HashMap, hash::Hash};

use crate::{
    assert_interval, decay,
    env::{DiscreteActionSpace, Environment},
    exploration::{Choice, EpsilonGreedy},
    memory::Exp,
};

/// Configuration for the [`QTableAgent`]
pub struct QTableAgentConfig {
    pub exploration: EpsilonGreedy<decay::Exponential>,
    pub alpha: f32,
    pub gamma: f32,
}

impl Default for QTableAgentConfig {
    fn default() -> Self {
        Self {
            exploration: EpsilonGreedy::new(decay::Exponential::new(0.1, 1.0, 0.01).unwrap()),
            alpha: 0.7,
            gamma: 0.99,
        }
    }
}

/// A simple Q-learning agent that utilizes a Q-table to learn its environment
///
/// ### Generics
/// - `E` - The [`Environment`] in which the agent will learn
///     - The environment's state and action spaces must both be discrete because a Q value will be recorded for each state action pair
///     - For the same reason, the state and action types must be `Copy`, `Eq`, and `Hash` to be used as keys in a [`HashMap`]
pub struct QTableAgent<E>
where
    E: Environment + DiscreteActionSpace,
    E::State: Copy + Eq + Hash,
    E::Action: Copy + Eq + Hash,
{
    q_table: HashMap<(E::State, E::Action), f32>,
    exploration: EpsilonGreedy<decay::Exponential>,
    alpha: f32,   // learning rate
    gamma: f32,   // discount factor
    episode: u32, // current episode
}

impl<E> QTableAgent<E>
where
    E: Environment + DiscreteActionSpace,
    E::State: Copy + Eq + Hash,
    E::Action: Copy + Eq + Hash,
{
    /// Initialize a new `QAgent` in a given environment
    ///
    /// ### Parameters
    /// - `alpha` - The learning rate - must be between 0 and 1
    /// - `gamma` - The discount factor - must be between 0 and 1
    /// - `exploration` - A customized [EpsilonGreedy] policy
    ///
    /// **Panics** if `alpha` or `gamma` is not in the interval `[0,1]`
    pub fn new(config: QTableAgentConfig) -> Self {
        assert_interval!(config.alpha, 0.0, 1.0);
        assert_interval!(config.gamma, 0.0, 1.0);
        Self {
            q_table: HashMap::new(),
            exploration: config.exploration,
            alpha: config.alpha,
            gamma: config.gamma,
            episode: 0,
        }
    }

    pub fn get_q_table(&self) -> &HashMap<(E::State, E::Action), f32> {
        &self.q_table
    }
}

impl<E> QTableAgent<E>
where
    E: Environment + DiscreteActionSpace,
    E::State: Copy + Eq + Hash,
    E::Action: Copy + Eq + Hash,
{
    fn act(&self, env: &E, state: E::State, actions: &[E::Action]) -> E::Action {
        match self.exploration.choose(self.episode) {
            Choice::Explore => env.random_action(),
            Choice::Exploit => *actions
                .iter()
                .max_by(|&a, &b| {
                    let a_value = *self.q_table.get(&(state, *a)).unwrap_or(&0.0);
                    let b_value = *self.q_table.get(&(state, *b)).unwrap_or(&0.0);
                    a_value.partial_cmp(&b_value).unwrap()
                })
                .expect("There is always at least one action available"), // Maybe make this more lenient by providing a default?
        }
    }

    fn learn(&mut self, experience: Exp<E>, next_actions: &[E::Action]) {
        let Exp {
            state,
            action,
            next_state,
            reward,
        } = experience;

        let q_value = *self.q_table.get(&(state, action)).unwrap_or(&0.0);
        let max_next_q = next_actions
            .iter()
            .map(|&a| {
                *next_state
                    .and_then(|s| self.q_table.get(&(s, a)))
                    .unwrap_or(&0.0)
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        let new_q_value = reward + self.gamma * max_next_q;
        let weighted_q_value = (1.0 - self.alpha) * q_value + self.alpha * new_q_value;

        self.q_table.insert((state, action), weighted_q_value);
    }

    pub fn go(&mut self, env: &mut E) {
        let mut next_state = Some(env.reset());
        let mut actions = env.actions();
        while let Some(state) = next_state {
            let action = self.act(env, state, &actions);
            let (next, reward) = env.step(action);
            next_state = next;
            actions = env.actions();

            self.learn(
                Exp {
                    state,
                    action,
                    next_state,
                    reward,
                },
                &actions,
            );
        }

        self.episode += 1;
    }
}
