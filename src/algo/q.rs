use rand::{seq::SliceRandom, thread_rng};
use std::{collections::HashMap, hash::Hash};

use crate::{
    assert_interval, decay,
    env::Environment,
    exploration::{Choice, EpsilonGreedy},
    memory::Exp,
};

pub trait QAgent {
    type Env: Environment;

    /// Perform an action given the current state and available actions
    fn act(
        &self,
        state: <Self::Env as Environment>::State,
        actions: &[<Self::Env as Environment>::Action],
    ) -> <Self::Env as Environment>::Action;

    /// Update the agent's policy after an [Experience]
    fn learn(
        &mut self,
        experience: Exp<Self::Env>,
        next_actions: &[<Self::Env as Environment>::Action],
    );

    /// Deploy agent into the environment for one episode
    fn go(&mut self);
}

/// A simple Q-learning agent that utilizes a Q-table to learn its environment
pub struct QTableAgent<E: Environment>
where
    E::State: Copy + Eq + Hash,
    E::Action: Copy + Eq + Hash,
{
    q_table: HashMap<(E::State, E::Action), f32>,
    alpha: f32, // learning rate
    gamma: f32, // discount factor
    exploration: EpsilonGreedy<decay::Exponential>,
    episode: u32, // current episode
}

impl<E: Environment> QTableAgent<E>
where
    E::State: Copy + Eq + Hash,
    E::Action: Copy + Eq + Hash,
{
    /// Initialize a new `QAgent` in a given environment
    ///
    /// ### Parameters
    /// - `alpha`: The learning rate - must be between 0 and 1
    /// - `gamma`: The discount factor - must be between 0 and 1
    /// - `exploration`: A customized [EpsilonGreedy] policy
    ///
    /// **Panics** if `alpha` or `gamma` is not in the interval `[0,1]`
    pub fn new(alpha: f32, gamma: f32, exploration: EpsilonGreedy<decay::Exponential>) -> Self {
        assert_interval!(alpha, 0.0, 1.0);
        assert_interval!(gamma, 0.0, 1.0);
        Self {
            q_table: HashMap::new(),
            alpha,
            gamma,
            exploration,
            episode: 0,
        }
    }

    pub fn get_q_table(&self) -> &HashMap<(E::State, E::Action), f32> {
        &self.q_table
    }
}

impl<E: Environment> QTableAgent<E>
where
    E::State: Copy + Eq + Hash,
    E::Action: Copy + Eq + Hash,
{
    fn act(&self, state: E::State, actions: &[E::Action]) -> E::Action {
        let random = || actions.choose(&mut thread_rng()).unwrap();
        *match self.exploration.choose(self.episode) {
            Choice::Explore => random(),
            Choice::Exploit => actions
                .iter()
                .max_by(|&a, &b| {
                    let a_value = *self.q_table.get(&(state, *a)).unwrap_or(&0.0);
                    let b_value = *self.q_table.get(&(state, *b)).unwrap_or(&0.0);
                    a_value.partial_cmp(&b_value).unwrap()
                })
                .unwrap_or_else(random),
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
            let action = self.act(state, &actions);
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
    }
}
