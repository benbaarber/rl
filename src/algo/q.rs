use rand::{seq::SliceRandom, thread_rng};
use std::{collections::HashMap, hash::Hash};

use crate::{
    assert_interval,
    env::Environment,
    exploration::{Choice, EpsilonGreedy},
    memory::Experience,
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
        experience: Experience<Self::Env>,
        next_actions: &[<Self::Env as Environment>::Action],
    );

    /// Deploy agent into the environment for one episode
    fn go(&mut self);
}

/// A simple Q-learning agent that utilizes a Q-table to learn its environment
pub struct QTableAgent<'a, E: Environment>
where
    E::State: Copy + Eq + Hash,
    E::Action: Copy + Eq + Hash,
{
    env: &'a mut E,
    q_table: HashMap<(E::State, E::Action), f64>,
    alpha: f64, // learning rate
    gamma: f64, // discount factor
    exploration: EpsilonGreedy,
    episode: u32, // current episode
}

impl<'a, E: Environment> QTableAgent<'a, E>
where
    E::State: Copy + Eq + Hash,
    E::Action: Copy + Eq + Hash,
{
    /// Initialize a new `QAgent` in a given environment
    ///
    /// ### Parameters
    /// - `env`: A simple [TableEnvironment]
    /// - `alpha`: The learning rate - must be between 0 and 1
    /// - `gamma`: The discount factor - must be between 0 and 1
    /// - `exploration`: A customized [EpsilonGreedy] policy
    ///
    /// **Panics** if `alpha` or `gamma` is not in the interval `[0,1]`
    pub fn new(env: &'a mut E, alpha: f64, gamma: f64, exploration: EpsilonGreedy) -> Self {
        assert_interval!(alpha, 0.0, 1.0);
        assert_interval!(gamma, 0.0, 1.0);
        Self {
            env,
            q_table: HashMap::new(),
            alpha,
            gamma,
            exploration,
            episode: 0,
        }
    }

    pub fn get_q_table(&self) -> &HashMap<(E::State, E::Action), f64> {
        &self.q_table
    }
}

impl<E: Environment> QAgent for QTableAgent<'_, E>
where
    E::State: Copy + Eq + Hash,
    E::Action: Copy + Eq + Hash,
{
    type Env = E;

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

    fn learn(&mut self, experience: Experience<E>, next_actions: &[E::Action]) {
        let Experience(state, action, next_state, reward) = experience;

        let q_value = *self.q_table.get(&(state, action)).unwrap_or(&0.0);
        let max_next_q = next_actions
            .iter()
            .map(|&a| *self.q_table.get(&(next_state, a)).unwrap_or(&0.0))
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        let new_q_value = reward + self.gamma * max_next_q;
        let weighted_q_value = (1.0 - self.alpha) * q_value + self.alpha * new_q_value;

        self.q_table.insert((state, action), weighted_q_value);
    }

    fn go(&mut self) {
        let mut state = self.env.reset();
        let mut actions = self.env.actions();
        while self.env.is_active() {
            let action = self.act(state, &actions);
            let (next_state, reward) = self.env.step(action);
            actions = self.env.actions();

            self.learn(Experience(state, action, next_state, reward), &actions);

            state = next_state;
        }

        self.episode += 1;
    }
}
