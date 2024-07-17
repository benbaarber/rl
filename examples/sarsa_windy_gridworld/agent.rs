use std::collections::HashMap;

use rl::{
    decay,
    env::{DiscreteActionSpace, Environment},
    exploration::{Choice, EpsilonGreedy},
};

use crate::env::{Action, Pos, WindyGridworld};

type E = WindyGridworld;
type State = Pos;

pub struct SarsaAgent {
    q_table: HashMap<(State, Action), f32>,
    exploration: EpsilonGreedy<decay::Constant>,
    alpha: f32,
    gamma: f32,
    episode: u32,
}

impl SarsaAgent {
    pub fn new(epsilon: f32, alpha: f32, gamma: f32) -> Self {
        Self {
            q_table: HashMap::new(),
            exploration: EpsilonGreedy::new(decay::Constant::new(epsilon)),
            alpha,
            gamma,
            episode: 0,
        }
    }

    /// Choose an action based on the current state and exploration policy
    fn act(&self, env: &E, state: State) -> Action {
        match self.exploration.choose(self.episode) {
            Choice::Explore => env.random_action(),
            Choice::Exploit => *env
                .actions()
                .iter()
                .max_by(|&a, &b| {
                    let a_value = *self.q_table.get(&(state, *a)).unwrap_or(&0.0);
                    let b_value = *self.q_table.get(&(state, *b)).unwrap_or(&0.0);
                    a_value.partial_cmp(&b_value).unwrap()
                })
                .expect("There is always at least one action available"),
        }
    }

    fn learn(
        &mut self,
        state: State,
        action: Action,
        reward: f32,
        next_state: State,
        next_action: Action,
    ) {
        let q_value = *self.q_table.get(&(state, action)).unwrap_or(&0.0);
        let next_q_value = *self.q_table.get(&(next_state, next_action)).unwrap_or(&0.0);
        let update = q_value + self.alpha * (reward + self.gamma * next_q_value - q_value);
        self.q_table.insert((state, action), update);
    }

    pub fn go(&mut self, env: &mut E) {
        let mut next_state = env.reset();
        let mut next_action = self.act(&env, next_state);

        loop {
            let (state, action) = (next_state, next_action);
            let (Some(next), reward) = env.step(next_action) else {
                break;
            };
            next_state = next;
            next_action = self.act(&env, state);

            self.learn(state, action, reward, next_state, next_action);
        }

        self.episode += 1;
    }
}
