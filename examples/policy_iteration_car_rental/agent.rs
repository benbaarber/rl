use std::collections::HashMap;

use rl::env::{DiscreteActionSpace, DiscreteStateSpace, Environment};

use crate::env::CarRental;

type State = [i32; 2];
type Action = i32;

/// A policy iteration agent
///
/// This agent applies policy iteration with respect to the state values. It is a dynamic programming approach
/// and requires a full model of the environment's dynamics. This implementation
/// simply approximates the dynamics through actual environment steps instead of requiring a complete model.
/// As a tabular method, a discrete, small, and hashable state and action space is required.
pub struct PolicyIterationAgent {
    state_value: HashMap<State, f32>,
    policy: HashMap<State, Action>,
    gamma: f32,
}

#[allow(unused)]
impl PolicyIterationAgent {
    /// Initialize a new `PolicyIterationAgent`
    pub fn new(gamma: f32) -> Self {
        Self {
            state_value: HashMap::new(),
            policy: HashMap::new(),
            gamma,
        }
    }

    fn evaluate(&mut self, env: &mut CarRental) {
        let mut delta = f32::INFINITY;
        while delta > 1e-4 {
            delta = 0.0;
            for state in env.states() {
                let action = self.policy.entry(state).or_default();

                let mut new_value = 0.0;
                for outcome in env.dynamics(state, *action) {
                    let next_state_value = *self.state_value.entry(outcome.next_state).or_default();
                    let ret = outcome.reward + self.gamma * next_state_value;
                    new_value += outcome.prob * ret;
                }

                let old_value = self
                    .state_value
                    .insert(state, new_value)
                    .unwrap_or_default();

                delta = delta.max((old_value - new_value).abs());
            }
        }
    }

    fn improve(&mut self, env: &mut CarRental) {
        for state in env.states() {
            let mut action_values = vec![];
            for action in env.actions() {
                let [i1, i2] = state;
                if !(-i2..i1).contains(&action) {
                    continue;
                }

                let mut action_value = 0.0;
                for outcome in env.dynamics(state, action) {
                    let next_state_value = *self.state_value.entry(outcome.next_state).or_default();
                    let ret = outcome.reward + self.gamma * next_state_value;
                    action_value += outcome.prob * ret;
                }

                action_values.push((action, action_value))
            }

            let new_action = action_values
                .into_iter()
                .max_by(|(_, v1), (_, v2)| v1.partial_cmp(v2).unwrap())
                .unwrap_or_default()
                .0;

            self.policy.insert(state, new_action);
        }
    }

    /// Run the policy iteration algorithm to train the agent
    pub fn learn(&mut self, env: &mut CarRental, num_iterations: u32) {
        for _ in 0..num_iterations {
            println!("Evaluating...");
            self.evaluate(env);
            println!("Improving...");
            self.improve(env);
        }
    }

    /// Deploy the trained agent into the environment
    pub fn go(&self, env: &mut CarRental) {
        let mut next_state = Some(env.reset());

        while let Some(state) = next_state {
            let action = self.policy.get(&state).cloned().unwrap_or_default();
            let (next, reward) = env.step(action);
            next_state = next;
        }
    }

    /// Get the agents policy
    pub fn policy(&self) -> &HashMap<State, Action> {
        &self.policy
    }

    /// Get the agents state value function
    pub fn state_value(&self) -> &HashMap<State, f32> {
        &self.state_value
    }
}
