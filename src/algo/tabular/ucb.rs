use std::collections::HashMap;

use crate::{
    env::{DiscreteActionSpace, Environment},
    memory::Exp,
};

use super::Hashable;

/// Configuration for the [`UCBAgent`]
#[derive(Debug, Clone)]
pub struct UCBAgentConfig {
    /// c value for the UCB exploration strategy
    ///
    /// **Default**: `1.0`
    pub ucb_c: f32,
    /// Default value for actions that have not been visited yet
    ///
    /// **Default**: `0.0`
    pub default_action_value: f32,
    /// A function α(n) that returns the learning rate given the number of occurrences of a state-action pair
    ///
    /// **Default**: `|n| 1.0 / n as f32`
    pub alpha_fn: fn(u32) -> f32,
}

impl Default for UCBAgentConfig {
    fn default() -> Self {
        Self {
            ucb_c: 1.0,
            default_action_value: 0.0,
            alpha_fn: |n| 1.0 / n as f32,
        }
    }
}

/// An entry in the table
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Entry {
    value: f32,
    count: u32,
}

/// Upper confidence bound agent
///
/// This agent is almost identical to the [`ActionOccurrenceAgent`](super::action_occurrence::ActionOccurrenceAgent),
/// but uses the UCB exploration policy instead of epsilon-greedy.
#[derive(Debug, Clone)]
pub struct UCBAgent<E>
where
    E: Environment + DiscreteActionSpace,
    E::State: Hashable,
    E::Action: Hashable,
{
    table: HashMap<(E::State, E::Action), Entry>,
    ucb_c: f32,
    default_action_value: f32,
    alpha_fn: fn(u32) -> f32,
    t: u32,
    episode: u32,
}

impl<E> UCBAgent<E>
where
    E: Environment + DiscreteActionSpace,
    E::State: Hashable,
    E::Action: Hashable + From<usize>,
{
    /// Initialize a new `SampleAverageAgent` in a given environment
    pub fn new(config: UCBAgentConfig) -> Self {
        Self {
            table: HashMap::new(),
            ucb_c: config.ucb_c,
            default_action_value: config.default_action_value,
            alpha_fn: config.alpha_fn,
            t: 0,
            episode: 0,
        }
    }

    /// Choose an action based on the current state
    fn act(&self, state: E::State, actions: &[E::Action]) -> E::Action {
        let action_entries = actions
            .iter()
            .map(|&action| {
                self.table.get(&(state, action)).copied().unwrap_or(Entry {
                    value: self.default_action_value,
                    count: 0,
                })
            })
            .collect::<Vec<_>>();

        let t = (self.t + 1) as f32;
        let k = self.ucb_c * t.ln().sqrt();
        let choice = action_entries
            .iter()
            .enumerate()
            .map(|(i, &Entry { value: q, count })| {
                let n = count as f32;
                if n <= 0.0 {
                    return (i, f32::MAX);
                }
                (i, q + k * n.powf(-0.5))
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .expect("`q_values` is not empty");

        choice.into()
    }

    /// Learn from a given experience and update the table
    fn learn(&mut self, experience: Exp<E>) {
        let Exp {
            state,
            action,
            next_state: _,
            reward,
        } = experience;

        self.table
            .entry((state, action))
            .and_modify(|e| {
                e.count += 1;
                e.value += (self.alpha_fn)(e.count) * (reward - e.value);
            })
            .or_insert(Entry {
                value: reward,
                count: 1,
            });
    }

    /// Run the agent in the given environment
    pub fn go(&mut self, env: &mut E) {
        let mut next_state = Some(env.reset());
        let mut actions = env.actions();
        while let Some(state) = next_state {
            let action = self.act(state, &actions);
            let (next, reward) = env.step(action);
            next_state = next;
            actions = env.actions();

            self.learn(Exp {
                state,
                action,
                next_state,
                reward,
            });

            self.t += 1;
        }

        self.episode += 1;
    }
}
