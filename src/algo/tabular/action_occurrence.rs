use std::collections::HashMap;

use crate::{
    decay::{self, Decay},
    env::{DiscreteActionSpace, Environment},
    exploration::{Choice, EpsilonGreedy},
    memory::Exp,
};

use super::Hashable;

/// Configuration for the [`SampleAverageAgent`]
pub struct ActionOccurrenceAgentConfig<D> {
    /// Decay strategy for the exploration parameter
    ///
    /// **Default**: A [`Constant`](decay::Constant) decay strategy with a value of `0.1`
    pub epsilon_decay_strategy: D,
    /// Default value for actions that have not been visited yet
    ///
    /// **Default**: `0.0`
    pub default_action_value: f32,
    /// A function α(n) that returns the learning rate given the number of occurrences of a state-action pair
    ///
    /// **Default**: `|n| 1.0 / n as f32`
    pub alpha_fn: fn(u32) -> f32,
}

impl Default for ActionOccurrenceAgentConfig<decay::Constant> {
    fn default() -> Self {
        Self {
            epsilon_decay_strategy: decay::Constant::new(0.1),
            default_action_value: 0.0,
            alpha_fn: |n| 1.0 / n as f32,
        }
    }
}

/// An entry in the table
#[derive(Default, Debug, Clone, Copy)]
struct Entry {
    value: f32,
    count: u32,
}

/// The simplest tabular agent
///
/// This agent uses a table to store the value and number of occurrences of each state-action pair.
/// The values are updated using the update rule:
///
/// Q<sub>n+1</sub> = Q<sub>n</sub> + α(n)(R<sub>n</sub> - Q<sub>n</sub>)
///
/// where n is the current number of occurrences, Q is the value of the state-action pair, α is the learning rate, and R is the reward.
pub struct ActionOccurrenceAgent<E, D>
where
    E: Environment + DiscreteActionSpace,
    E::State: Hashable,
    E::Action: Hashable,
    D: Decay,
{
    table: HashMap<(E::State, E::Action), Entry>,
    exploration: EpsilonGreedy<D>,
    default_action_value: f32,
    alpha_fn: fn(u32) -> f32,
    episode: u32,
}

impl<E, D> ActionOccurrenceAgent<E, D>
where
    E: Environment + DiscreteActionSpace,
    E::State: Hashable,
    E::Action: Hashable,
    D: Decay,
{
    /// Initialize a new `SampleAverageAgent` in a given environment
    pub fn new(config: ActionOccurrenceAgentConfig<D>) -> Self {
        Self {
            table: HashMap::new(),
            exploration: EpsilonGreedy::new(config.epsilon_decay_strategy),
            default_action_value: config.default_action_value,
            alpha_fn: config.alpha_fn,
            episode: 0,
        }
    }

    /// Choose an action based on the current state and exploration policy
    fn act(&self, env: &E, state: E::State, actions: &[E::Action]) -> E::Action {
        match self.exploration.choose(self.episode) {
            Choice::Explore => env.random_action(),
            Choice::Exploit => *actions
                .iter()
                .max_by(|&a, &b| {
                    let a_value = self
                        .table
                        .get(&(state, *a))
                        .copied()
                        .unwrap_or(Entry {
                            value: self.default_action_value,
                            count: 0,
                        })
                        .value;
                    let b_value = self
                        .table
                        .get(&(state, *b))
                        .copied()
                        .unwrap_or(Entry {
                            value: self.default_action_value,
                            count: 0,
                        })
                        .value;
                    a_value.partial_cmp(&b_value).unwrap()
                })
                .expect("There is always at least one action available"),
        }
    }

    /// Learn from a given experience and update the table
    fn learn(&mut self, experience: Exp<E>) {
        let Exp {
            state,
            action,
            reward,
            ..
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
            let action = self.act(env, state, &actions);
            let (next, reward) = env.step(action);
            next_state = next;
            actions = env.actions();

            self.learn(Exp {
                state,
                action,
                next_state,
                reward,
            });
        }

        self.episode += 1;
    }
}
