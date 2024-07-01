use std::collections::HashMap;

use crate::{
    decay::{self, Decay},
    env::{DiscreteActionSpace, Environment},
    exploration::{Choice, EpsilonGreedy},
    memory::Exp,
};

use super::Hashable;

/// Configuration for the [`SampleAverageAgent`]
pub struct SampleAverageAgentConfig<D: Decay> {
    pub exploration: EpsilonGreedy<D>,
}

impl Default for SampleAverageAgentConfig<decay::Constant> {
    fn default() -> Self {
        Self {
            exploration: EpsilonGreedy::new(decay::Constant::new(0.1)),
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
pub struct SampleAverageAgent<E, D>
where
    E: Environment + DiscreteActionSpace,
    E::State: Hashable,
    E::Action: Hashable,
    D: Decay,
{
    table: HashMap<(E::State, E::Action), Entry>,
    exploration: EpsilonGreedy<D>,
    episode: u32,
}

impl<E, D> SampleAverageAgent<E, D>
where
    E: Environment + DiscreteActionSpace,
    E::State: Hashable,
    E::Action: Hashable,
    D: Decay,
{
    /// Initialize a new `SampleAverageAgent` in a given environment
    pub fn new(config: SampleAverageAgentConfig<D>) -> Self {
        Self {
            table: HashMap::new(),
            exploration: config.exploration,
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
                        .unwrap_or_default()
                        .value;
                    let b_value = self
                        .table
                        .get(&(state, *b))
                        .copied()
                        .unwrap_or_default()
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
            next_state: _,
            reward,
        } = experience;

        self.table
            .entry((state, action))
            .and_modify(|e| {
                e.count += 1;
                e.value += (reward - e.value) / (e.count as f32);
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
