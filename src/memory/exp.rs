use std::fmt::Debug;

use crate::env::Environment;

/// Represents a single experience or transition in the environment
pub struct Exp<E: Environment> {
    /// The state of the environment before taking the action
    pub state: E::State,
    /// The action taken in the given state
    pub action: E::Action,
    /// The reward received after taking the action
    pub reward: f32,
    /// The state of the environment after the action is taken, or if terminal, `None`
    pub next_state: Option<E::State>,
}

impl<E: Environment> Clone for Exp<E> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            action: self.action.clone(),
            reward: self.reward,
            next_state: self.next_state.clone(),
        }
    }
}

impl<E: Environment> Debug for Exp<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Exp")
            .field("state", &self.state)
            .field("action", &self.action)
            .field("reward", &self.reward)
            .field("next_state", &self.next_state)
            .finish()
    }
}

/// A zipped batch of [experiences](Exp)
#[derive(Clone, Debug)]
pub struct ExpBatch<E: Environment> {
    /// The state of the environment before taking the action
    pub states: Vec<E::State>,
    /// The action taken in the given state
    pub actions: Vec<E::Action>,
    /// The reward received after taking the action
    pub rewards: Vec<f32>,
    /// The state of the environment after the action is taken, or if terminal, `None`
    pub next_states: Vec<Option<E::State>>,
}

impl<E: Environment> ExpBatch<E> {
    /// Construct an `ExpBatch` from an iterator of [experience](Exp) references and a specified batch size
    pub fn from_iter(iter: impl IntoIterator<Item = Exp<E>>, batch_size: usize) -> Self {
        let batch = Self {
            states: Vec::with_capacity(batch_size),
            actions: Vec::with_capacity(batch_size),
            rewards: Vec::with_capacity(batch_size),
            next_states: Vec::with_capacity(batch_size),
        };

        iter.into_iter().fold(batch, |mut b, e| {
            b.states.push(e.state.clone());
            b.actions.push(e.action.clone());
            b.rewards.push(e.reward);
            b.next_states.push(e.next_state.clone());
            b
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::env::tests::MockEnv;

    use super::*;

    const BATCH_SIZE: usize = 2;

    fn create_mock_exp_array() -> [Exp<MockEnv>; BATCH_SIZE] {
        let exp1 = Exp {
            state: 0,
            action: 1,
            reward: 1.0,
            next_state: Some(1),
        };
        let exp2 = Exp {
            state: 1,
            action: 2,
            reward: 0.0,
            next_state: None,
        };
        [exp1, exp2]
    }

    pub(crate) fn create_mock_exp_vec(n: i32) -> Vec<Exp<MockEnv>> {
        (0..n)
            .map(|i| Exp {
                state: i,
                action: i + 1,
                next_state: Some(i + 1),
                reward: 1.0,
            })
            .collect()
    }

    #[test]
    fn exp_batch_from_iter() {
        let experiences = create_mock_exp_array();
        let batch = ExpBatch::from_iter(experiences, BATCH_SIZE);

        assert_eq!(batch.states, [0, 1], "States constructed correctly");
        assert_eq!(batch.actions, [1, 2], "Actions constructed correctly");
        assert_eq!(batch.rewards, [1.0, 0.0], "Rewards constructed correctly");
        assert_eq!(
            batch.next_states,
            [Some(1), None],
            "Next states constructed correctly"
        );
    }
}
