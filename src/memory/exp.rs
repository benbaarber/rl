use crate::env::Environment;

/// Represents a single experience or transition in the environment
pub struct Exp<E: Environment> {
    /// The state of the environment before taking the action
    pub state: E::State,
    /// The action taken in the given state
    pub action: E::Action,
    /// The state of the environment after the action is taken, or if terminal, `None`
    pub next_state: Option<E::State>,
    /// The reward received after taking the action
    pub reward: f32,
}

impl<E: Environment> Clone for Exp<E> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            action: self.action.clone(),
            next_state: self.next_state.clone(),
            reward: self.reward,
        }
    }
}

/// A zipped batch of [experiences](Exp) where the batch size is known at compile time
///
/// The batch size must be passed to the const generic `S`
#[derive(Clone)]
pub struct ExpBatch<E: Environment, const S: usize> {
    /// The state of the environment before taking the action
    pub states: [E::State; S],
    /// The action taken in the given state
    pub actions: [E::Action; S],
    /// The state of the environment after the action is taken, or if terminal, `None`
    pub next_states: [Option<E::State>; S],
    /// The reward received after taking the action
    pub rewards: [f32; S],
}

impl<E: Environment, const S: usize> ExpBatch<E, S> {
    // TODO: try to avoid temporary heap allocation
    /// Construct an `ExpBatch` from an iterator of [experiences](Exp)
    pub fn from_iter(iter: impl IntoIterator<Item = Exp<E>>) -> Self {
        let batch = DynamicExpBatch::from_iter(iter, S);
        Self {
            states: batch.states.try_into().ok().unwrap(),
            actions: batch.actions.try_into().ok().unwrap(),
            next_states: batch.next_states.try_into().ok().unwrap(),
            rewards: batch.rewards.try_into().ok().unwrap(),
        }
    }
}

/// A zipped batch of [experiences](Exp) where the batch size is not known at compile time
#[derive(Clone)]
pub struct DynamicExpBatch<E: Environment> {
    /// The state of the environment before taking the action
    pub states: Vec<E::State>,
    /// The action taken in the given state
    pub actions: Vec<E::Action>,
    /// The state of the environment after the action is taken, or if terminal, `None`
    pub next_states: Vec<Option<E::State>>,
    /// The reward received after taking the action
    pub rewards: Vec<f32>,
}

impl<E: Environment> DynamicExpBatch<E> {
    /// Construct an `ExpBatch` from an iterator of [experience](Exp) references and a specified batch size
    pub fn from_iter(iter: impl IntoIterator<Item = Exp<E>>, batch_size: usize) -> Self {
        let batch = Self {
            states: Vec::with_capacity(batch_size),
            actions: Vec::with_capacity(batch_size),
            next_states: Vec::with_capacity(batch_size),
            rewards: Vec::with_capacity(batch_size),
        };

        iter.into_iter().fold(batch, |mut b, e| {
            b.states.push(e.state.clone());
            b.actions.push(e.action.clone());
            b.next_states.push(e.next_state.clone());
            b.rewards.push(e.reward);
            b
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockEnv;

    impl Environment for MockEnv {
        type State = i32;
        type Action = i32;

        fn step(&mut self, _action: Self::Action) -> (Option<Self::State>, f32) {
            (None, 0.0)
        }

        fn reset(&mut self) -> Self::State {
            0
        }
    }

    const BATCH_SIZE: usize = 2;

    fn create_mock_exp_array() -> [Exp<MockEnv>; BATCH_SIZE] {
        let exp1 = Exp {
            state: 0,
            action: 1,
            next_state: Some(1),
            reward: 1.0,
        };
        let exp2 = Exp {
            state: 1,
            action: 2,
            next_state: None,
            reward: 0.0,
        };
        [exp1, exp2]
    }

    #[test]
    fn dynamic_exp_batch_from_iter() {
        let experiences = create_mock_exp_array();
        let batch = DynamicExpBatch::from_iter(experiences, BATCH_SIZE);

        assert_eq!(batch.states, [0, 1], "States constructed correctly");
        assert_eq!(batch.actions, [1, 2], "States constructed correctly");
        assert_eq!(
            batch.next_states,
            [Some(1), None],
            "States constructed correctly"
        );
        assert_eq!(batch.rewards, [1.0, 0.0], "States constructed correctly");
    }

    #[test]
    fn exp_batch_from_iter() {
        let experiences = create_mock_exp_array();
        let batch = ExpBatch::<_, BATCH_SIZE>::from_iter(experiences);

        assert_eq!(batch.states, [0, 1], "States constructed correctly");
        assert_eq!(batch.actions, [1, 2], "States constructed correctly");
        assert_eq!(
            batch.next_states,
            [Some(1), None],
            "States constructed correctly"
        );
        assert_eq!(batch.rewards, [1.0, 0.0], "States constructed correctly");
    }
}
