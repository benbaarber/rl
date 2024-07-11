use burn::prelude::*;
use gym_rs::core::{ActionReward, Env};
use gym_rs::envs::classical_control::cartpole::{CartPoleEnv, CartPoleObservation};
use gym_rs::utils::renderer::RenderMode;
use rand::seq::IteratorRandom;
use rand::thread_rng;
use strum::{EnumIter, FromRepr, IntoEnumIterator, VariantArray};

use crate::env::{DiscreteActionSpace, Environment, Report};
use crate::traits::ToTensor;

fn obs2arr(observation: CartPoleObservation) -> [f32; 4] {
    Vec::from(observation)
        .into_iter()
        .map(|x| x as f32)
        .collect::<Vec<_>>()
        .try_into()
        .expect("vec is length 4")
}

/// Actions for the [`CartPole`] environment, representing applying a left or right force to the cart
#[derive(FromRepr, EnumIter, VariantArray, Clone, Copy, Debug)]
pub enum CPAction {
    Left = 0,
    Right = 1,
}

impl From<usize> for CPAction {
    fn from(value: usize) -> Self {
        Self::from_repr(value).expect("CPAction::from is only called with valid values [0, 1]")
    }
}

impl<B: Backend<IntElem = i32>> ToTensor<B, 2, Int> for Vec<CPAction> {
    fn to_tensor(self, device: &B::Device) -> Tensor<B, 2, Int> {
        let len = self.len();
        let data = Data::new(
            self.into_iter().map(|x| x as i32).collect::<Vec<_>>(),
            [len].into(),
        );
        Tensor::from_data(data, device).unsqueeze_dim(1)
    }
}

/// The classic CartPole reinforcement learning environment
///
/// This implementation is a thin wrapper around [gym_rs](https://github.com/MathisWellmann/gym-rs)
#[derive(Debug, Clone)]
pub struct CartPole {
    gym_env: CartPoleEnv,
    pub report: Report,
}

impl CartPole {
    pub fn new(render_mode: RenderMode) -> Self {
        Self {
            gym_env: CartPoleEnv::new(render_mode),
            report: Report::new(vec!["reward"]),
        }
    }
}

impl Environment for CartPole {
    type State = [f32; 4];
    type Action = CPAction;

    fn random_action(&self) -> Self::Action {
        CPAction::iter().choose(&mut thread_rng()).unwrap()
    }

    fn step(&mut self, action: Self::Action) -> (Option<Self::State>, f32) {
        let ActionReward {
            observation,
            reward,
            done,
            ..
        } = self.gym_env.step(action as usize);

        let next_state = if done {
            None
        } else {
            Some(obs2arr(observation))
        };

        self.report.entry("reward").and_modify(|x| *x += *reward);

        (next_state, *reward as f32)
    }

    fn reset(&mut self) -> Self::State {
        obs2arr(self.gym_env.reset(None, false, None).0)
    }
}

impl DiscreteActionSpace for CartPole {
    fn actions(&self) -> Vec<Self::Action> {
        CPAction::VARIANTS.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obs2arr_functional() {
        let obs = CartPoleObservation::new(0.0.into(), 1.0.into(), 2.0.into(), 3.0.into());
        let arr = obs2arr(obs);
        assert_eq!(arr, [0.0, 1.0, 2.0, 3.0], "obs2arr conversion works");
    }
}
