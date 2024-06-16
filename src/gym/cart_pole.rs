use burn::prelude::*;
use gym_rs::core::{ActionReward, Env};
use gym_rs::envs::classical_control::cartpole::{CartPoleEnv, CartPoleObservation};
use gym_rs::utils::renderer::RenderMode;
use rand::seq::IteratorRandom;
use rand::thread_rng;
use strum::{EnumIter, FromRepr, IntoEnumIterator};

use crate::env::{Environment, Report, ToTensor};

fn obs2arr(observation: CartPoleObservation) -> [f32; 4] {
    Vec::from(observation)
        .into_iter()
        .map(|x| x as f32)
        .collect::<Vec<_>>()
        .try_into()
        .expect("vec is length 4")
}

#[derive(FromRepr, EnumIter, Clone, Copy, Debug)]
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

impl<B: Backend<FloatElem = f32>> ToTensor<B, 2, Float> for Vec<[f32; 4]> {
    fn to_tensor(self, device: &B::Device) -> Tensor<B, 2, Float> {
        let len = self.len();
        let data = Data::new(
            self.into_iter().flatten().collect::<Vec<_>>(),
            [len * 4].into(),
        );
        Tensor::from_data(data, device).reshape([-1, 4])
    }
}

#[derive(Debug)]
pub struct CartPole {
    gym_env: CartPoleEnv,
    pub report: Report,
}

impl CartPole {
    pub fn new(render_mode: RenderMode) -> Self {
        Self {
            gym_env: CartPoleEnv::new(render_mode),
            report: Report::new(vec!["reward", "loss"]),
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
