use gym_rs::core::{ActionReward, Env};
use gym_rs::envs::classical_control::cartpole::{CartPoleEnv, CartPoleObservation};
use gym_rs::utils::renderer::RenderMode;
use strum::{EnumIter, FromRepr};

use crate::env::{Environment, Report};

fn obs2arr(observation: CartPoleObservation) -> [f32; 4] {
    Vec::from(observation)
        .into_iter()
        .map(|x| x as f32)
        .collect::<Vec<_>>()
        .try_into()
        .expect("vec is length 4")
}

#[derive(FromRepr, EnumIter, Clone, Copy)]
pub enum CPAction {
    Left = 0,
    Right = 1,
}

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
