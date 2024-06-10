// use burn::prelude::*;
// use rl::env::Environment;

// pub struct Agent<E, B, M, L, Exp>
// where
//     E: Environment,
//     B: Backend,
//     M: Module<B>,
// {
//     policy_net: Model<B>,
//     target_net: Model<B>,
//     memory: ReplayMemory<GrassyField<FIELD_SIZE>>,
//     loss: MseLoss<B>,
//     exploration: EpsilonGreedy<decay::Exponential>,
//     gamma: f32,
//     tau: f32,
//     lr: f32,
//     episode: u32,
// }

use burn::{
    grad_clipping::GradientClippingConfig,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::*,
};
use log::info;
use nn::loss::{MseLoss, Reduction};
use rand::{seq::IteratorRandom, thread_rng};
use rl::{
    decay,
    env::Environment,
    exploration::{Choice, EpsilonGreedy},
    gym::{cart_pole::CPAction, CartPole},
    memory::{Exp, ReplayMemory},
};
use strum::IntoEnumIterator;

use crate::{
    model::{Model, ModelConfig},
    DQNAutodiffBackend as B, DEVICE,
};

pub struct Agent {
    policy_net: Option<Model<B>>,
    target_net: Option<Model<B>>,
    memory: ReplayMemory<CartPole>,
    loss: MseLoss<B>,
    exploration: EpsilonGreedy<decay::Exponential>,
    gamma: f32,
    tau: f32,
    lr: f32,
    total_steps: u32,
}

impl Agent {
    pub fn new(model_config: ModelConfig, exploration: EpsilonGreedy<decay::Exponential>) -> Self {
        Self {
            policy_net: Some(model_config.init(&*DEVICE)),
            target_net: Some(model_config.init(&*DEVICE)),
            memory: ReplayMemory::new(50000),
            loss: MseLoss::new(),
            exploration,
            gamma: 0.99,
            tau: 1e-2,
            lr: 1e-3,
            total_steps: 0,
        }
    }
}

type State = [f32; 4];
type Action = CPAction;

impl Agent {
    fn act(&self, state: State) -> Action {
        match self.exploration.choose(self.total_steps) {
            Choice::Explore => CPAction::iter().choose(&mut thread_rng()).unwrap(),
            Choice::Exploit => {
                let input = Tensor::<B, 2>::from_floats([state], &*DEVICE);
                let output = self
                    .policy_net
                    .as_ref()
                    .unwrap()
                    .forward(input)
                    .argmax(1)
                    .into_scalar();
                CPAction::from_repr(output.try_into().unwrap()).unwrap()
            }
        }
    }

    fn learn(&mut self, optimizer: &mut impl Optimizer<Model<B>, B>) -> Option<f64> {
        const BATCH_SIZE: usize = 128;
        let batch = self.memory.sample_zipped::<BATCH_SIZE>()?;

        let mut non_terminal_mask = [false; BATCH_SIZE];
        for (i, s) in batch.next_states.iter().enumerate() {
            if s.is_some() {
                non_terminal_mask[i] = true;
            }
        }
        let non_terminal_mask = Tensor::<B, 1, Bool>::from_bool(non_terminal_mask.into(), &*DEVICE);

        let next_states = Tensor::<B, 2>::cat(
            batch
                .next_states
                .into_iter()
                .flatten()
                .map(|ns| Tensor::<B, 2>::from_floats([ns], &*DEVICE))
                .collect::<Vec<_>>(),
            0,
        );

        let states = Tensor::<B, 2>::from_floats(batch.states, &*DEVICE);
        let actions = Tensor::<B, 1, Int>::from_ints(
            Data::new(
                batch
                    .actions
                    .into_iter()
                    .map(|a| a as i32)
                    .collect::<Vec<_>>(),
                [BATCH_SIZE].into(),
            ),
            &*DEVICE,
        );
        let rewards = Tensor::<B, 1>::from_floats(batch.rewards, &*DEVICE);

        let policy_net = self.policy_net.take().unwrap();
        let target_net = self.target_net.take().unwrap();

        let q_values = policy_net
            .forward(states)
            .gather(1, actions.unsqueeze_dim(1))
            .squeeze(1);
        let max_next_q_values = Tensor::<B, 1>::zeros([BATCH_SIZE], &*DEVICE).mask_where(
            non_terminal_mask,
            target_net.forward(next_states).max_dim(1).squeeze(1),
        );

        let discounted_expected_return = rewards + (max_next_q_values * self.gamma);

        let loss = self
            .loss
            .forward(q_values, discounted_expected_return, Reduction::Mean);
        let grads = GradientsParams::from_grads(loss.backward(), &self.policy_net);

        self.policy_net = Some(optimizer.step(self.lr.into(), policy_net, grads));
        self.target_net = Some(target_net.soft_update(self.policy_net.as_ref().unwrap(), self.tau));

        Some(loss.into_scalar() as f64)
    }

    pub fn go(&mut self, env: &mut CartPole) {
        let mut next_state = Some(env.reset());
        let mut optimizer = AdamWConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Value(100.0)))
            .init::<B, Model<B>>();

        while let Some(state) = next_state {
            let action = self.act(state);
            let (next, reward) = env.step(action);
            next_state = next;

            self.memory.push(Exp {
                state,
                action,
                next_state,
                reward,
            });

            let loss = self.learn(&mut optimizer);

            if let Some(loss) = loss {
                info!("{:?}", loss);
                env.report.entry("loss").and_modify(|x| *x = loss);
            }

            self.total_steps += 1;
        }
    }
}
