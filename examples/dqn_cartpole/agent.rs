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
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::*,
};
use nn::loss::{HuberLoss, HuberLossConfig, Reduction};
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
    policy_net: Model<B>,
    target_net: Model<B>,
    memory: ReplayMemory<CartPole>,
    loss: HuberLoss<B>,
    exploration: EpsilonGreedy<decay::Exponential>,
    gamma: f32,
    tau: f32,
    lr: f32,
    episode: u32,
}

impl Agent {
    pub fn new(model_config: ModelConfig, exploration: EpsilonGreedy<decay::Exponential>) -> Self {
        Self {
            policy_net: model_config.init(&*DEVICE),
            target_net: model_config.init(&*DEVICE),
            memory: ReplayMemory::new(50000),
            loss: HuberLossConfig::new(1.35).init(&*DEVICE), // Make delta a hyperparameter
            exploration,
            gamma: 0.86,
            tau: 2.7e-2,
            lr: 3.58e-3,
            episode: 0,
        }
    }
}

type State = [f32; 4];
type Action = CPAction;

impl Agent {
    fn act(&self, state: State) -> Action {
        match self.exploration.choose(self.episode) {
            Choice::Explore => CPAction::iter().choose(&mut thread_rng()).unwrap(),
            Choice::Exploit => {
                let input = Tensor::<B, 2>::from_floats([state], &*DEVICE);
                let output = self.policy_net.forward(input).argmax(1).into_scalar();
                CPAction::from_repr(output.try_into().unwrap()).unwrap()
            }
        }
    }

    fn learn(&mut self, optimizer: &mut impl Optimizer<Model<B>, B>) {
        const BATCH_SIZE: usize = 128;
        let Some(batch) = self.memory.sample_zipped::<BATCH_SIZE>() else {
            return;
        };

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

        let q_values = self
            .policy_net
            .forward(states)
            .gather(1, actions.unsqueeze_dim(1))
            .squeeze(1);
        let max_next_q_values = Tensor::<B, 1>::zeros([BATCH_SIZE], &*DEVICE).mask_where(
            non_terminal_mask,
            self.target_net.forward(next_states).max_dim(1).squeeze(1),
        );

        let discounted_expected_return = rewards + (max_next_q_values * self.gamma);

        let loss = self
            .loss
            .forward(q_values, discounted_expected_return, Reduction::Auto);
        let grads = GradientsParams::from_grads(loss.backward(), &self.policy_net);

        let policy_net = unsafe { std::ptr::read(&self.policy_net) };
        self.policy_net = optimizer.step(self.lr.into(), policy_net, grads);

        let target_net = unsafe { std::ptr::read(&self.target_net) };
        self.target_net = target_net.soft_update(&self.policy_net, self.tau);
    }

    pub fn go(&mut self, env: &mut CartPole) {
        let mut next_state = Some(env.reset());
        let mut optimizer = AdamWConfig::new()
            .with_epsilon(3.58e-3)
            .init::<B, Model<B>>();

        while let Some(state) = next_state {
            let action = self.act(state);
            // println!("Action: {:?}", action);
            let (next, reward) = env.step(action);
            next_state = next;

            self.memory.push(Exp {
                state,
                action,
                next_state,
                reward,
            });

            self.learn(&mut optimizer);
        }

        self.episode += 1;
    }
}
