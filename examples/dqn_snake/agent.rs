use burn::{
    nn::loss::Reduction,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::*,
};
use nn::loss::MseLoss;
use rand::{seq::IteratorRandom, thread_rng};
use rl::{
    decay,
    env::Environment,
    exploration::{Choice, EpsilonGreedy},
    gym::{
        grassy_field::{Dir, Grid},
        GrassyField,
    },
    memory::{Exp, ReplayMemory},
};
use strum::IntoEnumIterator;

use crate::{
    model::{Model, ModelConfig},
    DQNAutodiffBackend as B, DEVICE, FIELD_SIZE,
};

pub struct SnakeDQN {
    policy_net: Model<B>,
    target_net: Model<B>,
    memory: ReplayMemory<GrassyField<FIELD_SIZE>>,
    loss: MseLoss<B>,
    exploration: EpsilonGreedy<decay::Exponential>,
    gamma: f32,
    tau: f32,
    lr: f32,
    episode: u32,
}

impl SnakeDQN {
    pub fn new(model_config: ModelConfig, exploration: EpsilonGreedy<decay::Exponential>) -> Self {
        Self {
            policy_net: model_config.init(&*DEVICE),
            target_net: model_config.init(&*DEVICE),
            memory: ReplayMemory::new(50000),
            loss: MseLoss::new(),
            exploration,
            gamma: 0.86,
            tau: 2.7e-2,
            lr: 3.58e-3,
            episode: 0,
        }
    }
}

type State = Grid<FIELD_SIZE>;
type Action = Dir;

impl SnakeDQN {
    fn act(&self, state: State, _actions: &[Action]) -> Action {
        match self.exploration.choose(self.episode) {
            Choice::Explore => Dir::iter().choose(&mut thread_rng()).unwrap(),
            Choice::Exploit => {
                let field_tensor = Tensor::<B, 2>::from_floats(state, &*DEVICE)
                    .reshape([1, FIELD_SIZE, FIELD_SIZE]);
                let choice = self
                    .policy_net
                    .forward(field_tensor)
                    .squeeze::<1>(0)
                    .argmax(0)
                    .into_scalar();
                Dir::from_repr(choice.try_into().unwrap()).unwrap()
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

        let next_states = Tensor::<B, 3>::cat(
            batch
                .next_states
                .into_iter()
                .flatten()
                .map(|ns| Tensor::<B, 3>::from_floats([ns], &*DEVICE))
                .collect::<Vec<_>>(),
            0,
        );

        let states = Tensor::<B, 3>::from_floats(batch.states, &*DEVICE)
            .reshape([0, FIELD_SIZE, FIELD_SIZE]);
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

    pub fn go(&mut self, env: &mut GrassyField<FIELD_SIZE>) {
        let mut next_state = Some(env.reset());
        let mut optimizer = AdamWConfig::new()
            .with_epsilon(3.58e-3)
            .init::<B, Model<B>>();

        while let Some(state) = next_state {
            let actions = env.actions();
            let action = self.act(state, &actions);
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
        let score = env.score() as f64;
        env.report.entry("score").and_modify(|x| *x += score);
    }
}
