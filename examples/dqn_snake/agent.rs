use burn::{
    nn::loss::{HuberLoss, HuberLossConfig, Reduction},
    optim::GradientsParams,
    tensor::{Bool, Int, Tensor},
};
use rand::{seq::IteratorRandom, thread_rng};
use rl::{
    algo::QAgent,
    env::Environment,
    exploration::{Choice, EpsilonGreedy},
    gym::{grassy_field::Dir, GrassyField},
    memory::{Exp, Memory, ReplayMemory},
};
use strum::IntoEnumIterator;

use crate::{
    model::{Model, ModelConfig},
    DQNAutodiffBackend as B, DEVICE,
};

pub struct SnakeDQN<'a> {
    env: &'a mut GrassyField,
    policy_net: Model<B>,
    target_net: Model<B>,
    memory: ReplayMemory<GrassyField>,
    loss: HuberLoss<B>,
    exploration: EpsilonGreedy,
    gamma: f64,
    tau: f64,
    lr: f64,
    episode: u32,
}

impl<'a> SnakeDQN<'a> {
    pub fn new(
        env: &'a mut GrassyField,
        model_config: ModelConfig,
        loss_config: HuberLossConfig,
        exploration: EpsilonGreedy,
    ) -> Self {
        Self {
            env,
            policy_net: model_config.init(&DEVICE),
            target_net: model_config.init(&DEVICE),
            memory: ReplayMemory::new(50000),
            loss: loss_config.init(&DEVICE),
            exploration,
            gamma: 0.86,
            tau: 2.7e-2,
            lr: 3.58e-3,
            episode: 0,
        }
    }
}

impl QAgent for SnakeDQN<'_> {
    type Env = GrassyField;

    fn act(
        &self,
        state: <Self::Env as Environment>::State,
        actions: &[<Self::Env as Environment>::Action],
    ) -> <Self::Env as Environment>::Action {
        match self.exploration.choose(self.episode) {
            Choice::Explore => Dir::iter().choose(&mut thread_rng()).unwrap(),
            Choice::Exploit => {
                let size = self.env.field_size();
                let field_tensor =
                    Tensor::<B, 1>::from_floats(state, &DEVICE).reshape([1, size, size]);
                let choice = self
                    .policy_net
                    .forward(field_tensor)
                    .argmax(0)
                    .into_scalar();
                Dir::from_repr(choice.try_into().unwrap()).unwrap()
            }
        }
    }

    fn learn(
        &mut self,
        experience: Exp<Self::Env>,
        next_actions: &[<Self::Env as Environment>::Action],
    ) {
        let batch_size = 128;
        let Some(batch) = self.memory.sample_zipped(batch_size) else {
            return;
        };

        let non_terminal_mask = Tensor::<B, 1, Bool>::from_bool(
            batch.next_states.iter().map(|s| s.is_some()).collect(),
            &DEVICE,
        );

        let field_size = self.env.field_size();
        let next_states =
            Tensor::<B, 2>::from_floats(batch.next_states.into_iter().flatten().collect(), &DEVICE)
                .reshape([0, field_size, field_size]);

        let states =
            Tensor::<B, 2>::from_floats(batch.states, &DEVICE).reshape([0, field_size, field_size]);
        let actions = Tensor::<B, 1, Int>::from_ints(
            batch.actions.into_iter().map(|&a| a as i32).collect(),
            &DEVICE,
        );
        let rewards = Tensor::<B, 1>::from_floats(
            batch.rewards.into_iter().map(|r| r as f32).collect(),
            &DEVICE,
        );

        let q_values = self
            .policy_net
            .forward(states)
            .gather(1, actions.unsqueeze_dim(1))
            .squeeze(1);
        let max_next_q_values = Tensor::<B, 1>::zeros([batch_size], &DEVICE).mask_where(
            non_terminal_mask,
            self.target_net.forward(next_states).max_dim(1).squeeze(1),
        );

        let discounted_expected_return = rewards + (max_next_q_values * self.gamma);

        let loss = self
            .loss
            .forward(q_values, discounted_expected_return, Reduction::Auto);
        let grads = GradientsParams::from_grads(loss.backward(), &self.policy_net);

        todo!()
        // self.optimizer.step(self.lr, self.policy_net, grads, None);
    }

    fn go(&mut self) {
        todo!()
    }
}
