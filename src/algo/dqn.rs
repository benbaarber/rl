use std::{fmt::Debug, marker::PhantomData};

use burn::{
    module::AutodiffModule,
    optim::{adaptor::OptimizerAdaptor, AdamW, AdamWConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use nn::loss::{MseLoss, Reduction};

use crate::{
    decay,
    env::Environment,
    exploration::{Choice, EpsilonGreedy},
    memory::{Exp, ReplayMemory},
};

/// A burn module used with a Deep Q network agent
///
/// ### Generics
/// - `B`: A burn backend
/// - `D`: The dimension of the input tensor
pub trait DQNModel<B: AutodiffBackend, const D: usize>: AutodiffModule<B> {
    /// Run the forward pass
    fn forward(&self, input: Tensor<B, D>) -> Tensor<B, 2>;

    /// Soft update the parameters of the target network
    ///
    /// θ′ ← τθ + (1 − τ)θ′
    ///
    /// ```ignore
    /// target_net = target_net.soft_update(policy_net, tau);
    /// ```
    fn soft_update(self, other: &Self, tau: f32) -> Self;
}

pub trait DQNState {}

pub struct DQNAgentConfig<B, M, E, O, const D: usize, const BS: usize>
where
    B: AutodiffBackend,
    E: Environment,
{
    pub memory: ReplayMemory<BS, E>,
    pub optimizer: O,
    pub loss: MseLoss<B>,
    pub exploration: EpsilonGreedy<decay::Exponential>,
    pub gamma: f32,
    pub tau: f32,
    pub lr: f32,
    pub phantom: PhantomData<M>,
}

type AdamWOptimizer<M, B> = OptimizerAdaptor<AdamW<<B as AutodiffBackend>::InnerBackend>, M, B>;

impl<B, M, E, const D: usize, const BS: usize> Default
    for DQNAgentConfig<B, M, E, AdamWOptimizer<M, B>, D, BS>
where
    B: AutodiffBackend,
    M: DQNModel<B, D>,
    E: Environment,
{
    fn default() -> Self {
        Self {
            memory: ReplayMemory::new(50000),
            optimizer: AdamWConfig::new().init::<B, M>(),
            loss: MseLoss::new(),
            exploration: EpsilonGreedy::new(decay::Exponential::new(1e-3, 1.0, 0.05).unwrap()),
            gamma: 0.999,
            tau: 5e-3,
            lr: 1e-3,
            phantom: PhantomData,
        }
    }
}

/// A Deep Q Network agent
///
/// ### Generics
/// - `B`: A burn backend
/// - `M`: The [`DQNModel`] used for the policy and target networks
/// - `E`: The [`Environment`] in which the agent will learn
///     - The environment's action space must be discrete, since the policy network produces a Q value for each action.
///     - The state and action types' implementations of [`Clone`] should be very lightweight, as they are cloned often.
///       Ideally, both types are [`Copy`].
/// - `O`: An [`Optimizer`]
pub struct DQNAgent<B, M, E, O, const D: usize, const BS: usize>
where
    B: AutodiffBackend,
    E: Environment,
{
    policy_net: Option<M>,
    target_net: Option<M>,
    device: &'static B::Device,
    memory: ReplayMemory<BS, E>,
    optimizer: O,
    loss: MseLoss<B>,
    exploration: EpsilonGreedy<decay::Exponential>,
    gamma: f32,
    tau: f32,
    lr: f32,
    total_steps: u32,
}

impl<B, M, E, O, const D: usize, const BS: usize> DQNAgent<B, M, E, O, D, BS>
where
    B: AutodiffBackend,
    M: DQNModel<B, D>,
    E: Environment,
    O: Optimizer<M, B>,
    E::State: Into<Data<B::FloatElem, D>>,
    [E::State; BS]: Into<Data<B::FloatElem, D>>,
    E::Action: Into<Data<B::FloatElem, 2>>,
    [E::Action; BS]: Into<Data<B::IntElem, 2>>,
    E::Action: From<usize>,
    B::IntElem: TryInto<usize, Error: Debug>,
{
    pub fn new(
        model: M,
        device: &'static B::Device,
        config: DQNAgentConfig<B, M, E, O, D, BS>,
    ) -> Self {
        let model_clone = model.clone();
        Self {
            policy_net: Some(model),
            target_net: Some(model_clone),
            device,
            memory: config.memory,
            optimizer: config.optimizer,
            loss: config.loss,
            exploration: config.exploration,
            gamma: config.gamma,
            tau: config.tau,
            lr: config.lr,
            total_steps: 0,
        }
    }

    fn act(&self, state: E::State) -> E::Action {
        match self.exploration.choose(self.total_steps) {
            Choice::Explore => E::random_action(),
            Choice::Exploit => {
                let input = Tensor::from_data(state, self.device);
                let output = self
                    .policy_net
                    .as_ref()
                    .unwrap()
                    .forward(input)
                    .argmax(1)
                    .into_scalar();
                E::Action::from(output.try_into().unwrap())
            }
        }
    }

    fn learn(&mut self) {
        let Some(batch) = self.memory.sample_zipped() else {
            return;
        };

        let mut non_terminal_mask = [false; BS];
        for (i, s) in batch.next_states.iter().enumerate() {
            if s.is_some() {
                non_terminal_mask[i] = true;
            }
        }

        let non_terminal_mask =
            Tensor::<B, 1, Bool>::from_bool(non_terminal_mask.into(), self.device).unsqueeze_dim(1);

        let next_states = Tensor::<B, D>::cat(
            batch
                .next_states
                .into_iter()
                .flatten()
                .map(|ns| Tensor::from_data(ns, self.device))
                .collect::<Vec<_>>(),
            0,
        );

        let states = Tensor::from_data(batch.states, self.device);
        let actions = Tensor::from_data(batch.actions, self.device);
        let rewards = Tensor::<B, 1>::from_floats(batch.rewards, self.device).unsqueeze_dim(1);

        let policy_net = self.policy_net.take().unwrap();
        let target_net = self.target_net.take().unwrap();

        let q_values = policy_net.forward(states).gather(1, actions);

        let expected_q_values = Tensor::<B, 2>::zeros([BS, 1], self.device).mask_where(
            non_terminal_mask,
            target_net.forward(next_states).max_dim(1).detach(),
        );

        let discounted_expected_return = rewards + (expected_q_values * self.gamma);

        let loss = self
            .loss
            .forward(q_values, discounted_expected_return, Reduction::Mean);
        let grads = GradientsParams::from_grads(loss.backward(), &policy_net);

        self.policy_net = Some(self.optimizer.step(self.lr.into(), policy_net, grads));
        self.target_net = Some(target_net.soft_update(self.policy_net.as_ref().unwrap(), self.tau));
    }

    pub fn go(&mut self, env: &mut E) {
        let mut next_state = Some(env.reset());

        while let Some(state) = next_state {
            let action = self.act(state.clone());
            let (next, reward) = env.step(action.clone());
            next_state = next;

            self.memory.push(Exp {
                state,
                action,
                next_state: next_state.clone(),
                reward,
            });

            self.learn();
            self.total_steps += 1;
        }
    }
}
