use std::fmt::Debug;

use burn::{
    module::AutodiffModule,
    optim::{adaptor::OptimizerAdaptor, AdamW, AdamWConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use nn::loss::{MseLoss, Reduction};

use crate::{
    decay,
    env::{Environment, ToTensor},
    exploration::{Choice, EpsilonGreedy},
    memory::{Exp, ReplayMemory},
};

/// A burn module used with a Deep Q network agent
///
/// ### Generics
/// - `B`: A burn backend
/// - `D`: The dimension of the input tensor
pub trait DQNModel<B: AutodiffBackend, const D: usize>: AutodiffModule<B> {
    /// Forward pass through the model
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

/// Configuration for the [`DQNAgent`] (see for information on generic types)
pub struct DQNAgentConfig<E, const D: usize, O>
where
    E: Environment,
{
    /// A [`ReplayMemory`] to store and sample the agent's past experiences
    pub memory: ReplayMemory<E>,
    /// The [`Optimizer`] to train the policy network with
    pub optimizer: O,
    /// The exploration policy, currently limited to epsilon greedy
    pub exploration: EpsilonGreedy<decay::Exponential>,
    /// The discount factor
    pub gamma: f32,
    /// The rate at which the target network's parameters are soft updated with the policy network's parameters
    pub tau: f32,
    /// The learning rate for the optimizer
    pub lr: f32,
}

type AdamWOptimizer<M, B> = OptimizerAdaptor<AdamW<<B as AutodiffBackend>::InnerBackend>, M, B>;

impl<B, M, E, const D: usize> Default for DQNAgentConfig<E, D, AdamWOptimizer<M, B>>
where
    B: AutodiffBackend,
    M: DQNModel<B, D>,
    E: Environment,
{
    fn default() -> Self {
        Self {
            memory: ReplayMemory::new(50000, 128),
            optimizer: AdamWConfig::new().init(),
            exploration: EpsilonGreedy::new(decay::Exponential::new(1e-3, 1.0, 0.05).unwrap()),
            gamma: 0.999,
            tau: 5e-3,
            lr: 1e-3,
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
/// - `D`: The dimension of the input
pub struct DQNAgent<B, M, E, const D: usize, O>
where
    B: AutodiffBackend,
    E: Environment,
{
    policy_net: Option<M>,
    target_net: Option<M>,
    device: &'static B::Device,
    memory: ReplayMemory<E>,
    optimizer: O,
    loss: MseLoss<B>, // TODO: make this generic
    exploration: EpsilonGreedy<decay::Exponential>,
    gamma: f32,
    tau: f32,
    lr: f32,
    total_steps: u32,
}

impl<B, M, E, const D: usize, O> DQNAgent<B, M, E, D, O>
where
    B: AutodiffBackend,
    M: DQNModel<B, D>,
    E: Environment,
    O: Optimizer<M, B>,
    Vec<E::State>: ToTensor<B, D, Float>,
    Vec<E::Action>: ToTensor<B, 2, Int>,
    E::Action: From<usize>,
    B::IntElem: TryInto<usize, Error: Debug>,
{
    /// Initialize a new `DQNAgent`
    ///
    /// ### Arguments
    /// - `model` A [`DQNModel`] to be used as the policy and target networks
    /// - `config` A [`DQNAgentConfig`] containing components and hyperparameters for the agent
    /// - `device` A static reference to the device used for the `model`
    pub fn new(model: M, config: DQNAgentConfig<E, D, O>, device: &'static B::Device) -> Self {
        let model_clone = model.clone();
        Self {
            policy_net: Some(model),
            target_net: Some(model_clone),
            device,
            memory: config.memory,
            optimizer: config.optimizer,
            loss: MseLoss::new(),
            exploration: config.exploration,
            gamma: config.gamma,
            tau: config.tau,
            lr: config.lr,
            total_steps: 0,
        }
    }

    /// Invoke the agent's policy along with the exploration strategy to choose an action from the given state
    fn act(&self, env: &E, state: E::State) -> E::Action {
        match self.exploration.choose(self.total_steps) {
            Choice::Explore => env.random_action(),
            Choice::Exploit => {
                let input = vec![state].to_tensor(self.device);
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

    /// Perform one DQN learning step
    fn learn(&mut self) {
        // Sample a batch of memories to train on
        let Some(batch) = self.memory.sample_zipped() else {
            return;
        };

        // Create a boolean mask for non-terminal next states so tensor shapes can match in the Bellman Equation
        let non_terminal_mask = Tensor::<B, 1, Bool>::from_bool(
            batch
                .next_states
                .iter()
                .map(Option::is_some)
                .collect::<Vec<_>>()
                .as_slice()
                .into(),
            self.device,
        )
        .unsqueeze_dim(1);

        // Tensor conversions
        let states = batch.states.to_tensor(self.device);
        let actions = batch.actions.to_tensor(self.device);
        let next_states = Tensor::<B, D>::cat(
            batch
                .next_states
                .into_iter()
                .flatten()
                .map(|ns| vec![ns].to_tensor(self.device)) // finish
                .collect::<Vec<_>>(),
            0,
        );
        let rewards =
            Tensor::<B, 1>::from_floats(batch.rewards.as_slice(), self.device).unsqueeze_dim(1);

        let policy_net = self.policy_net.take().unwrap();
        let target_net = self.target_net.take().unwrap();

        // Calculate the Q values of the chosen actions in each state
        let q_values = policy_net.forward(states).gather(1, actions);

        // Calculate the maximum Q values obtainable from each next state
        let expected_q_values = Tensor::<B, 2>::zeros([self.memory.batch_size, 1], self.device)
            .mask_where(
                non_terminal_mask,
                target_net.forward(next_states).max_dim(1).detach(),
            );

        let discounted_expected_return = rewards + (expected_q_values * self.gamma);

        // Calculate loss between actual Q values and expected return
        let loss = self
            .loss
            .forward(q_values, discounted_expected_return, Reduction::Mean);

        // Perform backpropagation on policy net
        let grads = GradientsParams::from_grads(loss.backward(), &policy_net);
        self.policy_net = Some(self.optimizer.step(self.lr.into(), policy_net, grads));

        // Perform a soft update on the parameters of the target network for stable convergence
        self.target_net = Some(target_net.soft_update(self.policy_net.as_ref().unwrap(), self.tau));
    }

    /// Deploy the `DQNAgent` into the environment for one episode
    pub fn go(&mut self, env: &mut E) {
        let mut next_state = Some(env.reset());

        while let Some(state) = next_state {
            let action = self.act(env, state.clone());
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
