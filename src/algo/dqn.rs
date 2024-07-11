use burn::{
    grad_clipping::GradientClippingConfig,
    module::AutodiffModule,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use nn::loss::{MseLoss, Reduction};

use crate::{
    decay::{self, Decay},
    env::Environment,
    exploration::{Choice, EpsilonGreedy},
    memory::{Exp, Memory, PrioritizedReplayMemory, ReplayMemory},
    traits::ToTensor,
};

/// A burn module used with a Deep Q network agent
///
/// ### Generics
/// - `B` - A burn backend
/// - `D` - The dimension of the input tensor
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

/// Configuration for the [`DQNAgent`]
#[derive(Debug, Clone)]
pub struct DQNAgentConfig<D> {
    /// The capacity of the replay memory
    ///
    /// **Default:** `16384`
    pub memory_capacity: usize,
    /// The size of batches to be sampled from the replay memory
    ///
    /// **Default:** `128`
    pub memory_batch_size: usize,
    /// Use [`PrioritizedReplayMemory`] instead of the base [`ReplayMemory`]
    ///
    /// **Default:** `false`
    pub use_prioritized_memory: bool,
    /// The number of episode this agent is going to be trained for
    ///
    /// This value is only used if `use_prioritized_replay` is set to true
    ///
    /// **Default:** `500`
    pub num_episodes: usize,
    /// The prioritization exponent, which affects degree of prioritization used in the stochastic sampling of experiences (see [`PrioritizedReplayMemory`])
    ///
    /// This value is only used if `use_prioritized_replay` is set to true
    ///
    /// **Default:** `0.7`
    pub prioritized_memory_alpha: f32,
    /// The initial value for beta, the importance sampling exponent, which is annealed from β<sub>0</sub> to 1 to apply IS weights to the temporal difference errors
    /// (see [`PrioritizedReplayMemory`])
    ///
    /// This value is only used if `use_prioritized_replay` is set to true
    ///
    /// **Default:** `0.5`
    pub prioritized_memory_beta_0: f32,
    // /// The [`Optimizer`] to train the policy network with
    // pub optimizer: O,
    /// The epsilon decay strategy
    ///
    /// **Default:** [`Exponential`](decay::Exponential) decay with decay rate `1e-3`, start value `1.0`, and end value `0.05`
    pub epsilon_decay_strategy: D,
    /// The discount factor
    ///
    /// **Default:** `0.999`
    pub gamma: f32,
    /// The interval at which to perform soft updates on the target network
    ///
    /// **Default:** `1`
    pub target_update_interval: usize,
    /// The rate at which the target network's parameters are soft updated with the policy network's parameters
    ///
    /// **Default:** `5e-3`
    pub tau: f32,
    /// The learning rate for the optimizer
    ///
    /// **Default:** `1e-3`
    pub lr: f32,
}

// type AdamWOptimizer<M, B> = OptimizerAdaptor<AdamW<<B as AutodiffBackend>::InnerBackend>, M, B>;

impl Default for DQNAgentConfig<decay::Exponential> {
    fn default() -> Self {
        Self {
            memory_capacity: 16384,
            memory_batch_size: 128,
            use_prioritized_memory: false,
            num_episodes: 500,
            prioritized_memory_alpha: 0.7,
            prioritized_memory_beta_0: 0.5,
            // optimizer: AdamWConfig::new().init(),
            epsilon_decay_strategy: decay::Exponential::new(1e-3, 1.0, 0.05).unwrap(),
            gamma: 0.999,
            target_update_interval: 1,
            tau: 5e-3,
            lr: 1e-3,
        }
    }
}

/// A Deep Q Network agent
///
/// ### Generics
/// - `B` - A burn backend
/// - `M` - The [`DQNModel`] used for the policy and target networks
/// - `E` - The [`Environment`] in which the agent will learn
///     - The environment's action space must be discrete, since the policy network produces a Q value for each action.
///     - The state and action types' implementations of [`Clone`] should be very lightweight, as they are cloned often.
///       Ideally, both types are [`Copy`].
/// - `DEC` - The decay strategy for epsilon-greedy exploration
/// - `D` - The dimension of the input
///
/// A generic optimizer will be added when burn v0.14.0 releases, until then the [`AdamW`](burn::optim::AdamW) optimizer will be used
#[derive(Debug, Clone)]
pub struct DQNAgent<B, M, E, DEC, const D: usize>
where
    B: AutodiffBackend,
    E: Environment,
    DEC: Decay,
{
    policy_net: Option<M>,
    target_net: Option<M>,
    device: &'static B::Device,
    memory: Memory<E>,
    // optimizer: O,
    exploration: EpsilonGreedy<DEC>,
    gamma: f32,
    target_update_interval: usize,
    tau: f32,
    lr: f32,
    total_steps: u32,
    episodes_elapsed: usize,
}

impl<B, M, E, DEC, const D: usize> DQNAgent<B, M, E, DEC, D>
where
    B: AutodiffBackend<FloatElem = f32, IntElem = i32>,
    M: DQNModel<B, D>,
    E: Environment,
    DEC: Decay,
    // O: Optimizer<M, B>,
    Vec<E::State>: ToTensor<B, D, Float>,
    Vec<E::Action>: ToTensor<B, 2, Int>,
    E::Action: From<usize>,
{
    /// Initialize a new `DQNAgent`
    ///
    /// ### Arguments
    /// - `model` A [`DQNModel`] to be used as the policy and target networks
    /// - `config` A [`DQNAgentConfig`] containing components and hyperparameters for the agent
    /// - `device` A static reference to the device used for the `model`
    pub fn new(model: M, config: DQNAgentConfig<DEC>, device: &'static B::Device) -> Self {
        let model_clone = model.clone();
        let memory = if config.use_prioritized_memory {
            Memory::Prioritized(PrioritizedReplayMemory::new(
                config.memory_capacity,
                config.memory_batch_size,
                config.prioritized_memory_alpha,
                config.prioritized_memory_beta_0,
                config.num_episodes,
            ))
        } else {
            Memory::Base(ReplayMemory::new(
                config.memory_capacity,
                config.memory_batch_size,
            ))
        };

        Self {
            policy_net: Some(model),
            target_net: Some(model_clone),
            device,
            memory,
            // optimizer: config.optimizer,
            exploration: EpsilonGreedy::new(config.epsilon_decay_strategy),
            gamma: config.gamma,
            target_update_interval: config.target_update_interval,
            tau: config.tau,
            lr: config.lr,
            total_steps: 0,
            episodes_elapsed: 0,
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
    fn learn(&mut self, optimizer: &mut impl Optimizer<M, B>) {
        // Sample a batch of memories to train on
        let Memory::Base(memory) = &mut self.memory else {
            return;
        };
        let Some(batch) = memory.sample_zipped() else {
            return;
        };
        let batch_size = memory.batch_size;

        // Create a boolean mask for non-terminal next states so tensor shapes can match in the Bellman Equation
        let non_terminal_mask = batch
            .next_states
            .iter()
            .map(Option::is_some)
            .collect::<Vec<_>>()
            .to_tensor(self.device)
            .unsqueeze_dim(1);

        // Tensor conversions
        let states = batch.states.to_tensor(self.device);
        let actions = batch.actions.to_tensor(self.device);
        let next_states = batch
            .next_states
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .to_tensor(self.device);
        let rewards = batch.rewards.to_tensor(self.device).unsqueeze_dim(1);

        let policy_net = self.policy_net.take().unwrap();
        let target_net = self.target_net.take().unwrap();

        // Compute the Q values of the chosen actions in each state
        let q_values = policy_net.forward(states).gather(1, actions);

        // Compute the maximum Q values obtainable from each next state
        let expected_q_values = Tensor::zeros([batch_size, 1], self.device).mask_where(
            non_terminal_mask,
            target_net.forward(next_states).max_dim(1).detach(),
        );

        let discounted_expected_return = rewards + (expected_q_values * self.gamma);

        // Compute loss (mean sqared temporal difference error)
        let loss = MseLoss::new().forward(q_values, discounted_expected_return, Reduction::Mean);

        // Perform backpropagation on policy net
        let grads = GradientsParams::from_grads(loss.backward(), &policy_net);
        self.policy_net = Some(optimizer.step(self.lr.into(), policy_net, grads));

        // Perform a periodic soft update on the parameters of the target network for stable convergence
        self.target_net = if self.episodes_elapsed % self.target_update_interval == 0 {
            Some(target_net.soft_update(self.policy_net.as_ref().unwrap(), self.tau))
        } else {
            Some(target_net)
        };
    }

    fn learn_prioritized(&mut self, optimizer: &mut impl Optimizer<M, B>) {
        // Sample a batch of memories to train on
        let Memory::Prioritized(memory) = &mut self.memory else {
            return;
        };
        let Some((batch, weights, indices)) = memory.sample_zipped(self.episodes_elapsed) else {
            return;
        };
        let batch_size = memory.batch_size;

        // Create a boolean mask for non-terminal next states so tensor shapes can match in the Bellman Equation
        let non_terminal_mask = batch
            .next_states
            .iter()
            .map(Option::is_some)
            .collect::<Vec<_>>()
            .to_tensor(self.device)
            .unsqueeze_dim(1);

        // Tensor conversions
        let states = batch.states.to_tensor(self.device);
        let actions = batch.actions.to_tensor(self.device);
        let next_states = batch
            .next_states
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .to_tensor(self.device);
        let rewards = batch.rewards.to_tensor(self.device).unsqueeze_dim(1);

        let policy_net = self.policy_net.take().unwrap();
        let target_net = self.target_net.take().unwrap();

        // Compute the Q values of the chosen actions in each state
        let q_values = policy_net.forward(states).gather(1, actions);

        // Compute the maximum Q values obtainable from each next state
        let expected_q_values = Tensor::zeros([batch_size, 1], self.device).mask_where(
            non_terminal_mask,
            target_net.forward(next_states).max_dim(1).detach(),
        );

        let discounted_expected_return = rewards + (expected_q_values * self.gamma);

        // Compute temporal difference errors
        let tde: Tensor<B, 1> = (discounted_expected_return - q_values).squeeze(1);

        // Update priorities of sampled experiences
        let td_errors = tde.to_data().value;
        memory.update_priorities(&indices, &td_errors);

        // Apply importance sampling weights from prioritized memory replay and compute mean squared weighted TD error
        let weights = weights.to_tensor(self.device);
        let loss = (weights * tde.powf_scalar(2.0)).mean();

        // Perform backpropagation on policy net
        let grads = GradientsParams::from_grads(loss.backward(), &policy_net);
        self.policy_net = Some(optimizer.step(self.lr.into(), policy_net, grads));

        // Perform a periodic soft update on the parameters of the target network for stable convergence
        self.target_net = if self.episodes_elapsed % self.target_update_interval == 0 {
            Some(target_net.soft_update(self.policy_net.as_ref().unwrap(), self.tau))
        } else {
            Some(target_net)
        };
    }

    /// Deploy the `DQNAgent` into the environment for one episode
    pub fn go(&mut self, env: &mut E) {
        let mut optimizer = AdamWConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Value(100.0)))
            .init();
        let mut next_state = Some(env.reset());

        while let Some(state) = next_state {
            let action = self.act(env, state.clone());
            let (next, reward) = env.step(action.clone());
            next_state = next;

            let exp = Exp {
                state,
                action,
                reward,
                next_state: next_state.clone(),
            };

            match &mut self.memory {
                Memory::Base(memory) => {
                    memory.push(exp);
                    self.learn(&mut optimizer);
                }
                Memory::Prioritized(memory) => {
                    memory.push(exp);
                    self.learn_prioritized(&mut optimizer);
                }
            }

            self.total_steps += 1;
        }

        self.episodes_elapsed += 1;
    }
}
