use std::fmt::Debug;

use burn::{
    module::AutodiffModule,
    optim::{GradientsParams, Optimizer},
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

/// A Deep Q Network agent
///
/// ### Generics
/// - `B`: A burn backend
/// - `M`: The [`DQNModel`] used for the policy and target networks
/// - `E`: The [`Environment`] in which the agent will learn
pub struct DQNAgent<B, M, E, O, const D: usize>
where
    B: AutodiffBackend,
    M: DQNModel<B, D>,
    E: Environment,
    O: Optimizer<M, B>,
{
    policy_net: Option<M>,
    target_net: Option<M>,
    device: &'static B::Device,
    memory: ReplayMemory<E>,
    optimizer: O,
    loss: MseLoss<B>,
    exploration: EpsilonGreedy<decay::Exponential>,
    gamma: f32,
    tau: f32,
    lr: f32,
    total_steps: u32,
}

impl<B, M, E, O, const D: usize> DQNAgent<B, M, E, O, D>
where
    B: AutodiffBackend,
    M: DQNModel<B, D>,
    E: Environment,
    O: Optimizer<M, B>,
    for<'a> &'a E::State: Into<Data<B::FloatElem, D>> + Debug,
    for<'a> &'a [E::State]: Into<Data<B::FloatElem, D>> + Debug,
    for<'a> &'a E::Action: Into<Data<B::FloatElem, 2>> + Debug,
    for<'a> &'a [E::Action]: Into<Data<B::IntElem, 2>> + Debug,
    E::Action: From<usize>,
    B::IntElem: TryInto<usize, Error: Debug>,
{
    pub fn new() -> Self {
        todo!()
    }

    fn act(&self, state: &E::State) -> E::Action {
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

    fn learn<const BATCH_SIZE: usize>(&mut self) -> Option<()> {
        let batch = self.memory.sample_zipped::<BATCH_SIZE>()?;

        let mut non_terminal_mask = [false; BATCH_SIZE];
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
                .map(|ns| Tensor::from_data(&ns, self.device))
                .collect::<Vec<_>>(),
            0,
        );

        let states = Tensor::from_data(batch.states.as_slice(), self.device);
        let actions = Tensor::from_data(batch.actions.as_slice(), self.device);
        let rewards = Tensor::<B, 1>::from_floats(batch.rewards, self.device).unsqueeze_dim(1);

        let policy_net = self.policy_net.take().unwrap();
        let target_net = self.target_net.take().unwrap();

        let q_values = policy_net.forward(states).gather(1, actions);

        let expected_q_values = Tensor::<B, 2>::zeros([BATCH_SIZE, 1], self.device).mask_where(
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

        Some(())
    }

    pub fn go(&mut self, env: &mut E) {
        let mut next_state = Some(env.reset());

        while let Some(state) = next_state {
            let action = self.act(&state);
            let (next, reward) = env.step(action.clone()); // TODO: probably a better way than cloning
            next_state = next;

            self.memory.push(Exp {
                state,
                action,
                next_state: next_state.clone(), // TODO: probably a better way than cloning
                reward,
            });

            // TODO: Figure out how to pass batch size down cleanly
            let _ = self.learn::<128>();

            self.total_steps += 1;
        }
    }
}
