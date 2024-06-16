use burn::{
    module::Param,
    prelude::*,
    tensor::{activation::relu, backend::AutodiffBackend},
};
use nn::{Linear, LinearConfig};
use rl::algo::dqn::DQNModel;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    fc1_out: usize,
    fc2_out: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            fc1: LinearConfig::new(4, self.fc1_out).init(device),
            fc2: LinearConfig::new(self.fc1_out, self.fc2_out).init(device),
            fc3: LinearConfig::new(self.fc2_out, 2).init(device),
        }
    }
}

impl<B: AutodiffBackend> DQNModel<B, 2> for Model<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = relu(self.fc1.forward(input));
        let x = relu(self.fc2.forward(x));
        let x = self.fc3.forward(x);

        x
    }

    fn soft_update(self, other: &Self, tau: f32) -> Self {
        Self {
            fc1: soft_update_linear(self.fc1, &other.fc1, tau),
            fc2: soft_update_linear(self.fc2, &other.fc2, tau),
            fc3: soft_update_linear(self.fc3, &other.fc3, tau),
        }
    }
}

fn soft_update_tensor<B: Backend, const D: usize>(
    this: Param<Tensor<B, D>>,
    that: &Param<Tensor<B, D>>,
    tau: f32,
) -> Param<Tensor<B, D>> {
    this.map(|tensor| tensor * (1.0 - tau) + that.val() * tau)
}

fn soft_update_linear<B: Backend>(mut this: Linear<B>, that: &Linear<B>, tau: f32) -> Linear<B> {
    this.weight = soft_update_tensor(this.weight, &that.weight, tau);
    this.bias = match (this.bias, &that.bias) {
        (Some(b1), Some(b2)) => Some(soft_update_tensor(b1, b2, tau)),
        _ => None,
    };

    this
}
