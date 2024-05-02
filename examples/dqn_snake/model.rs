use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        Linear, LinearConfig, Relu,
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    field_size: usize,
    conv1_out: usize,
    conv2_out: usize,
    fc1_out: usize,
    fc2_out: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let k_size: usize = 3;
        let conv_output_size = self.conv2_out * (self.field_size - 2 * (k_size - 1)).pow(2);

        Model {
            conv1: Conv2dConfig::new([1, self.conv1_out], [k_size, k_size]).init(device),
            conv2: Conv2dConfig::new([self.conv1_out, self.conv2_out], [k_size, k_size])
                .init(device),
            fc1: LinearConfig::new(self.conv2_out * 16, self.fc1_out).init(device),
            fc2: LinearConfig::new(self.fc1_out, self.fc2_out).init(device),
            fc3: LinearConfig::new(self.fc2_out, 4).init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> Model<B> {
    /// In shape: `[num_batches, size, size]`
    ///
    /// Out shape: `[num_batches, 4]`
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let x = input.unsqueeze_dim(1);

        let x = self.conv1.forward(x);
        let x = self.conv2.forward(x);
        let x = self.activation.forward(x);

        let [n, c, h, w] = x.dims();
        let x = x.reshape([n, c * h * w]);
        let x = self.fc1.forward(x);
        let x = self.activation.forward(x);
        let x = self.fc2.forward(x);
        let x = self.activation.forward(x);
        let x = self.fc3.forward(x);

        x
    }
}
