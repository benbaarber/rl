use burn::tensor::{backend::Backend, Tensor};

/// Upper confidence bound exploration policy
pub struct UCB<const A: usize> {
    c: f32,
    counter: [f32; A],
}

impl<const A: usize> UCB<A> {
    /// Initialize UCB policy with exploration parameter `c`
    ///
    /// A higher `c` value equates to more exploration. If unsure where to start, 1 is a good default value.
    pub fn new(c: f32) -> Self {
        Self {
            c,
            counter: [1.0; A],
        }
    }

    /// Invoke UCB policy at time `t` with provided Q values
    pub fn choose(&mut self, t: f32, q_values: &[f32; A]) -> usize {
        let k = self.c * t.log10().sqrt();
        let choice = q_values
            .into_iter()
            .enumerate()
            .map(|(i, x)| (i, x + k * self.counter[i].powf(-0.5)))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .expect("`q_values` is not empty");

        self.counter[choice] += 1.0;
        choice
    }

    /// (not yet implemented) Invoke UCB policy at time `t` with provided 1D [Tensor] of Q values
    pub fn _choose_from_tensor<B: Backend>(&self, t: f32, tensor: Tensor<B, 1>) -> usize {
        todo!()
    }
}
