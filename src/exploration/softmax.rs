use std::ops::AddAssign;

use burn::tensor::{self, backend::Backend, Tensor};
use rand::{
    distributions::{uniform::SampleUniform, Distribution, WeightedIndex},
    thread_rng,
};

use crate::decay::Decay;

/// Softmax exploration policy (also known as Boltzmann exploration) with time-decaying temperature
pub struct Softmax<D: Decay> {
    temperature: D,
}

impl<D: Decay> Softmax<D> {
    /// Initialize softmax exploration policy with a decay strategy
    pub fn new(decay: D) -> Self {
        Self { temperature: decay }
    }

    /// Invoke softmax exploration policy at time `t` with provided Q values
    pub fn choose(&self, t: f32, q_values: &[f32]) -> usize {
        let tau = self.temperature.evaluate(t);
        let exponentials = q_values.iter().map(|x| (x / tau).exp());
        let sum: f32 = exponentials.clone().sum();
        let weights = exponentials.map(|x| x / sum);
        let dist = WeightedIndex::new(weights).expect("`q_values` is not empty");
        dist.sample(&mut thread_rng())
    }

    /// Invoke softmax exploration policy at time `t` with provided 1D [Tensor] of Q values
    pub fn choose_from_tensor<B>(&self, t: f32, tensor: Tensor<B, 1>) -> usize
    where
        B: Backend,
        B::FloatElem: PartialOrd + SampleUniform,
        for<'a> B::FloatElem: AddAssign<&'a B::FloatElem>,
    {
        let tau = self.temperature.evaluate(t);
        let weights = tensor::activation::softmax(tensor / tau, 0)
            .iter_dim(0)
            .map(|t| t.into_scalar());
        let dist = WeightedIndex::new(weights).expect("`tensor` is not empty");
        dist.sample(&mut thread_rng())
    }
}
