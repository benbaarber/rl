use std::ops::AddAssign;

use burn::tensor::{self, backend::Backend, Tensor};
use rand::{
    distributions::{uniform::SampleUniform, Distribution, WeightedIndex},
    seq::SliceRandom,
    thread_rng,
};

use crate::decay::Decay;

/// Softmax exploration policy (also known as Boltzmann exploration) with time-decaying temperature
pub struct Softmax<D: Decay> {
    temperature: D,
}

impl<D: Decay> Softmax<D> {
    pub fn new(decay: D) -> Self {
        Self { temperature: decay }
    }

    pub fn choose(&self, t: f32, q_values: &[f32]) -> usize {
        let tau = self.temperature.evaluate(t);
        let exponentials = q_values.into_iter().map(|x| (x / tau).exp());
        let sum: f32 = exponentials.clone().sum();
        let weights = exponentials.map(|x| x / sum);
        let dist = WeightedIndex::new(weights).expect("`q_values` is not empty");
        dist.sample(&mut thread_rng())
    }

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
