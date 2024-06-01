#![allow(unused)]
use std::marker::PhantomData;

use crate::{env::Environment, memory::Exp, prob::ProbModel};

/// Thompson sampling exploration policy (also known as probability matching)
///
/// ### Type parameters
/// - `E`: Environment
/// - `M`: The type of probabilistic model to model the reward for each action with
/// - `A`: The size of the action space
pub struct Thompson<E, M, const A: usize>
where
    E: Environment,
    M: ProbModel<usize, Exp<E>>,
{
    models: [M; A],
    phantom: PhantomData<E>,
}

impl<E, M, const A: usize> Thompson<E, M, A>
where
    E: Environment,
    M: ProbModel<usize, Exp<E>>,
{
    pub fn new() -> Self {
        Self {
            models: core::array::from_fn(|_| M::init()),
            phantom: PhantomData,
        }
    }
}
