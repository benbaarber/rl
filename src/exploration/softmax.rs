use crate::{assert_interval, decay::Decay};

/// Softmax exploration policy (also known as Boltzmann exploration) with time-decaying temperature
pub struct Softmax<D: Decay> {
    temperature: D,
}
