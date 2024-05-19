use rand::distributions::Distribution;

/// Trait for probabilistic models
///
/// ### Type parameters
/// - `T`: The type returned by sampling the distribution
/// - `O`: The type of the observation, contains information relevant to updating the model
pub trait ProbModel<T, O>: Distribution<T> {
    /// Initialize the model in a default state
    fn init() -> Self;

    /// Update the model given a new observation
    fn update(&mut self, observation: O);
}
