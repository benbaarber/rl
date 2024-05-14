/// Exploration policy result
pub enum Choice {
    Explore,
    Exploit,
}

mod epsilon_greedy;
mod softmax;

pub use epsilon_greedy::EpsilonGreedy;
pub use softmax::Softmax;
