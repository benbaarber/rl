/// Exploration policy result
pub enum Choice {
    Explore,
    Exploit,
}

mod epsilon_greedy;
mod softmax;
mod thompson;
mod ucb;

pub use epsilon_greedy::EpsilonGreedy;
pub use softmax::Softmax;
// pub use thompson::Thompson;
pub use ucb::UCB;
