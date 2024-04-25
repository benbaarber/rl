/// Implemented RL algorithms
pub mod algo;

/// Data structures
pub mod ds;

/// Environment
pub mod env;

/// Exploration policies
pub mod exploration;

/// Experience replay
pub mod memory;

/// Test environments inspired by python [gymnasium](https://gymnasium.farama.org/)
#[cfg(feature = "gym")]
pub mod gym;

mod util;
