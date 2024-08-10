/// Implemented RL algorithms
pub mod algo;

/// Implementations of strategies for time-decaying hyperparameters
pub mod decay;

/// Data structures
pub mod ds;

/// Environment
pub mod env;

pub mod agent;

/// Exploration policies
pub mod exploration;

/// Experience replay
pub mod memory;

/// Library traits
pub mod traits;

/// Probabilistic models
mod prob;

/// Training visualization TUI
#[cfg(feature = "viz")]
pub mod viz;

/// Testing environments
#[cfg(feature = "gym")]
pub mod gym;

mod util;
