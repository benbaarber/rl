/// Implemented RL algorithms
pub mod algo;

/// Implementations of strategies for time-decaying hyperparameters
pub mod decay;

/// Data structures
pub mod ds;

/// Environment
pub mod env;

/// Exploration policies
pub mod exploration;

/// Experience replay
pub mod memory;

/// Testing environments
#[cfg(feature = "gym")]
pub mod gym;

mod util;
