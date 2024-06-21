mod base;
mod exp;
mod prioritized;

pub use base::ReplayMemory;
pub use exp::*;
pub use prioritized::PrioritizedReplayMemory;

use crate::env::Environment;

pub(crate) enum Memory<E: Environment> {
    Base(ReplayMemory<E>),
    Prioritized(PrioritizedReplayMemory<E>),
}
