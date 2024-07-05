pub mod action_occurrence;
pub mod q_table;
pub mod ucb;

/// A trait for state and action types that can be used as keys in a [`HashMap`](std::collections::HashMap)
pub trait Hashable: Copy + Eq + std::hash::Hash {}

impl<T> Hashable for T where T: Copy + Eq + std::hash::Hash {}
