pub mod q;

pub use q::QAgent;

pub trait Agent {
    /// Start the RL loop
    fn go(&mut self);
}
