pub mod q;

pub trait Agent {
    /// Start the RL loop
    fn go(&mut self);
}
