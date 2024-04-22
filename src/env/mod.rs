pub trait Environment {
    type State: Clone;
    type Action: Clone;

    fn react(&mut self, action: Self::Action) -> (Self::State, f64);
    fn reward(&self, state: Self::State, next_state: Self::State, action: Self::Action) -> f64;
}
