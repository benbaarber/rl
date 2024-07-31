use crate::{env::Environment, memory::Exp};

pub trait Agent<E>
where
    E: Environment,
{
    fn act(&self, state: &E::State, actions: &[E::Action]) -> E::Action;
    fn learn(&mut self, exp: Exp<E>, next_actions: &[E::Action]);
}
