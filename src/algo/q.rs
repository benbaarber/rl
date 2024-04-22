use crate::{env::Environment, memory::Memory};

pub struct QAgent<E: Environment, M: Memory> {
    environment: E,
    state: E::State,
    memory: M,
}
