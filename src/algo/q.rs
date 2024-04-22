use std::collections::HashMap;

use crate::env::Environment;

pub struct QAgent<E: Environment> {
    environment: E,
    q_table: HashMap<(E::State, E::Action), f64>,
    alpha: f64, // learning rate
    gamma: f64, // discount factor
}

impl<E: Environment> QAgent<E> {
    pub fn new(env: E, alpha: f64, gamma: f64) -> Self {
        Self {
            environment: env,
            q_table: HashMap::new(),
            alpha,
            gamma,
        }
    }
}
