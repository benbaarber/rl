use std::{collections::HashMap, ops::Range};

use rand::Rng;
use rand_distr::Distribution;
use rl::env::{DiscreteActionSpace, DiscreteStateSpace, Environment};
use statrs::distribution::{Discrete, Poisson};

type PMFCache = HashMap<(i32, i32), f64>;

pub struct Outcome {
    pub next_state: [i32; 2],
    pub reward: f32,
    pub prob: f32,
}

struct Location {
    inventory: i32,
    requests: Poisson,
    returns: Poisson,
}

impl Location {
    fn new(initial_inventory: i32, expected_requests: f64, expected_returns: f64) -> Self {
        Self {
            inventory: initial_inventory,
            requests: Poisson::new(expected_requests).unwrap(),
            returns: Poisson::new(expected_returns).unwrap(),
        }
    }
}

pub struct CarRental {
    locations: [Location; 2],
    pmf_cache: PMFCache,
}

impl CarRental {
    pub fn new() -> Self {
        Self {
            locations: [Location::new(10, 3.0, 3.0), Location::new(10, 4.0, 2.0)],
            pmf_cache: generate_pmf_cache(&[2, 3, 4], 0..10),
        }
    }

    fn get_state(&self) -> [i32; 2] {
        [self.locations[0].inventory, self.locations[1].inventory]
    }

    pub fn dynamics(&mut self, state: [i32; 2], action: i32) -> Vec<Outcome> {
        let mut outcomes = Vec::with_capacity(10000);
        let (loc1_req_lambda, loc2_req_lambda, loc1_ret_lambda, loc2_ret_lambda) = (3, 4, 3, 2);
        for loc1_req in 0..10 {
            let loc1_req_p = self.pmf_cache[&(loc1_req_lambda, loc1_req)];
            for loc2_req in 0..10 {
                let loc2_req_p = self.pmf_cache[&(loc2_req_lambda, loc2_req)];
                for loc1_ret in 0..10 {
                    let loc1_ret_p = self.pmf_cache[&(loc1_ret_lambda, loc1_ret)];
                    for loc2_ret in 0..10 {
                        let loc2_ret_p = self.pmf_cache[&(loc2_ret_lambda, loc2_ret)];
                        let (next_state, reward) =
                            transition(&state, action, loc1_req, loc2_req, loc1_ret, loc2_ret);
                        outcomes.push(Outcome {
                            next_state,
                            reward,
                            prob: (loc1_req_p * loc2_req_p * loc1_ret_p * loc2_ret_p) as f32,
                        });
                    }
                }
            }
        }

        outcomes
    }
}

impl Environment for CarRental {
    /// Inventory of cars at each location
    type State = [i32; 2];
    /// Number of cars moved from location 1 to location 2
    type Action = i32;

    fn random_action(&self) -> Self::Action {
        rand::thread_rng().gen_range(-5..=5)
    }

    fn step(&mut self, action: Self::Action) -> (Option<Self::State>, f32) {
        let mut net = 0;

        // Move cars
        let [i1, i2] = self.get_state();
        let action = action.clamp(-i2, i1);
        self.locations[0].inventory -= action;
        self.locations[1].inventory += action;
        net -= action.abs() * 2;

        let mut rng = rand::thread_rng();
        for location in &mut self.locations {
            location.inventory += location.returns.sample(&mut rng).round() as i32;
            let requests = location.requests.sample(&mut rng).round() as i32;
            let fulfilled = requests.min(location.inventory);
            location.inventory -= fulfilled;
            net += fulfilled * 10;
            location.inventory = location.inventory.min(19);
        }

        dbg!(
            net,
            self.get_state(),
            action,
            self.locations[0].inventory,
            self.locations[1].inventory
        );

        (Some(self.get_state()), net as f32)
    }

    fn reset(&mut self) -> Self::State {
        for location in &mut self.locations {
            location.inventory = 10;
        }

        self.get_state()
    }
}

impl DiscreteStateSpace for CarRental {
    fn states(&self) -> Vec<Self::State> {
        (0..400).map(|i| [i as i32 / 20, i as i32 % 20]).collect()
    }
}

impl DiscreteActionSpace for CarRental {
    fn actions(&self) -> Vec<Self::Action> {
        (-5..=5).collect()
    }
}

fn generate_pmf_cache(lambdas: &[i32], range: Range<i32>) -> HashMap<(i32, i32), f64> {
    let mut cache = HashMap::new();

    for lambda in lambdas {
        let poisson = Poisson::new(*lambda as f64).unwrap();
        for value in range.clone() {
            let key = (*lambda, value);
            let p = poisson.pmf(value as u64);
            cache.insert(key, p);
        }
    }

    cache
}

fn transition(
    state: &[i32; 2],
    action: i32,
    loc1_req: i32,
    loc2_req: i32,
    loc1_ret: i32,
    loc2_ret: i32,
) -> ([i32; 2], f32) {
    let mut next_state = state.clone();
    let mut reward = 0.0;

    // Move cars
    let action = action.clamp(-state[1], state[0]);
    next_state[0] -= action;
    next_state[1] += action;
    reward -= action.abs() as f32 * 2.0;

    for (loc, req, ret) in [(0, loc1_req, loc1_ret), (1, loc2_req, loc2_ret)] {
        let mut inv = next_state[loc];
        inv += ret;
        let fulfilled = req.min(inv);
        inv -= fulfilled;
        reward += fulfilled as f32 * 10.0;
        inv = inv.min(19);
        next_state[loc] = inv;
    }

    (next_state, reward)
}
