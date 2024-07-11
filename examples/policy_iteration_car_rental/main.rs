use std::{error::Error, fs};

use agent::PolicyIterationAgent;
use env::CarRental;

mod agent;
mod env;

fn main() -> Result<(), Box<dyn Error>> {
    let mut env = CarRental::new();
    let mut agent = PolicyIterationAgent::new(0.9);

    let mut policies = Vec::with_capacity(5);

    for i in 0..5 {
        println!("Iteration {}", i + 1);
        agent.learn(&mut env, 1);
        let mut policy = agent.policy().clone().into_iter().collect::<Vec<_>>();
        policy.sort_unstable_by_key(|(k, _)| *k);

        policies.push(policy);
    }

    // Write data to CSV

    fs::create_dir_all("examples/policy_iteration_car_rental/out")?;

    let mut wtr = csv::Writer::from_path("examples/policy_iteration_car_rental/out/data.csv")?;

    for policy in policies {
        let actions = policy.iter().map(|e| e.1.to_string()).collect::<Vec<_>>();
        wtr.write_record(actions)?;
    }

    wtr.flush()?;

    // Plot data

    std::process::Command::new("python")
        .arg("examples/policy_iteration_car_rental/plot.py")
        .spawn()?
        .wait()?;

    Ok(())
}
