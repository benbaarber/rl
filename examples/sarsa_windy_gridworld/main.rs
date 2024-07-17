use std::{error::Error, fs};

use agent::SarsaAgent;
use env::WindyGridworld;

mod agent;
mod env;

const NUM_EPISODES: u16 = 500;

fn main() -> Result<(), Box<dyn Error>> {
    let mut env = WindyGridworld::new();
    let mut agent = SarsaAgent::new(0.1, 0.5, 1.0);

    fs::create_dir_all("examples/sarsa_windy_gridworld/out")?;

    let mut wtr = csv::Writer::from_path("examples/sarsa_windy_gridworld/out/data.csv")?;
    wtr.write_record(&["steps", "episodes"])?;

    for i in 0..NUM_EPISODES {
        agent.go(&mut env);
        wtr.write_record(&[&env.report["steps"].to_string(), &i.to_string()])?;
    }

    wtr.flush()?;

    std::process::Command::new("python")
        .arg("examples/sarsa_windy_gridworld/plot.py")
        .spawn()?
        .wait()?;

    Ok(())
}
