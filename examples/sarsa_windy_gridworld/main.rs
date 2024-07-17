use std::{error::Error, fs, path::Path};

use agent::SarsaAgent;
use rl::gym::WindyGridworld;

mod agent;

const NUM_EPISODES: u16 = 500;

fn main() -> Result<(), Box<dyn Error>> {
    let path = Path::new("examples/sarsa_windy_gridworld");

    let mut env = WindyGridworld::new();
    let mut agent = SarsaAgent::new(0.1, 0.5, 1.0);

    fs::create_dir_all(path.join("out"))?;

    let mut wtr = csv::Writer::from_path(path.join("out/data.csv"))?;
    wtr.write_record(&["steps", "episodes"])?;

    for i in 0..NUM_EPISODES {
        agent.go(&mut env);
        wtr.write_record(&[&env.report["steps"].to_string(), &i.to_string()])?;
    }

    wtr.flush()?;

    std::process::Command::new("python")
        .arg(path.join("plot.py"))
        .spawn()?
        .wait()?;

    Ok(())
}
