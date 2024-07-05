use std::error::Error;

use rl::{
    algo::tabular::{
        action_occurrence::{ActionOccurrenceAgent, ActionOccurrenceAgentConfig},
        ucb::{UCBAgent, UCBAgentConfig},
    },
    decay,
    gym::KArmedBandit,
};

const STEP_LIMIT: usize = 1000;

fn main() -> Result<(), Box<dyn Error>> {
    let powers = (-7..=2).map(|x| 2f64.powi(x)).collect::<Vec<_>>();
    let e_greedy_param_values = &powers[0..6];
    let ucb_param_values = &powers[3..];
    let goi_param_values = &powers[5..];

    let mut env = KArmedBandit::<10>::new(STEP_LIMIT, false);

    // Epsilon greedy
    let mut e_greedy_data = vec![];
    for &x in e_greedy_param_values {
        let config = ActionOccurrenceAgentConfig {
            epsilon_decay_strategy: decay::Constant::new(x as f32),
            ..Default::default()
        };
        let mut agent = ActionOccurrenceAgent::new(config);
        agent.go(&mut env);
        let avg_reward = env.take_rewards().into_iter().sum::<f32>() as f64 / STEP_LIMIT as f64;
        e_greedy_data.push((x, avg_reward));
    }

    // UCB
    let mut ucb_data = vec![];
    for &x in ucb_param_values {
        let config = UCBAgentConfig {
            ucb_c: x as f32,
            ..Default::default()
        };
        let mut agent = UCBAgent::new(config);
        agent.go(&mut env);
        let avg_reward = env.take_rewards().into_iter().sum::<f32>() as f64 / STEP_LIMIT as f64;
        ucb_data.push((x, avg_reward));
    }

    // Greedy optimistic initialization
    let mut goi_data = vec![];
    for &x in goi_param_values {
        let config = ActionOccurrenceAgentConfig {
            alpha_fn: |_| 0.1,
            default_action_value: x as f32,
            ..Default::default()
        };
        let mut agent = ActionOccurrenceAgent::new(config);
        agent.go(&mut env);
        let avg_reward = env.take_rewards().into_iter().sum::<f32>() as f64 / STEP_LIMIT as f64;
        goi_data.push((x, avg_reward));
    }

    // Write data to CSV

    let mut wtr = csv::Writer::from_path("examples/ten_armed_testbed/out/data.csv")?;
    wtr.write_record(&["param", "reward", "algo"])?;

    for data in e_greedy_data {
        wtr.write_record(&[&data.0.to_string(), &data.1.to_string(), "Epsilon greedy"])?;
    }

    for data in ucb_data {
        wtr.write_record(&[&data.0.to_string(), &data.1.to_string(), "UCB"])?;
    }

    for data in goi_data {
        wtr.write_record(&[
            &data.0.to_string(),
            &data.1.to_string(),
            "Greedy optimistic initialization",
        ])?;
    }

    wtr.flush()?;

    // Plot data

    std::process::Command::new("python")
        .arg("examples/ten_armed_testbed/plot.py")
        .output()?;

    Ok(())
}
