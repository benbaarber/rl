use std::error::Error;

use plotters::{
    backend::BitMapBackend,
    chart::ChartBuilder,
    drawing::IntoDrawingArea,
    element::PathElement,
    series::LineSeries,
    style::{Color, IntoFont, BLACK, BLUE, GREEN, RED, WHITE},
};
use rl::{
    algo::tabular::sample_average::{SampleAverageAgent, SampleAverageAgentConfig},
    decay,
    exploration::EpsilonGreedy,
    gym::KArmedBandit,
};

const STEP_LIMIT: usize = 1000;
const NUM_EPISODES: usize = 2000;

fn main() -> Result<(), Box<dyn Error>> {
    let mut eps_01_reward = [0.0; STEP_LIMIT];
    let mut eps_001_reward = [0.0; STEP_LIMIT];
    let mut eps_0_reward = [0.0; STEP_LIMIT];
    for _ in 0..NUM_EPISODES {
        let mut env = KArmedBandit::<10>::new(STEP_LIMIT);

        // Epsilon = 0.1
        let config = SampleAverageAgentConfig::default();
        let mut agent = SampleAverageAgent::new(config);
        agent.go(&mut env);
        for (i, x) in env.take_rewards().into_iter().enumerate() {
            eps_01_reward[i] += x as f64;
        }

        // Epsilon = 0.01
        let config = SampleAverageAgentConfig {
            exploration: EpsilonGreedy::new(decay::Constant::new(0.01)),
        };
        let mut agent = SampleAverageAgent::new(config);
        agent.go(&mut env);
        for (i, x) in env.take_rewards().into_iter().enumerate() {
            eps_001_reward[i] += x as f64;
        }

        // Epsilon = 0.0 (greedy)
        let config = SampleAverageAgentConfig {
            exploration: EpsilonGreedy::new(decay::Constant::new(0.0)),
        };
        let mut agent = SampleAverageAgent::new(config);
        agent.go(&mut env);
        for (i, x) in env.take_rewards().into_iter().enumerate() {
            eps_0_reward[i] += x as f64;
        }
    }

    let [eps_01_reward, eps_001_reward, eps_0_reward] =
        [eps_01_reward, eps_001_reward, eps_0_reward].map(|arr| {
            arr.into_iter()
                .enumerate()
                .map(|(i, x)| (i as i32, x / NUM_EPISODES as f64))
        });

    // Plot the results

    let root = BitMapBackend::new("local/ten_armed_testbed.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Average Reward", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..1000, 0.0..2.0)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(eps_01_reward, &RED))?
        .label("Epsilon = 0.1")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .draw_series(LineSeries::new(eps_001_reward, &BLUE))?
        .label("Epsilon = 0.01")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .draw_series(LineSeries::new(eps_0_reward, &GREEN))?
        .label("Epsilon = 0.0 (greedy)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    std::process::Command::new("xdg-open")
        .arg("local/ten_armed_testbed.png")
        .output()?;

    Ok(())
}
