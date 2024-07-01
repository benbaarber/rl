use std::error::Error;

use plotters::{
    backend::BitMapBackend,
    chart::ChartBuilder,
    drawing::IntoDrawingArea,
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
    let mut total_reward = [[0.0; STEP_LIMIT]; 3];
    for _ in 0..NUM_EPISODES {
        let mut env = KArmedBandit::<10>::new(STEP_LIMIT);

        // Epsilon = 0.1
        let config = SampleAverageAgentConfig::default();
        let mut agent = SampleAverageAgent::new(config);
        agent.go(&mut env);
        for (i, x) in env.take_rewards().into_iter().enumerate() {
            total_reward[0][i] += x;
        }

        // Epsilon = 0.01
        let config = SampleAverageAgentConfig {
            exploration: EpsilonGreedy::new(decay::Constant::new(0.01)),
        };
        let mut agent = SampleAverageAgent::new(config);
        agent.go(&mut env);
        for (i, x) in env.take_rewards().into_iter().enumerate() {
            total_reward[1][i] += x;
        }

        // Epsilon = 0.0 (greedy)
        let config = SampleAverageAgentConfig {
            exploration: EpsilonGreedy::new(decay::Constant::new(0.0)),
        };
        let mut agent = SampleAverageAgent::new(config);
        agent.go(&mut env);
        for (i, x) in env.take_rewards().into_iter().enumerate() {
            total_reward[2][i] += x;
        }
    }

    let data = total_reward.map(|d| {
        d.into_iter()
            .enumerate()
            .map(|(i, x)| (i as i32, x as f64 / NUM_EPISODES as f64))
    });

    // Plot the results

    let root = BitMapBackend::new("examples/ten_armed_testbed/results.png", (1024, 768))
        .into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Average Reward", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..1000, 0.0..2.0)?;

    chart.configure_mesh().draw()?;

    let [data1, data2, data3] = data;

    chart
        .draw_series(LineSeries::new(data1, &RED))?
        .label("Epsilon = 0.1");

    chart
        .draw_series(LineSeries::new(data2, &BLUE))?
        .label("Epsilon = 0.01");

    chart
        .draw_series(LineSeries::new(data3, &GREEN))?
        .label("Epsilon = 0.0 (greedy)");

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    std::process::Command::new("xdg-open")
        .arg("examples/ten_armed_testbed/results.png")
        .output()?;

    Ok(())
}
