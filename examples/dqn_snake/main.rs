use std::{sync::mpsc, thread};

use agent::SnakeDQN;
use burn::{
    backend::{
        wgpu::{self, WgpuDevice},
        Autodiff, Wgpu,
    },
    nn::loss::HuberLossConfig,
};
use model::ModelConfig;
use once_cell::sync::Lazy;
use rl::{
    decay,
    exploration::EpsilonGreedy,
    gym::GrassyField,
    viz::{self, Update},
};

mod agent;
mod model;

type DQNBackend = Wgpu<wgpu::AutoGraphicsApi, f32, i32>;
type DQNAutodiffBackend = Autodiff<DQNBackend>;

static DEVICE: Lazy<WgpuDevice> = Lazy::new(WgpuDevice::default);

const FIELD_SIZE: usize = 8;
const NUM_EPISODES: u16 = 1000;

fn main() {
    let mut env = GrassyField::new();
    let model_config = ModelConfig::new(FIELD_SIZE, 32, 128, 64, 128);
    let loss_config = HuberLossConfig::new(1.35);
    let exploration = EpsilonGreedy::new(decay::Exponential::new(18276.0, 0.915, 0.1).unwrap());

    let mut agent = SnakeDQN::new(&mut env, model_config, loss_config, exploration);

    let mut app = viz::App::new(&["Score", "Steps"], NUM_EPISODES);
    let (tx, rx) = mpsc::channel();
    let app_handle = thread::spawn(move || app.run(rx));

    for i in 0..NUM_EPISODES {
        let summary = agent.go();
        tx.send(Update {
            episode: i,
            data: vec![
                (i as f64, summary.score as f64),
                (i as f64, summary.steps as f64),
            ],
        })
        .unwrap();
    }

    app_handle.join();
}
