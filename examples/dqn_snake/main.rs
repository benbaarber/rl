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

const FIELD_SIZE: usize = 6;
const NUM_EPISODES: u16 = 50000;
const UPDATE_FREQ: u16 = 10;

fn main() {
    let mut env = GrassyField::<FIELD_SIZE>::new();
    let model_config = ModelConfig::new(FIELD_SIZE, 32, 128, 64, 128);
    let loss_config = HuberLossConfig::new(1.35);
    let exploration = EpsilonGreedy::new(decay::Exponential::new(1e-5, 0.915, 0.1).unwrap());

    let mut agent = SnakeDQN::new(&mut env, model_config, loss_config, exploration);

    let mut app = viz::App::new(&["Score", "Steps", "Reward"], NUM_EPISODES);
    let (tx, rx) = mpsc::channel();
    let app_handle = thread::spawn(move || app.run(rx));

    for i in 0..(NUM_EPISODES / UPDATE_FREQ) {
        let mut total_score = 0.0;
        let mut total_steps = 0.0;
        let mut total_reward = 0.0;
        for _ in 0..UPDATE_FREQ {
            let summary = agent.go();
            total_score += summary.score as f64;
            total_steps += summary.steps as f64;
            total_reward += summary.reward as f64;
        }
        total_score /= UPDATE_FREQ as f64;
        total_steps /= UPDATE_FREQ as f64;
        total_reward /= UPDATE_FREQ as f64;
        let ep = i * UPDATE_FREQ;
        tx.send(Update {
            episode: ep,
            data: vec![
                (ep as f64, total_score),
                (ep as f64, total_steps),
                (ep as f64, total_reward),
            ],
        })
        .unwrap();
    }

    let _ = app_handle.join();
}
