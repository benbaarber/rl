use std::{sync::mpsc, thread};

use agent::SnakeDQN;
use burn::backend::{wgpu, Autodiff, Wgpu};
use model::ModelConfig;
use once_cell::sync::Lazy;
use rl::{decay, exploration::EpsilonGreedy, gym::GrassyField, viz};

mod agent;
mod model;

type DQNBackend = Wgpu<wgpu::AutoGraphicsApi, f32, i32>;
type DQNAutodiffBackend = Autodiff<DQNBackend>;

static DEVICE: Lazy<wgpu::WgpuDevice> = Lazy::new(wgpu::WgpuDevice::default);

const FIELD_SIZE: usize = 6;
const NUM_EPISODES: u16 = 50000;
const UPDATE_FREQ: u16 = 10;

fn main() {
    let mut env = GrassyField::<FIELD_SIZE>::new();
    let model_config = ModelConfig::new(FIELD_SIZE, 32, 128, 64, 128);
    let exploration = EpsilonGreedy::new(decay::Exponential::new(1e-5, 0.915, 0.1).unwrap());

    let mut app = viz::App::new(env.report.keys(), NUM_EPISODES);
    let (tx, rx) = mpsc::channel();
    let app_handle = thread::spawn(move || app.run(rx));

    let mut agent = SnakeDQN::new(model_config, exploration);

    for i in 0..(NUM_EPISODES / UPDATE_FREQ) {
        for _ in 0..UPDATE_FREQ {
            agent.go(&mut env);
        }
        let report = env.report.take();
        let ep = i * UPDATE_FREQ;
        tx.send(viz::Update {
            episode: ep,
            data: report.values().map(|x| *x / UPDATE_FREQ as f64).collect(),
        })
        .unwrap();
    }

    let _ = app_handle.join();
}
