use agent::Agent;
use burn::backend::{wgpu, Autodiff, Wgpu};
use gym_rs::utils::renderer::RenderMode;
use model::ModelConfig;
use once_cell::sync::Lazy;
use rl::{decay, exploration::EpsilonGreedy, gym::CartPole, viz};

mod agent;
mod model;

type DQNBackend = Wgpu<wgpu::AutoGraphicsApi, f32, i32>;
type DQNAutodiffBackend = Autodiff<DQNBackend>;

static DEVICE: Lazy<wgpu::WgpuDevice> = Lazy::new(wgpu::WgpuDevice::default);

const NUM_EPISODES: u16 = 50000;
const UPDATE_FREQ: u16 = 10;

fn main() {
    let mut env = CartPole::new(RenderMode::Human);

    let model_config = ModelConfig::new(32, 64);
    let exploration = EpsilonGreedy::new(decay::Exponential::new(1e-5, 0.915, 0.1).unwrap());
    let mut agent = Agent::new(model_config, exploration);

    let (handle, tx) = viz::init(env.report.keys(), NUM_EPISODES);

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

    let _ = handle.join();
}
