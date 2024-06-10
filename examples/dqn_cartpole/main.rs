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

const NUM_EPISODES: u16 = 1000;

fn main() {
    let mut env = CartPole::new(RenderMode::None);

    let model_config = ModelConfig::new(64, 128);
    let exploration = EpsilonGreedy::new(decay::Exponential::new(1e-3, 0.915, 0.05).unwrap());
    let mut agent = Agent::new(model_config, exploration);

    let (handle, tx) = viz::init(env.report.keys(), NUM_EPISODES);

    for i in 0..NUM_EPISODES {
        agent.go(&mut env);
        let report = env.report.take();
        tx.send(viz::Update {
            episode: i,
            data: report.values().cloned().collect(),
        })
        .unwrap();
    }

    let _ = handle.join();
}
