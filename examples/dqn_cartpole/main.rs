use burn::backend::{wgpu::WgpuDevice, Autodiff, Wgpu};
use gym_rs::utils::renderer::RenderMode;
use model::ModelConfig;
use once_cell::sync::Lazy;
use rl::{
    algo::dqn::{DQNAgent, DQNAgentConfig},
    gym::CartPole,
    viz,
};

mod model;

type DQNBackend = Autodiff<Wgpu>;

static DEVICE: Lazy<WgpuDevice> = Lazy::new(WgpuDevice::default);

const NUM_EPISODES: u16 = 128;

fn main() {
    let mut env = CartPole::new(RenderMode::Human);

    let model = ModelConfig::new(64, 128).init::<DQNBackend>(&*DEVICE);
    let agent_config = DQNAgentConfig::default();
    let mut agent = DQNAgent::new(model, agent_config, &*DEVICE);

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
