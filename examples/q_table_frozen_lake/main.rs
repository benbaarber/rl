use rl::{
    algo::tabular::q_table::{QTableAgent, QTableAgentConfig},
    decay,
    exploration::EpsilonGreedy,
    gym::FrozenLake,
    viz,
};

const NUM_EPISODES: u16 = 10000;

fn main() {
    let mut env = FrozenLake::new();
    let config = QTableAgentConfig {
        exploration: EpsilonGreedy::new(decay::Exponential::new(1e-3, 1.0, 0.01).unwrap()),
        ..Default::default()
    };
    let mut agent = QTableAgent::new(config);

    let (handle, tx) = viz::init(env.report.keys(), NUM_EPISODES);

    for i in 0..NUM_EPISODES {
        agent.go(&mut env);
        let report = env.report.take();
        tx.send(viz::Update {
            episode: i,
            data: report.values().copied().collect(),
        })
        .unwrap();
    }

    let _ = handle.join();
}
