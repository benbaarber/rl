use rl::{
    algo::tabular::q_table::{QTableAgent, QTableAgentConfig},
    gym::GrassyField,
    viz,
};

const FIELD_SIZE: usize = 20;
const NUM_EPISODES: u16 = 10000;

fn main() {
    let mut env = GrassyField::<FIELD_SIZE>::new();
    let config = QTableAgentConfig {
        gamma: 0.95,
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
