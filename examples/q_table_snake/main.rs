use rl::{algo::q_table::QTableAgent, decay, exploration::EpsilonGreedy, gym::GrassyField, viz};

const FIELD_SIZE: usize = 20;
const NUM_EPISODES: u16 = 10000;

fn main() {
    let mut env = GrassyField::<FIELD_SIZE>::new();
    let mut agent = QTableAgent::new(
        0.7,
        0.95,
        EpsilonGreedy::new(decay::Exponential::new(1e-1, 1.0, 0.01).unwrap()),
    );

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
