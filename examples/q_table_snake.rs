use std::{sync::mpsc, thread};

use rl::{algo::QTableAgent, decay, exploration::EpsilonGreedy, gym::GrassyField2, viz};

const FIELD_SIZE: usize = 20;
const NUM_EPISODES: u16 = 10000;

fn main() {
    let mut env = GrassyField2::<FIELD_SIZE>::new();
    let mut agent = QTableAgent::new(
        0.7,
        0.95,
        EpsilonGreedy::new(decay::Exponential::new(1e-1, 1.0, 0.01).unwrap()),
    );

    let mut app = viz::App::new(env.report.keys(), NUM_EPISODES);
    let (tx, rx) = mpsc::channel();

    let app_handle = thread::spawn(move || app.run(rx));

    for i in 0..NUM_EPISODES {
        agent.go(&mut env);
        let report = env.report.take();
        tx.send(viz::Update {
            episode: i,
            data: report.values().copied().collect(),
        })
        .unwrap();
    }

    let _ = app_handle.join();
}
