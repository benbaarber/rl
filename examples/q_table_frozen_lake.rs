use std::{sync::mpsc, thread};

use rl::{
    algo::{QAgent, QTableAgent},
    decay,
    exploration::EpsilonGreedy,
    gym::FrozenLake,
    viz::{self, App},
};

fn main() {
    const NUM_EPISODES: u16 = 10000;
    let mut env = FrozenLake::new();
    let mut agent = QTableAgent::new(
        &mut env,
        0.7,
        0.95,
        EpsilonGreedy::new(decay::Exponential::new(1e-3, 1.0, 0.01).unwrap()),
    );

    let mut app = App::new(&["Steps"], NUM_EPISODES);
    let (tx, rx) = mpsc::channel();

    let app_handle = thread::spawn(move || app.run(rx));

    for e in 0..NUM_EPISODES {
        let summary = agent.go();
        tx.send(viz::Update {
            episode: e,
            data: vec![(e as f64, summary.steps as f64)],
        })
        .unwrap();
    }

    app_handle.join();
}
