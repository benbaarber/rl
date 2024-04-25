use rl::{algo::QAgent, exploration::EpsilonGreedy, gym::FrozenLake};

fn main() {
    let env = FrozenLake::new();
    let mut agent = QAgent::new(env, 0.7, 0.95, EpsilonGreedy::new(1.0, 0.01, 1e-3));

    for _ in 0..10000 {
        agent.go();
    }

    println!("{:?}", agent.get_q_table());
}
