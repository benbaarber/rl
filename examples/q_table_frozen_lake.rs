use rl::{algo::QAgent, exploration::EpsilonGreedy, gym::FrozenLake};

fn main() {
    let env = FrozenLake::new();
    let mut agent = QAgent::new(env, 0.01, 0.999, EpsilonGreedy::new(0.9, 0.01, 1e6));

    for _ in 0..10000 {
        agent.go();
    }

    println!("{:?}", agent.get_q_table());
}
