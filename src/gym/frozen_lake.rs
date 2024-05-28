use crate::env::{EnvState, Environment};

pub enum Square {
    Frozen = 0,
    Hole = 1,
    Start = 2,
    Goal = 3,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Move {
    Left = 0,
    Down = 1,
    Right = 2,
    Up = 3,
}

pub struct Summary {
    pub won: bool,
    pub steps: u32,
    pub reward: f64,
}

/// A very simple RL environment taken from Python [gymnasium](https://gymnasium.farama.org/)
///
/// Intended for use with a [QTableAgent](crate::algo::QTableAgent)
pub struct FrozenLake {
    map: [Square; 16],
    pos: usize,
    steps: u32,
    reward: f64,
}

impl FrozenLake {
    pub fn new() -> Self {
        // TODO: Support custom maps
        let map = [
            Square::Start,
            Square::Frozen,
            Square::Frozen,
            Square::Frozen,
            Square::Frozen,
            Square::Hole,
            Square::Frozen,
            Square::Hole,
            Square::Frozen,
            Square::Frozen,
            Square::Frozen,
            Square::Hole,
            Square::Hole,
            Square::Frozen,
            Square::Frozen,
            Square::Goal,
        ];
        Self {
            map,
            pos: 0,
            steps: 0,
            reward: 0.0,
        }
    }
}

impl Environment for FrozenLake {
    type State = usize;
    type Action = Move;
    type Summary = Summary;

    fn actions(&self) -> Vec<Self::Action> {
        let mut actions = Vec::with_capacity(4);

        if self.pos % 4 != 0 {
            actions.push(Move::Left)
        }
        if self.pos < 12 {
            actions.push(Move::Down)
        }
        if self.pos % 4 != 3 {
            actions.push(Move::Right)
        }
        if self.pos > 3 {
            actions.push(Move::Up)
        }

        actions
    }

    fn get_activity_state(&self) -> EnvState<Self> {
        match self.map[self.pos] {
            Square::Frozen | Square::Start => EnvState::Active,
            Square::Hole => EnvState::Terminal(Summary {
                won: false,
                steps: self.steps,
                reward: self.reward,
            }),
            Square::Goal => EnvState::Terminal(Summary {
                won: true,
                steps: self.steps,
                reward: self.reward,
            }),
        }
    }

    fn step(&mut self, action: Self::Action) -> (Option<Self::State>, f32) {
        self.steps += 1;

        match action {
            Move::Left => self.pos -= 1,
            Move::Down => self.pos += 4,
            Move::Right => self.pos += 1,
            Move::Up => self.pos -= 4,
        };

        let (next_state, reward) = match self.map[self.pos] {
            Square::Hole => (None, -1.0),
            Square::Goal => (None, 1.0),
            _ => (Some(self.pos), -0.1),
        };

        self.reward += reward as f64;

        (next_state, reward)
    }

    fn reset(&mut self) -> Self::State {
        self.steps = 0;
        self.reward = 0.0;
        self.pos = 0;
        self.pos
    }
}
