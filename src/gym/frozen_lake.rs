use crate::env::Environment;

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

/// A very simple RL environment taken from Python [gymnasium](https://gymnasium.farama.org/)
///
/// Intended for use with a [QTableAgent](crate::algo::QTableAgent)
pub struct FrozenLake {
    map: [Square; 16],
    pos: usize,
    steps: u32,
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
        }
    }
}

impl Environment for FrozenLake {
    type State = usize;
    type Action = Move;

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

    fn is_active(&self) -> bool {
        match self.map[self.pos] {
            Square::Frozen | Square::Start => true,
            Square::Hole => {
                println!("Agent fell into a hole after {} steps.", self.steps);
                false
            }
            Square::Goal => {
                println!("Agent reached the goal after {} steps!", self.steps);
                false
            }
        }
    }

    fn step(&mut self, action: Self::Action) -> (Self::State, f64) {
        self.steps += 1;

        match action {
            Move::Left => self.pos -= 1,
            Move::Down => self.pos += 4,
            Move::Right => self.pos += 1,
            Move::Up => self.pos -= 4,
        };

        let reward = match self.map[self.pos] {
            Square::Hole => -1.0,
            Square::Goal => 1.0,
            _ => -0.1,
        };

        (self.pos, reward)
    }

    fn reset(&mut self) -> Self::State {
        self.steps = 0;
        self.pos = 0;
        self.pos
    }
}
