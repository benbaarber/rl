use crate::env::TableEnvironment;

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

pub struct FrozenLake {
    map: [Square; 16],
    pos: usize,
    steps: u32,
}

impl FrozenLake {
    pub fn new() -> Self {
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

impl TableEnvironment for FrozenLake {
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

    fn reward(&self, _state: Self::State, _action: Self::Action, next_state: Self::State) -> f64 {
        match self.map[next_state] {
            Square::Hole => -1.0,
            Square::Goal => 1.0,
            _ => -0.1,
        }
    }

    fn step(&mut self, action: Self::Action) -> (Self::State, f64) {
        self.steps += 1;

        let last_pos = self.pos;
        match action {
            Move::Left => self.pos -= 1,
            Move::Down => self.pos += 4,
            Move::Right => self.pos += 1,
            Move::Up => self.pos -= 4,
        };

        let reward = self.reward(last_pos, action, self.pos);
        (self.pos, reward)
    }

    fn reset(&mut self) -> Self::State {
        self.steps = 0;
        self.pos = 0;
        self.pos
    }
}
