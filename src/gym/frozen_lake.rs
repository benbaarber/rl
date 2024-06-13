use crate::env::{DiscreteActionSpace, Environment, Report};

#[derive(PartialEq)]
pub enum Square {
    Frozen = 0,
    Hole = 1,
    Start = 2,
    Goal = 3,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FLAction {
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
    pub report: Report,
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
            report: Report::new(vec!["reward", "steps"]),
        }
    }
}

impl DiscreteActionSpace for FrozenLake {
    fn actions(&self) -> Vec<Self::Action> {
        let mut actions = Vec::with_capacity(4);

        if self.pos % 4 != 0 {
            actions.push(FLAction::Left)
        }
        if self.pos < 12 {
            actions.push(FLAction::Down)
        }
        if self.pos % 4 != 3 {
            actions.push(FLAction::Right)
        }
        if self.pos > 3 {
            actions.push(FLAction::Up)
        }

        actions
    }
}

impl Environment for FrozenLake {
    type State = usize;
    type Action = FLAction;

    fn is_active(&self) -> bool {
        match self.map[self.pos] {
            Square::Frozen | Square::Start => true,
            Square::Hole | Square::Goal => false,
        }
    }

    fn random_action() -> Self::Action {
        FLAction::Down // TODO: random actions when available actions are state dependent require instance as parameter
    }

    fn step(&mut self, action: Self::Action) -> (Option<Self::State>, f32) {
        self.report.entry("steps").and_modify(|x| *x += 1.0);

        match action {
            FLAction::Left => self.pos -= 1,
            FLAction::Down => self.pos += 4,
            FLAction::Right => self.pos += 1,
            FLAction::Up => self.pos -= 4,
        };

        let (next_state, reward) = match self.map[self.pos] {
            Square::Hole => (None, -1.0),
            Square::Goal => (None, 1.0),
            _ => (Some(self.pos), -0.1),
        };

        self.report.entry("reward").and_modify(|x| *x += reward);

        (next_state, reward as f32)
    }

    fn reset(&mut self) -> Self::State {
        self.pos = 0;
        self.pos
    }
}
