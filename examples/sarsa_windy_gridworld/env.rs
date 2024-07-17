use rand::seq::IteratorRandom;
use rl::env::{DiscreteActionSpace, Environment, Report};
use strum::{EnumIter, VariantArray};

pub type Pos = (i32, i32);

#[derive(EnumIter, VariantArray, Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Action {
    Up,
    Left,
    Down,
    Right,
    UpLeft,
    DownLeft,
    DownRight,
    UpRight,
    Stay,
}

pub struct WindyGridworld {
    pos: Pos,
    goal: Pos,
    currents: [i32; 10],
    pub report: Report,
}

impl WindyGridworld {
    pub fn new() -> Self {
        Self {
            pos: (3, 0),
            goal: (3, 7),
            currents: [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
            report: Report::new(vec!["steps"]),
        }
    }
}

impl Environment for WindyGridworld {
    type State = Pos;
    type Action = Action;

    fn step(&mut self, action: Self::Action) -> (Option<Self::State>, f32) {
        self.report.entry("steps").and_modify(|x| *x += 1.0);

        let wind = self.currents[self.pos.0 as usize];
        self.pos.1 += wind;

        let change = match action {
            Action::Up => (0, -1),
            Action::Left => (-1, 0),
            Action::Down => (0, 1),
            Action::Right => (1, 0),
            Action::UpLeft => (-1, -1),
            Action::DownLeft => (-1, 1),
            Action::DownRight => (1, 1),
            Action::UpRight => (1, -1),
            Action::Stay => (0, 0),
        };

        self.pos.0 += change.0;
        self.pos.1 += change.1;
        self.pos = (self.pos.0.clamp(0, 9), self.pos.1.clamp(0, 7));

        if self.pos == self.goal {
            (None, 0.0)
        } else {
            (Some(self.pos), -1.0)
        }
    }

    fn reset(&mut self) -> Self::State {
        self.pos = (3, 0);
        self.pos
    }

    fn random_action(&self) -> Self::Action {
        self.actions()
            .into_iter()
            .choose(&mut rand::thread_rng())
            .expect("Iterator is not empty")
    }
}

impl DiscreteActionSpace for WindyGridworld {
    fn actions(&self) -> Vec<Self::Action> {
        Action::VARIANTS.to_vec()
    }
}
