use std::collections::{HashSet, VecDeque};

use rand::{seq::IteratorRandom, thread_rng, Rng};
use strum::{EnumIter, FromRepr, IntoEnumIterator, VariantArray};

use crate::env::{DiscreteActionSpace, Environment, Report};

/// Position coordinates in the field with 1 unit of padding as a death zone
type Pos = (usize, usize);

#[derive(EnumIter, VariantArray, FromRepr, Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Dir {
    Up = 0,
    Right = 1,
    Down = 2,
    Left = 3,
}

pub struct Snake {
    body: VecDeque<Pos>,
    dir: Dir,
}

impl Snake {
    fn new(field_size: usize) -> Self {
        let mut rng = thread_rng();
        Self {
            body: VecDeque::from([(
                rng.gen_range(3..(field_size - 1)),
                rng.gen_range(3..(field_size - 1)),
            )]),
            dir: Dir::iter().choose(&mut rng).unwrap(),
        }
    }

    fn is_intersecting(&self) -> bool {
        let mut uniq = HashSet::new();
        self.body.iter().any(move |x| !uniq.insert(x))
    }

    fn head(&self) -> Pos {
        *self.body.front().expect("body is not empty")
    }

    fn len(&self) -> usize {
        self.body.len()
    }

    fn turn(&mut self, dir: Dir) -> Dir {
        let d1 = self.dir as isize;
        let d2 = dir as isize;
        if (d1 - d2).abs() != 2 {
            self.dir = dir;
        }

        self.dir
    }
}

/// A field for the game of snake
pub struct GrassyField2<const S: usize> {
    snake: Snake,
    food: Pos,
    pub report: Report,
}

impl<const S: usize> GrassyField2<S> {
    pub fn new() -> Self {
        Self {
            snake: Snake::new(S),
            food: (1, 1),
            report: Report::new(vec!["score", "reward", "steps"]),
        }
    }

    pub fn score(&self) -> usize {
        self.snake.len() - 1
    }

    pub const fn field_size(&self) -> usize {
        S
    }

    fn spawn_food(&mut self) {
        let occupied = HashSet::<&Pos>::from_iter(&self.snake.body);
        let mut vacant = Vec::with_capacity(S.pow(2) - self.snake.body.len());
        for i in 1..=S {
            for j in 1..=S {
                let pos = (i, j);
                if !occupied.contains(&pos) {
                    vacant.push(pos);
                }
            }
        }

        self.food = vacant
            .into_iter()
            .choose(&mut thread_rng())
            .unwrap_or((1, 1));
    }

    fn is_in_bounds(&self, pos: Pos) -> bool {
        pos.0 >= 1 && pos.1 >= 1 && pos.0 <= S && pos.1 <= S
    }

    /// Get environment state as an array of boolean features
    ///
    /// `[facing_up, facing_right, facing_down, facing_left, food_up, food_right, food_down, food_left, danger_up, danger_right, danger_down, danger_left]`
    fn get_state(&self) -> [bool; 12] {
        let mut features = [false; 12];

        features[self.snake.dir as usize] = true;

        let head = self.snake.head();
        if self.food.1 > head.1 {
            features[6] = true;
        } else if self.food.1 < head.1 {
            features[4] = true;
        }
        if self.food.0 > head.0 {
            features[5] = true;
        } else if self.food.0 < head.0 {
            features[7] = true;
        }

        for (i, dir) in Dir::iter().enumerate() {
            let pos = step_dir(head, dir);
            features[8 + i] = !self.is_in_bounds(pos) || self.snake.body.contains(&pos);
        }

        features
    }
}

impl<const S: usize> DiscreteActionSpace for GrassyField2<S> {
    fn actions(&self) -> Vec<Self::Action> {
        Dir::VARIANTS.to_vec()
    }
}

impl<const S: usize> Environment for GrassyField2<S> {
    type State = [bool; 12];
    type Action = Dir;

    fn is_active(&self) -> bool {
        self.is_in_bounds(self.snake.head())
            && !self.snake.is_intersecting()
            && self.snake.len() < S * S
    }

    fn random_action() -> Self::Action {
        Dir::iter().choose(&mut thread_rng()).unwrap()
    }

    fn reset(&mut self) -> Self::State {
        self.snake = Snake::new(S);
        self.spawn_food();
        self.get_state()
    }

    fn step(&mut self, action: Self::Action) -> (Option<Self::State>, f32) {
        self.report.entry("steps").and_modify(|x| *x += 1.0);
        let mut reward = -0.01;

        let head = self.snake.head();
        let dir = self.snake.turn(action);
        let new_head = step_dir(head, dir);
        self.snake.body.push_front(new_head);

        if self.snake.head() == self.food {
            self.report.entry("score").and_modify(|x| *x += 1.0);
            self.spawn_food();
            reward = 1.0;
        } else {
            self.snake.body.pop_back();
        }

        let next_state = if self.is_active() {
            Some(self.get_state())
        } else {
            reward = if self.snake.len() < S * S { -10.0 } else { 1.0 };
            None
        };

        self.report.entry("reward").and_modify(|x| *x += reward);
        (next_state, reward as f32)
    }
}

fn step_dir(pos: Pos, dir: Dir) -> Pos {
    let t = dir as isize;
    (
        (pos.0 as isize + (t & 1) * (2 - t)) as usize,
        (pos.1 as isize + ((t + 1) & 1) * (t - 1)) as usize,
    )
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn snake_actions() {
        let snake = Snake {
            body: VecDeque::from([(1, 1)]),
            dir: Dir::Down,
        };

        let mut env = GrassyField2::<6> {
            snake,
            food: (2, 1),
            report: Report::new(vec!["score", "reward", "steps"]),
        };

        env.step(Dir::Down);
        assert_eq!(env.snake.head(), (1, 2), "Down action works");

        env.step(Dir::Right);
        assert_eq!(env.snake.head(), (2, 2), "Right action works");

        env.step(Dir::Right);
        env.step(Dir::Up);
        assert_eq!(env.snake.head(), (3, 1), "Up action works");

        env.step(Dir::Left);
        assert_eq!(env.snake.head(), (2, 1), "Left action works");

        assert!(env.is_active(), "Env is active");

        assert_ne!(env.food, (2, 1), "Food was moved after being eaten");

        let report = env.report.take();
        assert_eq!(*report.get("score").unwrap(), 1.0, "Report score correct");
        assert_eq!(*report.get("steps").unwrap(), 5.0, "Report steps correct");
    }
}
