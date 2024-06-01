use std::{
    collections::{HashSet, VecDeque},
    ops::{Index, IndexMut},
};

use rand::{seq::IteratorRandom, thread_rng, Rng};
use strum::{EnumIter, FromRepr, IntoEnumIterator, VariantArray};

use crate::env::Environment;

/// Position coordinates in the field with 1 unit of padding as a death zone
type Pos = (usize, usize);

pub type Grid<const S: usize> = [[f32; S]; S];

struct IGrid<const S: usize> {
    grid: Grid<S>,
}

impl<const S: usize> IGrid<S> {
    fn new() -> Self {
        Self {
            grid: [[0.0; S]; S],
        }
    }

    fn take(self) -> Grid<S> {
        self.grid
    }
}

impl<const S: usize> Index<&Pos> for IGrid<S> {
    type Output = f32;

    fn index(&self, index: &Pos) -> &Self::Output {
        &self.grid[index.0 - 1][index.1 - 1]
    }
}

impl<const S: usize> IndexMut<&Pos> for IGrid<S> {
    fn index_mut(&mut self, index: &Pos) -> &mut Self::Output {
        &mut self.grid[index.0 - 1][index.1 - 1]
    }
}

#[derive(EnumIter, VariantArray, FromRepr, Clone, Copy, Debug)]
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

#[derive(Default)]
pub struct Summary {
    pub score: usize,
    pub steps: u32,
    pub reward: f32,
}

/// A field for the game of snake
pub struct GrassyField<const S: usize> {
    snake: Snake,
    food: Pos,
    steps: u32,
    reward: f32,
}

impl<const S: usize> GrassyField<S> {
    pub fn new() -> Self {
        Self {
            snake: Snake::new(S),
            food: (1, 1),
            steps: 0,
            reward: 0.0,
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

        self.food = vacant.into_iter().choose(&mut thread_rng()).unwrap();
    }

    fn is_in_bounds(&self, pos: Pos) -> bool {
        pos.0 >= 1 && pos.1 >= 1 && pos.0 <= S && pos.1 <= S
    }

    fn get_state(&self) -> Grid<S> {
        let mut state = IGrid::new();
        state[&self.food] = 1.0;
        state[&self.snake.head()] = -0.5;
        for segment in self.snake.body.iter().skip(1) {
            state[segment] = -1.0;
        }

        state.take()
    }
}

impl<const S: usize> Environment for GrassyField<S> {
    type State = Grid<S>;
    type Action = Dir;
    type Summary = Summary;

    fn actions(&self) -> Vec<Self::Action> {
        Dir::VARIANTS.to_vec()
    }

    fn is_active(&self) -> bool {
        self.is_in_bounds(self.snake.head()) && !self.snake.is_intersecting()
    }

    fn summary(&self) -> Self::Summary {
        Summary {
            score: self.score(),
            steps: self.steps,
            reward: self.reward,
        }
    }

    fn reset(&mut self) -> Self::State {
        self.steps = 0;
        self.reward = 0.0;
        self.snake = Snake::new(S);
        self.spawn_food();
        self.get_state()
    }

    fn step(&mut self, action: Self::Action) -> (Option<Self::State>, f32) {
        self.steps += 1;
        let mut reward = -0.05;
        let head = self.snake.head();

        let t = self.snake.turn(action) as isize;
        self.snake.body.push_front((
            (head.0 as isize + (t & 1) * (2 - t)) as usize,
            (head.1 as isize + ((t + 1) & 1) * (t - 1)) as usize,
        ));

        if self.snake.head() == self.food {
            self.spawn_food();
            reward = 1.0;
        } else {
            self.snake.body.pop_back();
        }

        let next_state = if self.is_active() {
            Some(self.get_state())
        } else {
            reward = -1.0;
            None
        };

        self.reward += reward;
        (next_state, reward)
    }
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

        let mut env = GrassyField::<6> {
            snake,
            food: (1, 2),
            steps: 0,
            reward: 0.0,
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

        assert_ne!(env.food, (1, 2), "Food was moved after being eaten");

        let summary = env.summary();
        assert_eq!(summary.score, 1, "Summary score correct");
        assert_eq!(summary.steps, 5, "Summary steps correct");
    }
}
