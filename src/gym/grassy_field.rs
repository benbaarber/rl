use std::{
    collections::{HashSet, VecDeque},
    ops::{Index, IndexMut},
};

use rand::{seq::IteratorRandom, thread_rng, Rng};
use strum::{EnumIter, FromRepr, IntoEnumIterator, VariantArray};

use crate::env::{EnvState, Environment};

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
        &self.grid[index.0][index.1]
    }
}

impl<const S: usize> IndexMut<&Pos> for IGrid<S> {
    fn index_mut(&mut self, index: &Pos) -> &mut Self::Output {
        &mut self.grid[index.0][index.1]
    }
}

#[derive(EnumIter, VariantArray, FromRepr, Clone, Copy)]
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
        self.body.iter().all(move |x| uniq.insert(x))
    }

    fn head(&self) -> Pos {
        *self.body.front().expect("body is not empty")
    }

    fn len(&self) -> usize {
        self.body.len()
    }
}

pub struct Summary {
    score: usize,
    steps: u32,
}

/// A field for the game of snake
pub struct GrassyField<const S: usize> {
    snake: Snake,
    food: Pos,
    steps: u32,
}

impl<const S: usize> GrassyField<S> {
    pub fn new() -> Self {
        Self {
            snake: Snake::new(S),
            food: (1, 1),
            steps: 0,
        }
    }

    pub fn score(&self) -> usize {
        self.snake.len()
    }

    pub fn field_size(&self) -> usize {
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
        pos >= (1, 1) && pos <= (S, S)
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

    fn is_alive(&self) -> bool {
        self.is_in_bounds(self.snake.head()) && !self.snake.is_intersecting()
    }
}

impl<const S: usize> Environment for GrassyField<S> {
    type State = Grid<S>;
    type Action = Dir;
    type Summary = Summary;

    fn actions(&self) -> Vec<Self::Action> {
        Dir::VARIANTS.to_vec()
    }

    fn get_activity_state(&self) -> EnvState<Self> {
        if self.is_alive() {
            EnvState::Active
        } else {
            EnvState::Terminal(Summary {
                score: self.snake.len() - 1,
                steps: self.steps,
            })
        }
    }

    fn reset(&mut self) -> Self::State {
        self.steps = 0;
        self.snake = Snake::new(S);
        self.spawn_food();
        self.get_state()
    }

    fn step(&mut self, action: Self::Action) -> (Option<Self::State>, f32) {
        let mut reward = -0.05;
        let head = self.snake.head();

        self.snake.dir = action;
        let t = action as usize;
        self.snake
            .body
            .push_front((head.0 + (t & 1) * (2 - t), head.1 + ((t + 1) & 1) * (t - 1)));

        if self.snake.head() == self.food {
            self.spawn_food();
            reward = 1.0;
        } else {
            self.snake.body.pop_back();
        }

        if !self.is_alive() {
            (None, -1.0)
        } else {
            (Some(self.get_state()), reward)
        }
    }
}
