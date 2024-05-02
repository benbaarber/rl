use std::collections::{HashSet, VecDeque};

use rand::{seq::IteratorRandom, thread_rng, Rng};
use strum::{EnumIter, FromRepr, IntoEnumIterator, VariantArray};

use crate::env::Environment;

/// Position coordinates in the field with 1 unit of padding as a death zone
type Pos = (usize, usize);

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

/// A field for the game of snake
pub struct GrassyField {
    size: usize,
    snake: Snake,
    food: Pos,
}

impl GrassyField {
    pub fn new(size: usize) -> Self {
        Self {
            size,
            snake: Snake::new(size),
            food: (1, 1),
        }
    }

    pub fn score(&self) -> usize {
        self.snake.len()
    }

    pub fn field_size(&self) -> usize {
        self.size
    }

    fn spawn_food(&mut self) {
        let occupied = HashSet::<&Pos>::from_iter(&self.snake.body);
        let mut vacant = Vec::with_capacity(self.size.pow(2) - self.snake.body.len());
        for i in 1..=self.size {
            for j in 1..=self.size {
                let pos = (i, j);
                if !occupied.contains(&pos) {
                    vacant.push(pos);
                }
            }
        }

        self.food = vacant.into_iter().choose(&mut thread_rng()).unwrap();
    }

    fn is_in_bounds(&self, pos: Pos) -> bool {
        pos >= (1, 1) && pos <= (self.size, self.size)
    }

    fn flat_pos(&self, pos: Pos) -> usize {
        pos.0 + self.size * pos.1
    }

    fn get_state(&self) -> Vec<f32> {
        let mut state = vec![0.0; self.size.pow(2)];
        state[self.flat_pos(self.food)] = 1.0;
        state[self.flat_pos(self.snake.head())] = -0.5;
        for segment in self.snake.body.iter().skip(1) {
            state[self.flat_pos(*segment)] = -1.0;
        }

        state
    }
}

impl Environment for GrassyField {
    type State = Vec<f32>;
    type Action = Dir;

    fn actions(&self) -> Vec<Self::Action> {
        Dir::VARIANTS.to_vec()
    }

    fn is_active(&self) -> bool {
        self.is_in_bounds(self.snake.head()) && !self.snake.is_intersecting()
    }

    fn reset(&mut self) -> Self::State {
        self.snake = Snake::new(self.size);
        self.spawn_food();
        self.get_state()
    }

    fn step(&mut self, action: Self::Action) -> (Option<Self::State>, f64) {
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

        if !self.is_active() {
            (None, -1.0)
        } else {
            (Some(self.get_state()), reward)
        }
    }
}
