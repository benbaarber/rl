use rand::{seq::SliceRandom, thread_rng};
use std::ops::Index;

/// A fixed-size ringbuffer
struct RingBuffer<T, const CAP: usize> {
    buffer: [T; CAP],
    i: usize,
}

impl<T, const CAP: usize> RingBuffer<T, CAP> {
    /// Constructs a new `RingBuffer` from a provided array
    fn from(arr: [T; CAP]) -> Self {
        Self { buffer: arr, i: 0 }
    }

    /// Insert an element into the buffer, overwriting the oldest element
    fn push(&mut self, item: T) {
        self.buffer[self.i] = item;
        self.i = (self.i + 1) % CAP;
    }

    /// Get a slice view of the internal buffer
    fn view(&self) -> &[T; CAP] {
        &self.buffer
    }
}

impl<T: Copy, const CAP: usize> Index<usize> for RingBuffer<T, CAP> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.buffer[index]
    }
}

impl<T: Default + Copy, const CAP: usize> RingBuffer<T, CAP> {
    /// Construct a new `RingBuffer` of default values for `T`
    fn new() -> Self {
        Self {
            buffer: [T::default(); CAP],
            i: 0,
        }
    }
}

/// Represents a single experience or transition in the environment
///
/// **Type Parameters:**
/// - `S`: State
/// - `A`: Action
///
/// **Fields:**
/// - `.0` (state): The state of the environment before taking the action
/// - `.1` (action): The action taken in the given state
/// - `.2` (next state): The state of the environment after the action is taken
/// - `.3` (reward): The reward received after taking the action
#[derive(Copy, Clone)]
pub struct Experience<S, A>(S, A, S, f64);

/// A fixed-size memory storage for reinforcement learning experiences
///
/// This structure uses a ring buffer to store experiences, which are tuples of (state, action, next state, reward).
/// It automatically overwrites the oldest experiences once it reaches its capacity.
///
/// **Type Parameters:**
/// - `S`: Represents the type of the states in the environment
/// - `A`: Represents the type of the actions
/// - `CAP`: The maximum number of experiences the memory can hold, specified at compile time
///
/// **Fields:**
/// - `memory`: A `RingBuffer` that stores the experiences
pub struct ReplayMemory<S, A, const CAP: usize> {
    memory: RingBuffer<Experience<S, A>, CAP>,
}

impl<S: Copy, A: Copy, const CAP: usize> ReplayMemory<S, A, CAP> {
    /// Construct a new `ReplayMemory` from a provided array of experiences
    pub fn from(arr: [Experience<S, A>; CAP]) -> Self {
        Self {
            memory: RingBuffer::from(arr),
        }
    }

    /// Add a new experience to the memory
    pub fn push(&mut self, e: Experience<S, A>) {
        self.memory.push(e);
    }

    /// Sample a random batch of experiences from the memory
    pub fn sample(&self, batch_size: usize) -> Vec<Experience<S, A>> {
        let mut rng = thread_rng();
        self.memory
            .view()
            .choose_multiple(&mut rng, batch_size)
            .cloned()
            .collect()
    }
}
