use std::ops::Index;

/// A fixed-size ringbuffer
pub struct RingBuffer<T> {
    buffer: Vec<T>,
    i: usize,
}

impl<T> RingBuffer<T> {
    /// Constructs a new `RingBuffer` from a provided `Vec`
    pub fn from(data: Vec<T>) -> Self {
        Self { buffer: data, i: 0 }
    }

    /// Returns the buffer length
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Insert an element into the buffer, overwriting the oldest element
    pub fn push(&mut self, item: T) {
        self.buffer[self.i] = item;
        self.i = (self.i + 1) % self.len();
    }

    /// Get a slice view of the internal buffer
    pub fn view(&self) -> &[T] {
        &self.buffer
    }
}

impl<T: Clone> Index<usize> for RingBuffer<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.buffer[index]
    }
}

/// A fixed-size ring buffer with capacity known at compile time
pub struct StaticRingBuffer<T, const CAP: usize> {
    buffer: [T; CAP],
    i: usize,
}

impl<T, const CAP: usize> StaticRingBuffer<T, CAP> {
    /// Constructs a new `StaticRingBuffer` from a provided array
    pub fn from(data: [T; CAP]) -> Self {
        Self { buffer: data, i: 0 }
    }

    /// Insert an element into the buffer, overwriting the oldest element
    pub fn push(&mut self, item: T) {
        self.buffer[self.i] = item;
        self.i = (self.i + 1) % CAP;
    }

    /// Get a slice view of the internal buffer
    pub fn view(&self) -> &[T; CAP] {
        &self.buffer
    }
}

impl<T: Clone, const CAP: usize> Index<usize> for StaticRingBuffer<T, CAP> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.buffer[index]
    }
}
