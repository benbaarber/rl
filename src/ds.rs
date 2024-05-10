use std::ops::Index;

/// A fixed-size ringbuffer
pub struct RingBuffer<T> {
    buffer: Vec<T>,
    i: usize,
    capacity: usize,
}

impl<T> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::<T>::with_capacity(capacity),
            i: 0,
            capacity,
        }
    }

    /// Constructs a new `RingBuffer` from a provided `Vec`
    pub fn from(data: Vec<T>) -> Self {
        let capacity = data.len();
        Self {
            buffer: data,
            i: 0,
            capacity,
        }
    }

    /// Returns the buffer length
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Insert an element into the buffer, overwriting the oldest element
    pub fn push(&mut self, item: T) {
        if self.i >= self.len() {
            self.buffer.push(item);
        } else {
            self.buffer[self.i] = item;
        }
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