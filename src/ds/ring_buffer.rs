#![allow(clippy::len_without_is_empty)]
use std::ops::Index;

/// A fixed-size ringbuffer
#[derive(Debug, Default, Clone)]
pub struct RingBuffer<T> {
    buffer: Vec<T>,
    ix: usize,
    capacity: usize,
}

impl<T> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::<T>::with_capacity(capacity),
            ix: 0,
            capacity,
        }
    }

    /// Constructs a new `RingBuffer` from a provided `Vec`
    pub fn from(data: Vec<T>) -> Self {
        let capacity = data.len();
        Self {
            buffer: data,
            ix: 0,
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

    /// Insert an element into the buffer, overwriting the oldest element, and return the write index
    pub fn push(&mut self, item: T) -> usize {
        let ix = self.ix;
        if ix >= self.len() {
            self.buffer.push(item);
        } else {
            self.buffer[ix] = item;
        }
        self.ix = (ix + 1) % self.capacity;
        ix
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ringbuffer_functional() {
        let mut buf = RingBuffer::new(4);
        assert_eq!(buf.len(), 0, "initialized empty");

        for i in 0..4 {
            buf.push(i * 2);
        }

        assert_eq!(buf.len(), 4, "length correct");
        assert_eq!(buf.view(), [0, 2, 4, 6], "contents correct");

        buf.push(1);
        let ix = buf.push(3);
        assert_eq!(ix, 1, "write index is correct");
        assert_eq!(buf.len(), 4, "length unchanged");
        assert_eq!(buf.view(), [1, 3, 4, 6], "contents overwritten correctly");
    }
}
