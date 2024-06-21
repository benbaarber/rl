/// A binary tree data structure where each parent node is the sum of its child nodes
pub struct SumTree {
    tree: Vec<f32>,
    max: f32,
    capacity: usize,
}

impl SumTree {
    /// Initialize a new `SumTree` with a given capacity
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two();
        Self {
            tree: vec![0.0; 2 * capacity - 1],
            max: 0.0,
            capacity,
        }
    }

    /// Update the value at a provided index
    pub fn update(&mut self, ix: usize, value: f32) {
        let mut ix = ix + self.capacity - 1;
        let change = value - self.tree[ix];

        self.tree[ix] = value;

        while ix > 0 {
            ix = (ix - 1) / 2;
            self.tree[ix] += change;
        }

        if value > self.max {
            self.max = value;
        }
    }

    /// Find the first index `i` and value `v` where the sum of the values from 0 to `i` is greater than `value`
    pub fn find(&self, value: f32) -> (usize, f32) {
        let mut ix = 0;
        let mut val = value;
        while ix < self.capacity - 1 {
            let left = 2 * ix + 1;
            let right = left + 1;
            ix = if val <= self.tree[left] {
                left
            } else {
                val -= self.tree[left];
                right
            }
        }

        let ix = ix - (self.capacity - 1);
        (ix, self.tree[ix])
    }

    /// Get the sum of all values stored
    pub fn sum(&self) -> f32 {
        self.tree[0]
    }

    /// Get the max of all values stored
    pub fn max(&self) -> f32 {
        self.max
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sumtree_functional() {
        let mut sumtree = SumTree::new(8);
        assert_eq!(
            sumtree.tree.len(),
            15,
            "tree was initialized with correct length"
        );

        for i in 0..8 {
            sumtree.update(i, i as f32);
        }

        assert_eq!(
            sumtree.tree[0], 28.0,
            "root node contains sum of entire tree"
        );
        assert_eq!(sumtree.find(4.0), (3, 3.0), "find works on left side");
        assert_eq!(sumtree.find(18.0), (6, 6.0), "find works on right side");

        sumtree.update(3, 12.0);
        assert_eq!(sumtree.max(), 12.0, "maximum value stored correctly");
    }
}
