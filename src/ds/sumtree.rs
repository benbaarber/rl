/// A binary tree data structure where each parent node is the sum of its child nodes
pub struct SumTree {
    tree: Vec<f32>,
    capacity: usize,
}

impl SumTree {
    /// Initialize a new `SumTree` with a given capacity
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two();
        Self {
            tree: vec![0.0; 2 * capacity - 1],
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
    }

    /// Find the first index `i` where the sum of the values from 0 to `i` is greater than `value`
    pub fn find(&self, value: f32) -> usize {
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

        ix - (self.capacity - 1)
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
        assert_eq!(sumtree.find(4.0), 3, "find works on left side");
        assert_eq!(sumtree.find(18.0), 6, "find works on right side");
    }
}
