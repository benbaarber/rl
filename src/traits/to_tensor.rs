use burn::{
    prelude::*,
    tensor::{BasicOps, DataSerialize, Element},
};

/// A trait for converting items to tensors
///
/// Commonly implemented for `Vec<T>` to convert batches of `T` to a tensor of dimension `D`
///
/// See implementations of this for [`CartPole`](crate::gym::CartPole) as an example of how to implement this trait
pub trait ToTensor<B: Backend, const D: usize, K: BasicOps<B>> {
    fn to_tensor(self, device: &B::Device) -> Tensor<B, D, K>;
}

// Implementations from burn

/// Marker trait to restrict blanket implementations
trait IntoData<const D: usize> {}

impl<E> IntoData<1> for &[E] {}
impl<E, const D: usize> IntoData<D> for DataSerialize<E> {}
impl<E, const D: usize> IntoData<D> for &DataSerialize<E> {}
impl<E, const A: usize, const B: usize, const C: usize, const D: usize> IntoData<4>
    for [[[[E; D]; C]; B]; A]
{
}
impl<E, const A: usize, const B: usize, const C: usize> IntoData<3> for [[[E; C]; B]; A] {}
impl<E, const A: usize, const B: usize> IntoData<2> for [[E; B]; A] {}
impl<E, const A: usize> IntoData<1> for [E; A] {}

impl<B, const D: usize, K, T> ToTensor<B, D, K> for T
where
    B: Backend,
    K: BasicOps<B>,
    T: Into<Data<K::Elem, D>> + IntoData<D>,
{
    fn to_tensor(self, device: &<B as Backend>::Device) -> Tensor<B, D, K> {
        Tensor::from_data(self.into(), device)
    }
}

// Implementations

impl<B, E, K> ToTensor<B, 1, K> for Vec<E>
where
    B: Backend,
    E: Element,
    K: BasicOps<B, Elem = E>,
{
    fn to_tensor(self, device: &<B as Backend>::Device) -> Tensor<B, 1, K> {
        let len = self.len();
        Tensor::from_data(Data::new(self, [len].into()), device)
    }
}

impl<B, E, K, const A: usize> ToTensor<B, 2, K> for Vec<[E; A]>
where
    B: Backend,
    E: Element,
    K: BasicOps<B, Elem = E>,
{
    fn to_tensor(self, device: &B::Device) -> Tensor<B, 2, K> {
        let len = self.len();
        let data = Data::new(
            self.into_iter().flatten().collect::<Vec<_>>(),
            [len * A].into(),
        );
        Tensor::from_data(data, device).reshape([-1, A as i32])
    }
}

// #[test]
// fn test() {
//     let b = vec![[1,2,3], [4,5,6]];
//     let a = vec![[1,2,3], [4,5,6]].to_tensor(Default::default());
// }
