use burn::backend::{
    wgpu::{self, WgpuDevice},
    Autodiff, Wgpu,
};
use once_cell::sync::Lazy;

mod agent;
mod model;

type DQNBackend = Wgpu<wgpu::AutoGraphicsApi, f32, i32>;
type DQNAutodiffBackend = Autodiff<DQNBackend>;

static DEVICE: Lazy<WgpuDevice> = Lazy::new(WgpuDevice::default);

const FIELD_SIZE: usize = 8;

fn main() {}
