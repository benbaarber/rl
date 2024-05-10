use burn::backend::{wgpu, Autodiff, Wgpu};

mod agent;
mod model;

type DQNBackend = Wgpu<wgpu::AutoGraphicsApi, f32, i32>;
type DQNAutodiffBackend = Autodiff<DQNBackend>;

static DEVICE: wgpu::WgpuDevice = wgpu::WgpuDevice::default();

const FIELD_SIZE: usize = 8;

fn main() {}
