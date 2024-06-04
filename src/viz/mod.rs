use std::{
    io,
    sync::mpsc::{self, Sender},
    thread::{self, JoinHandle},
};

use app::App;

pub mod app;
mod components;
mod tui;

pub use app::Update;

pub fn init(keys: &[&'static str], episodes: u16) -> (JoinHandle<io::Result<()>>, Sender<Update>) {
    let _ = tui_logger::init_logger(log::LevelFilter::Trace);
    tui_logger::set_default_level(log::LevelFilter::Trace);

    let mut app = App::new(keys, episodes);
    let (plot_tx, plot_rx) = mpsc::channel();
    let handle = thread::spawn(move || app.run(plot_rx));

    (handle, plot_tx)
}
