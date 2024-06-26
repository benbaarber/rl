use std::{
    io,
    sync::mpsc::{self, Sender},
    thread::{self, JoinHandle},
};

use app::App;

/// Root TUI component
pub mod app;
/// Components that make up the viz TUI
mod components;
/// Boilerplate
mod tui;
/// TUI utils
mod util;

pub use app::Update;

/// Initialize the viz training dashboard TUI in a separate thread
///
/// Sets up a global [logger](log) that sends log data to the TUI through the log macros
///
/// ### Arguments
/// - `plots` - The names of the plots to render in the TUI
/// - `episodes` - The number of episodes to show on the x-axis
///
/// ### Returns
/// A tuple `(handle, tx)`
/// - `handle` - The [JoinHandle] of the TUI thread
/// - `tx` - A [mpsc::Sender] for transmitting plot data updates to the TUI
pub fn init(plots: &[&'static str], episodes: u16) -> (JoinHandle<io::Result<()>>, Sender<Update>) {
    tui_logger::init_logger(log::LevelFilter::Trace).unwrap();
    tui_logger::set_default_level(log::LevelFilter::Warn);
    tui_logger::set_level_for_target("tui", log::LevelFilter::Trace);
    tui_logger::move_events();

    let mut app = App::new(plots, episodes);
    let (tx, rx) = mpsc::channel();
    let handle = thread::spawn(move || app.run(rx));

    (handle, tx)
}
