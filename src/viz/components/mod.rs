pub mod heatmap_scatter_plot;
pub mod help;
pub mod log;
pub mod plot;

use crossterm::event::Event;
pub use log::Logs;
pub use plot::Plots;
use ratatui::widgets::WidgetRef;

pub trait Component: WidgetRef {
    fn handle_ui_event(&mut self, event: &Event) -> bool;
}
