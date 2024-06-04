use std::{
    io,
    sync::mpsc::{Receiver, TryRecvError},
    time::Duration,
};

use super::components::{Logs, Plots};
use crossterm::event::{self, Event::Key, KeyCode, KeyEventKind};
use ratatui::{prelude::*, widgets::*};

use super::tui;

const TABS: [&str; 2] = ["Plots", "Logs"];

#[derive(Default)]
pub enum State {
    #[default]
    Train,
    Error(&'static str),
    Quit,
}

/// Format for updating plot data
pub struct Update {
    pub episode: u16,
    pub data: Vec<f64>,
}

/// The root TUI component which holds the main app state and runs the render loop
pub struct App {
    state: State,
    episode: u16,
    total_episodes: u16,
    selected_tab: usize,
    plots: Plots,
}

impl App {
    pub fn new(plots: &[&'static str], episodes: u16) -> Self {
        Self {
            state: Default::default(),
            episode: 0,
            total_episodes: episodes,
            selected_tab: 0,
            plots: Plots::new(plots.to_vec(), episodes),
        }
    }

    /// Initialize the terminal and run the main loop
    ///
    /// Restores the terminal on exit
    pub fn run(&mut self, plot_rx: Receiver<Update>) -> io::Result<()> {
        let mut terminal = tui::init()?;

        loop {
            match self.state {
                State::Train => {
                    loop {
                        match plot_rx.try_recv() {
                            Ok(update) => {
                                self.episode = update.episode;
                                self.plots.update(update)
                            }
                            Err(TryRecvError::Empty) => break,
                            Err(TryRecvError::Disconnected) => {
                                self.state = State::Error("Channel disconnected.");
                                break;
                            }
                        };
                    }

                    terminal.draw(|frame| frame.render_widget(&*self, frame.size()))?;

                    if event::poll(Duration::from_millis(16))? {
                        if let Key(key) = event::read()? {
                            if key.kind != KeyEventKind::Press {
                                continue;
                            }
                            match key.code {
                                KeyCode::Tab => {
                                    self.selected_tab = (self.selected_tab + 1) % TABS.len();
                                }
                                KeyCode::Char('q') => {
                                    self.state = State::Quit;
                                }
                                KeyCode::Left => {
                                    self.plots.prev_plot();
                                }
                                KeyCode::Right => {
                                    self.plots.next_plot();
                                }
                                _ => {}
                            }
                        }
                    }
                }
                State::Error(_) => todo!(),
                State::Quit => break,
            }
        }

        tui::restore()
    }
}

impl Widget for &App {
    fn render(self, area: Rect, buf: &mut Buffer) {
        // Layout
        let [menu_area, main_area, progress_area] = Layout::vertical([
            Constraint::Length(3),
            Constraint::Fill(1),
            Constraint::Length(3),
        ])
        .areas(area);

        // Menu
        Tabs::new(TABS)
            .block(Block::default().padding(Padding::uniform(1)))
            .white()
            .bold()
            .highlight_style(Style::default().light_green())
            .select(self.selected_tab)
            .render(menu_area, buf);

        // Main
        match self.selected_tab {
            0 => self.plots.render(main_area, buf),
            1 => Logs.render(main_area, buf),
            _ => {}
        }

        // Progress Bar
        Gauge::default()
            .block(
                Block::bordered()
                    .border_type(BorderType::Rounded)
                    .title("Progress"),
            )
            .gauge_style(Color::Cyan)
            .ratio(self.episode as f64 / self.total_episodes as f64)
            .render(progress_area, buf);
    }
}
