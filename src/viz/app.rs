use std::{
    io,
    sync::mpsc::{Receiver, TryRecvError},
    time::Duration,
};

use super::{
    components::{help::render_help, Component, Logs, Plots},
    util::event_keycode,
};
use crossterm::event::{
    self,
    Event::{self},
    KeyCode,
};
use ratatui::{prelude::*, widgets::*};

use super::tui;

const TABS: [&str; 2] = ["Plots", "Logs"];

#[derive(Default)]
pub enum AppMode {
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
    state: AppMode,
    episode: u16,
    total_episodes: u16,
    selected_tab: usize,
    show_help: bool,
    plots: Plots,
    logs: Logs,
}

impl App {
    pub fn new(plots: &[&'static str], episodes: u16) -> Self {
        Self {
            state: Default::default(),
            episode: 0,
            total_episodes: episodes,
            selected_tab: 0,
            show_help: false,
            plots: Plots::new(plots.to_vec(), episodes),
            logs: Logs::new(),
        }
    }

    fn handle_ui_event(&mut self, event: &Event) {
        let handled = match self.selected_tab {
            1 => self.logs.handle_ui_event(event),
            _ => self.plots.handle_ui_event(event),
        };

        if handled {
            return;
        }

        let Some(key) = event_keycode(event) else {
            return;
        };

        match key {
            KeyCode::Tab => {
                self.selected_tab = (self.selected_tab + 1) % TABS.len();
            }
            KeyCode::Char('q') => {
                self.state = AppMode::Quit;
            }
            KeyCode::Char('h') => {
                self.show_help ^= true;
            }
            _ => (),
        }
    }

    /// Initialize the terminal and run the main loop
    ///
    /// Restores the terminal on exit
    pub fn run(&mut self, rx: Receiver<Update>) -> io::Result<()> {
        let mut terminal = tui::init()?;

        loop {
            match self.state {
                AppMode::Train => {
                    loop {
                        match rx.try_recv() {
                            Ok(update) => {
                                self.episode = update.episode;
                                self.plots.update(update)
                            }
                            Err(TryRecvError::Empty) => break,
                            Err(TryRecvError::Disconnected) => {
                                self.state = AppMode::Error("Channel disconnected.");
                                break;
                            }
                        };
                    }

                    terminal.draw(|frame| frame.render_widget(&*self, frame.size()))?;

                    if event::poll(Duration::from_millis(16))? {
                        let event = event::read()?;
                        self.handle_ui_event(&event);
                    }
                }
                AppMode::Error(_) => todo!(),
                AppMode::Quit => break,
            }
        }

        tui::restore()
    }
}

impl WidgetRef for App {
    fn render_ref(&self, area: Rect, buf: &mut Buffer) {
        // Layout
        let [menu_area, main_area, progress_area] = Layout::vertical([
            Constraint::Length(3),
            Constraint::Fill(1),
            Constraint::Length(3),
        ])
        .areas(area);

        // Menu
        let [tabs_area, _, help_area] = Layout::horizontal([
            Constraint::Length(TABS.join(" | ").len() as u16 + 4),
            Constraint::Fill(1),
            Constraint::Length("H - Help Screen".len() as u16 + 4),
        ])
        .areas(menu_area);

        Tabs::new(TABS)
            .block(Block::new().padding(Padding::uniform(1)))
            .white()
            .bold()
            .highlight_style(Style::new().light_green())
            .select(self.selected_tab)
            .render(tabs_area, buf);

        Paragraph::new(Line::from(vec![
            Span::styled("H", Style::new().bold()),
            Span::raw(" - Help Screen"),
        ]))
        .block(Block::new().padding(Padding::uniform(1)))
        .render(help_area, buf);

        // Main
        match self.selected_tab {
            1 => self.logs.render(main_area, buf),
            _ => self.plots.render(main_area, buf),
        }

        // Progress
        Gauge::default()
            .block(
                Block::bordered()
                    .border_type(BorderType::Rounded)
                    .title("Progress"),
            )
            .gauge_style(Color::Cyan)
            .ratio((self.episode + 1) as f64 / self.total_episodes as f64)
            .render(progress_area, buf);

        // Help Popup
        if self.show_help {
            render_help(area, buf, self.selected_tab);
        }
    }
}
