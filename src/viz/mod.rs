use std::{
    io,
    sync::mpsc::{Receiver, TryRecvError},
    time::Duration,
};

use crossterm::{
    self as ct,
    event::{self, Event::Key, KeyCode, KeyEventKind},
};
use ratatui::{prelude::*, symbols::border, widgets::*};

use crate::util::{transpose, transpose_iter};

use self::components::Plot;

mod components;
mod tui;

#[derive(Default)]
pub enum State {
    #[default]
    Train,
    Error(&'static str),
    Quit,
}

pub struct App {
    state: State,
    plot_names: Vec<String>,
    plots: Vec<Plot>,
    selected_plot: usize,
}

impl App {
    pub fn new(plots: &[&str]) -> Self {
        Self {
            state: Default::default(),
            plot_names: plots.iter().map(|p| String::from(*p)).collect(),
            plots: plots.into_iter().map(|p| Plot::new(*p)).collect(),
            selected_plot: 0,
        }
    }

    pub fn run(&mut self, rx: Receiver<Vec<Vec<(f64, f64)>>>) -> io::Result<()> {
        let mut terminal = tui::init()?;

        loop {
            match self.state {
                State::Train => {
                    match rx.try_recv() {
                        Ok(data) => {
                            for (i, mut metric) in transpose_iter(data).enumerate() {
                                self.plots[i].data.append(&mut metric);
                            }
                        }
                        Err(TryRecvError::Empty) => {}
                        Err(TryRecvError::Disconnected) => {
                            self.state = State::Error("Channel disconnected.");
                        }
                    };

                    terminal.draw(|frame| frame.render_widget(&*self, frame.size()))?;

                    if event::poll(Duration::from_millis(16))? {
                        if let Key(key) = event::read()? {
                            if key.kind == KeyEventKind::Press && key.code == KeyCode::Char('q') {
                                self.state = State::Quit;
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
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(75), Constraint::Percentage(25)])
            .split(area);

        Block::new().title("Block 2").render(layout[1], buf);

        Tabs::new(self.plot_names.iter().map(String::as_str))
            .block(Block::bordered().title("Tabs"))
            .style(Style::default().white())
            .highlight_style(Style::default().yellow())
            .select(self.selected_plot)
            .render(layout[0], buf);

        self.plots[self.selected_plot].render(layout[0], buf);
    }
}
