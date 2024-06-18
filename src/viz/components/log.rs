use crossterm::event::{Event, KeyCode};
use ratatui::{prelude::*, widgets::WidgetRef};
use tui_logger::{TuiLoggerSmartWidget, TuiWidgetEvent, TuiWidgetState};

use crate::viz::util::event_keycode;

use super::Component;

pub struct Logs {
    state: TuiWidgetState,
}

impl Logs {
    pub fn new() -> Self {
        Self {
            state: TuiWidgetState::new().set_default_display_level(log::LevelFilter::Trace),
        }
    }
}

impl WidgetRef for Logs {
    fn render_ref(&self, area: Rect, buf: &mut Buffer) {
        TuiLoggerSmartWidget::default()
            .style(Style::default().white())
            .style_error(Style::default().light_red())
            .style_warn(Style::default().light_yellow())
            .style_info(Style::default().cyan())
            .output_separator(' ')
            .state(&self.state)
            .render(area, buf);
    }
}

impl Component for Logs {
    fn handle_ui_event(&mut self, event: &Event) -> bool {
        let Some(key) = event_keycode(event) else {
            return false;
        };

        let widget_event = match key {
            KeyCode::Char(' ') => TuiWidgetEvent::SpaceKey,
            KeyCode::Esc => TuiWidgetEvent::EscapeKey,
            KeyCode::PageUp => TuiWidgetEvent::PrevPageKey,
            KeyCode::PageDown => TuiWidgetEvent::NextPageKey,
            KeyCode::Up => TuiWidgetEvent::UpKey,
            KeyCode::Down => TuiWidgetEvent::DownKey,
            KeyCode::Left => TuiWidgetEvent::LeftKey,
            KeyCode::Right => TuiWidgetEvent::RightKey,
            KeyCode::Char('=') | KeyCode::Char('+') => TuiWidgetEvent::PlusKey,
            KeyCode::Char('-') | KeyCode::Char('_') => TuiWidgetEvent::MinusKey,
            KeyCode::Char('s') => TuiWidgetEvent::HideKey,
            KeyCode::Char('f') => TuiWidgetEvent::FocusKey,
            _ => return false,
        };

        self.state.transition(widget_event);
        true
    }
}
