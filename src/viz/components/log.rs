use ratatui::prelude::*;
use tui_logger::TuiLoggerSmartWidget;

pub struct Logs;

impl Widget for &Logs {
    fn render(self, area: Rect, buf: &mut Buffer) {
        TuiLoggerSmartWidget::default()
            .style(Style::default().white())
            .style_error(Style::default().light_red())
            .style_warn(Style::default().light_yellow())
            .style_info(Style::default().light_cyan())
            .output_separator(' ')
            .render(area, buf);
    }
}
