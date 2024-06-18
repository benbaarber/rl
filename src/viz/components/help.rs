use ratatui::{prelude::*, widgets::*};

pub fn render_help(area: Rect, buf: &mut Buffer, selected_tab: usize) {
    let lines = vec![
        vec![
            Span::from("  q  ").light_cyan().bold(),
            Span::raw(" : Stop training and exit viz"),
        ],
        vec![
            Span::from("  h  ").light_cyan().bold(),
            Span::raw(" : Toggle help popup"),
        ],
        vec![
            Span::from(" Tab ").light_cyan().bold(),
            Span::raw(" : Switch tabs"),
        ],
    ];

    let additional_lines = match selected_tab {
        0 => vec![vec![
            Span::from("⬅ / ➡").light_cyan().bold(),
            Span::raw(" : Switch plots"),
        ]],
        1 => vec![
            vec![
                Span::from("  s  ").light_cyan().bold(),
                Span::raw(" : Toggles target selector widget hidden/visible"),
            ],
            vec![
                Span::from("  f  ").light_cyan().bold(),
                Span::raw(" : Toggle focus on the selected target only"),
            ],
            vec![
                Span::from("⬆ / ⬇").light_cyan().bold(),
                Span::raw(" : Switch log target"),
            ],
            vec![
                Span::from("⬅ / ➡").light_cyan().bold(),
                Span::raw(" : Reduce/increase shown log messages by one level"),
            ],
            vec![
                Span::from("- / +").light_cyan().bold(),
                Span::raw(" : Reduce/increase captured log messages by one level"),
            ],
            vec![
                Span::from("PgUp ").light_cyan().bold(),
                Span::raw(" : Enter Page Mode and scroll approx. half page up in log history"),
            ],
            vec![
                Span::from("PgDn ").light_cyan().bold(),
                Span::raw(" : Only in page mode, scroll 10 events down in log history"),
            ],
            vec![
                Span::from(" Esc ").light_cyan().bold(),
                Span::raw(" : Exit page mode and go back to scrolling mode"),
            ],
            vec![
                Span::from("Space").light_cyan().bold(),
                Span::raw(" : Toggles hiding of targets, which have logfilter set to off"),
            ],
        ],
        _ => vec![],
    };

    let lines = [lines, additional_lines]
        .concat()
        .into_iter()
        .map(Line::from)
        .collect::<Vec<_>>();

    let [_, center_vert, _] = Layout::vertical([
        Constraint::Fill(1),
        Constraint::Length((lines.len() + 4) as u16),
        Constraint::Fill(1),
    ])
    .areas(area);

    let [_, center, _] = Layout::horizontal([
        Constraint::Fill(1),
        Constraint::Length(100),
        Constraint::Fill(1),
    ])
    .areas(center_vert);

    Clear.render(center, buf);

    Paragraph::new(lines)
        .block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .padding(Padding::proportional(1))
                .title("Help"),
        )
        .wrap(Wrap { trim: false })
        .render(center, buf);
}
