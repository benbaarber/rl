use ratatui::{
    prelude::*,
    widgets::{block::Title, *},
};

pub struct Plot {
    pub x_label: String,
    pub y_label: String,
    pub data: Vec<(f64, f64)>,
}

impl Plot {
    pub fn new(y_label: &str) -> Self {
        Self {
            x_label: String::from("Episode"),
            y_label: String::from(y_label),
            data: Vec::new(),
        }
    }
}

impl Widget for &Plot {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let dataset = Dataset::default()
            .marker(Marker::Braille)
            .graph_type(GraphType::Scatter)
            .cyan()
            .data(&self.data);

        let x_axis = Axis::default()
            .title(&*self.x_label)
            .labels(vec!["0".into(), "10".into()])
            .bounds([0.0, 10.0])
            .dark_gray();

        let y_axis = Axis::default()
            .title(&*self.y_label)
            .labels(vec!["0".into(), "30".into()])
            .bounds([0.0, 30.0])
            .dark_gray();

        let block = Block::bordered()
            .title(Title::from("Plots").alignment(Alignment::Center))
            .padding(Padding::uniform(10));

        let chart = Chart::new(vec![dataset])
            .block(block)
            .x_axis(x_axis)
            .y_axis(y_axis);

        chart.render(area, buf);
    }
}

pub struct PlotSection;
