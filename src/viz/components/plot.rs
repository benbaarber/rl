use ratatui::{
    prelude::*,
    style::Stylize,
    widgets::{block::Title, *},
};

pub struct Plot {
    pub x_title: String,
    pub y_title: String,
    x_bounds: [f64; 2],
    y_bounds: [f64; 2],
    x_labels: Vec<String>,
    y_labels: Vec<String>,
    data: Vec<(f64, f64)>,
}

impl Plot {
    pub fn new(y_label: &str) -> Self {
        Self {
            x_title: String::from("Episode"),
            y_title: String::from(y_label),
            x_bounds: [f64::MAX, f64::MIN],
            y_bounds: [f64::MAX, f64::MIN],
            x_labels: Vec::new(),
            y_labels: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Provide initial x bounds
    pub fn with_x_bounds(mut self, x_bounds: [f64; 2]) -> Self {
        self.x_bounds = x_bounds;
        self.x_labels = self.x_bounds.iter().map(|x| format!("{x:.2}")).collect();
        self
    }

    /// Provide initial y bounds
    pub fn with_y_bounds(mut self, y_bounds: [f64; 2]) -> Self {
        self.y_bounds = y_bounds;
        self.y_labels = self.y_bounds.iter().map(|x| format!("{x:.2}")).collect();
        self
    }

    pub fn update(&mut self, point: (f64, f64)) {
        let mut x_bounds_changed = false;
        let mut y_bounds_changed = false;
        if point.0 > self.x_bounds[1] {
            self.x_bounds[1] = point.0;
            x_bounds_changed = true;
        }
        if point.0 < self.x_bounds[0] {
            self.x_bounds[0] = point.0;
            x_bounds_changed = true;
        }
        if point.1 < self.y_bounds[0] {
            self.y_bounds[0] = point.1;
            y_bounds_changed = true;
        }
        if point.1 > self.y_bounds[1] {
            self.y_bounds[1] = point.1;
            y_bounds_changed = true;
        }

        if x_bounds_changed {
            self.x_labels = self.x_bounds.iter().map(|x| format!("{x:.2}")).collect();
        }
        if y_bounds_changed {
            self.y_labels = self.y_bounds.iter().map(|x| format!("{x:.2}")).collect();
        }

        self.data.push(point);
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
            .title(self.x_title.as_str())
            .dark_gray()
            .labels(
                self.x_labels
                    .clone()
                    .into_iter()
                    .map(|l| l.bold())
                    .collect(),
            )
            .bounds(self.x_bounds);

        let y_axis = Axis::default()
            .title(self.y_title.as_str())
            .dark_gray()
            .labels(
                self.y_labels
                    .clone()
                    .into_iter()
                    .map(|l| l.bold())
                    .collect(),
            )
            .bounds(self.y_bounds);

        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .title(Title::from("Plots").alignment(Alignment::Center))
            .padding(Padding::uniform(10));

        let chart = Chart::new(vec![dataset])
            .block(block)
            .x_axis(x_axis)
            .y_axis(y_axis);

        chart.render(area, buf);
    }
}
