use std::collections::HashMap;

use ratatui::{
    style::Color,
    widgets::canvas::{Painter, Shape},
};

use super::Hsl;

/// A group of points with a given color
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Points<'a> {
    /// List of points to draw
    pub coords: &'a [(f64, f64)],
    /// Density gradient of the points
    pub gradient: (Hsl, Hsl),
}

impl<'a> Shape for Points<'a> {
    fn draw(&self, painter: &mut Painter) {
        let mut density_map = HashMap::<(usize, usize), usize>::new();
        let mut grid_points = vec![];
        for (x, y) in self.coords {
            if let Some((x, y)) = painter.get_point(*x, *y) {
                grid_points.push((x, y));
                density_map
                    .entry((x / 2, y / 4)) // assuming BrailleGrid for now
                    .and_modify(|d| *d += 1)
                    .or_insert(0);
            }
        }

        if density_map.is_empty() {
            return;
        }

        let mean_density = f64::max(
            density_map.values().sum::<usize>() as f64 / density_map.len() as f64,
            5.0,
        );

        let color_map = density_map
            .into_iter()
            .map(|(p, d)| {
                (
                    p,
                    linear_gradient(self.gradient, d as f64 / (2.0 * mean_density)),
                )
            })
            .collect::<HashMap<_, _>>();

        for (x, y) in grid_points {
            let color = color_map.get(&(x / 2, y / 4)).copied().unwrap_or_else(|| {
                let Hsl(h, s, l) = self.gradient.0;
                Color::from_hsl(h, s, l)
            });
            painter.paint(x, y, color);
        }
    }
}

fn linear_gradient(gradient: (Hsl, Hsl), percent: f64) -> Color {
    let (Hsl(h1, s1, l1), Hsl(h2, s2, l2)) = gradient;
    Color::from_hsl(
        interpolate(h1, h2, percent),
        interpolate(s1, s2, percent),
        interpolate(l1, l2, percent),
    )
}

fn interpolate(a: f64, b: f64, p: f64) -> f64 {
    a + (p * (b - a))
}
