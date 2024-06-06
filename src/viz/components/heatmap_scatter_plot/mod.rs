use std::cmp::max;

use strum::{Display, EnumString};
use unicode_width::UnicodeWidthStr;

use points::Points;
use ratatui::{
    prelude::*,
    style::Styled,
    widgets::{canvas::Canvas, Block, WidgetRef},
};

mod points;

#[derive(Clone, Copy, PartialEq, Default, Debug)]
pub struct Hsl(pub f64, pub f64, pub f64);

/// An X or Y axis for the [`Chart`] widget
///
/// An axis can have a [title](Axis::title) which will be displayed at the end of the axis. For an
/// X axis this is the right, for a Y axis, this is the top.
///
/// You can also set the bounds and labels on this axis using respectively [`Axis::bounds`] and
/// [`Axis::labels`].
///
/// See [`Chart::x_axis`] and [`Chart::y_axis`] to set an axis on a chart.
///
/// # Example
///
/// ```rust
/// use ratatui::{prelude::*, widgets::*};
/// let axis = Axis::default()
///     .title("X Axis")
///     .style(Style::default().gray())
///     .bounds([0.0, 50.0])
///     .labels(vec!["0".bold(), "25".into(), "50".bold()]);
/// ```
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Axis<'a> {
    /// Title displayed next to axis end
    title: Option<Line<'a>>,
    /// Bounds for the axis (all data points outside these limits will not be represented)
    bounds: [f64; 2],
    /// A list of labels to put to the left or below the axis
    labels: Option<Vec<Span<'a>>>,
    /// The style used to draw the axis itself
    style: Style,
    /// The alignment of the labels of the Axis
    labels_alignment: Alignment,
}

impl<'a> Axis<'a> {
    /// Sets the axis title
    ///
    /// It will be displayed at the end of the axis. For an X axis this is the right, for a Y axis,
    /// this is the top.
    ///
    /// This is a fluent setter method which must be chained or used as it consumes self
    #[must_use = "method moves the value of self and returns the modified value"]
    pub fn title<T>(mut self, title: T) -> Self
    where
        T: Into<Line<'a>>,
    {
        self.title = Some(title.into());
        self
    }

    /// Sets the bounds of this axis
    ///
    /// In other words, sets the min and max value on this axis.
    ///
    /// This is a fluent setter method which must be chained or used as it consumes self
    #[must_use = "method moves the value of self and returns the modified value"]
    pub const fn bounds(mut self, bounds: [f64; 2]) -> Self {
        self.bounds = bounds;
        self
    }

    /// Sets the axis labels
    ///
    /// - For the X axis, the labels are displayed left to right.
    /// - For the Y axis, the labels are displayed bottom to top.
    ///
    /// Currently, you need to give at least two labels or the render will panic. Also, giving
    /// more than 3 labels is currently broken and the middle labels won't be in the correct
    /// position, see [issue 334].
    ///
    /// [issue 334]: https://github.com/ratatui-org/ratatui/issues/334
    ///
    /// This is a fluent setter method which must be chained or used as it consumes self
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ratatui::{prelude::*, widgets::*};
    /// let axis =
    ///     Axis::default()
    ///         .bounds([0.0, 50.0])
    ///         .labels(vec!["0".bold(), "25".into(), "50".bold()]);
    /// ```
    #[must_use = "method moves the value of self and returns the modified value"]
    pub fn labels(mut self, labels: Vec<Span<'a>>) -> Self {
        self.labels = Some(labels);
        self
    }

    /// Sets the axis style
    ///
    /// This is a fluent setter method which must be chained or used as it consumes self
    ///
    /// `style` accepts any type that is convertible to [`Style`] (e.g. [`Style`], [`Color`], or
    /// your own type that implements [`Into<Style>`]).
    ///
    /// # Example
    ///
    /// [`Axis`] also implements [`Stylize`](crate::style::Stylize) which mean you can style it
    /// like so
    ///
    /// ```rust
    /// # use ratatui::{prelude::*, widgets::*};
    /// let axis = Axis::default().red();
    /// ```
    #[must_use = "method moves the value of self and returns the modified value"]
    pub fn style<S: Into<Style>>(mut self, style: S) -> Self {
        self.style = style.into();
        self
    }

    /// Sets the labels alignment of the axis
    ///
    /// The alignment behaves differently based on the axis:
    /// - Y axis: The labels are aligned within the area on the left of the axis
    /// - X axis: The first X-axis label is aligned relative to the Y-axis
    ///
    /// On the X axis, this parameter only affects the first label.
    #[must_use = "method moves the value of self and returns the modified value"]
    pub const fn labels_alignment(mut self, alignment: Alignment) -> Self {
        self.labels_alignment = alignment;
        self
    }
}

/// Used to determine which style of graphing to use
#[derive(Debug, Default, Display, EnumString, Clone, Copy, Eq, PartialEq, Hash)]
pub enum GraphType {
    /// Draw each point. This is the default.
    #[default]
    Scatter,
    /// Draw a line between each following point.
    ///
    /// The order of the lines will be the same as the order of the points in the dataset, which
    /// allows this widget to draw lines both left-to-right and right-to-left
    Line,
}

/// Allow users to specify the position of a legend in a [`Chart`]
///
/// See [`Chart::legend_position`]
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub enum LegendPosition {
    /// Legend is centered on top
    Top,
    /// Legend is in the top-right corner. This is the **default**.
    #[default]
    TopRight,
    /// Legend is in the top-left corner
    TopLeft,
    /// Legend is centered on the left
    Left,
    /// Legend is centered on the right
    Right,
    /// Legend is centered on the bottom
    Bottom,
    /// Legend is in the bottom-right corner
    BottomRight,
    /// Legend is in the bottom-left corner
    BottomLeft,
}

impl LegendPosition {
    fn layout(
        self,
        area: Rect,
        legend_width: u16,
        legend_height: u16,
        x_title_width: u16,
        y_title_width: u16,
    ) -> Option<Rect> {
        let mut height_margin = i32::from(area.height - legend_height);
        if x_title_width != 0 {
            height_margin -= 1;
        }
        if y_title_width != 0 {
            height_margin -= 1;
        }
        if height_margin < 0 {
            return None;
        };

        let (x, y) = match self {
            Self::TopRight => {
                if legend_width + y_title_width > area.width {
                    (area.right() - legend_width, area.top() + 1)
                } else {
                    (area.right() - legend_width, area.top())
                }
            }
            Self::TopLeft => {
                if y_title_width != 0 {
                    (area.left(), area.top() + 1)
                } else {
                    (area.left(), area.top())
                }
            }
            Self::Top => {
                let x = (area.width - legend_width) / 2;
                if area.left() + y_title_width > x {
                    (area.left() + x, area.top() + 1)
                } else {
                    (area.left() + x, area.top())
                }
            }
            Self::Left => {
                let mut y = (area.height - legend_height) / 2;
                if y_title_width != 0 {
                    y += 1;
                }
                if x_title_width != 0 {
                    y = y.saturating_sub(1);
                }
                (area.left(), area.top() + y)
            }
            Self::Right => {
                let mut y = (area.height - legend_height) / 2;
                if y_title_width != 0 {
                    y += 1;
                }
                if x_title_width != 0 {
                    y = y.saturating_sub(1);
                }
                (area.right() - legend_width, area.top() + y)
            }
            Self::BottomLeft => {
                if x_title_width + legend_width > area.width {
                    (area.left(), area.bottom() - legend_height - 1)
                } else {
                    (area.left(), area.bottom() - legend_height)
                }
            }
            Self::BottomRight => {
                if x_title_width != 0 {
                    (
                        area.right() - legend_width,
                        area.bottom() - legend_height - 1,
                    )
                } else {
                    (area.right() - legend_width, area.bottom() - legend_height)
                }
            }
            Self::Bottom => {
                let x = area.left() + (area.width - legend_width) / 2;
                if x + legend_width > area.right() - x_title_width {
                    (x, area.bottom() - legend_height - 1)
                } else {
                    (x, area.bottom() - legend_height)
                }
            }
        };

        Some(Rect::new(x, y, legend_width, legend_height))
    }
}

/// A group of data points
///
/// This is the main element composing a [`Chart`].
///
/// A dataset can be [named](Dataset::name). Only named datasets will be rendered in the legend.
///
/// After that, you can pass it data with [`Dataset::data`]. Data is an array of `f64` tuples
/// (`(f64, f64)`), the first element being X and the second Y. It's also worth noting that, unlike
/// the [`Rect`], here the Y axis is bottom to top, as in math.
///
/// You can also customize the rendering by using [`Dataset::marker`] and [`Dataset::graph_type`].
///
/// # Example
///
/// This example draws a red line between two points.
///
/// ```rust
/// use ratatui::{prelude::*, symbols::Marker, widgets::*};
///
/// let dataset = Dataset::default()
///     .name("dataset 1")
///     .data(&[(1., 1.), (5., 5.)])
///     .marker(Marker::Braille)
///     .graph_type(GraphType::Line)
///     .red();
/// ```
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Dataset<'a> {
    /// Name of the dataset (used in the legend if shown)
    name: Option<Line<'a>>,
    /// A reference to the actual data
    data: &'a [(f64, f64)],
    /// Symbol used for each points of this dataset
    marker: symbols::Marker,
    /// Style used to plot this dataset
    style: Style,
    /// Start and end HSL colors of density gradient
    gradient: (Hsl, Hsl),
}

impl<'a> Dataset<'a> {
    /// Sets the name of the dataset
    ///
    /// The dataset's name is used when displaying the chart legend. Datasets don't require a name
    /// and can be created without specifying one. Once assigned, a name can't be removed, only
    /// changed
    ///
    /// The name can be styled (see [`Line`] for that), but the dataset's style will always have
    /// precedence.
    ///
    /// This is a fluent setter method which must be chained or used as it consumes self
    #[must_use = "method moves the value of self and returns the modified value"]
    pub fn name<S>(mut self, name: S) -> Self
    where
        S: Into<Line<'a>>,
    {
        self.name = Some(name.into());
        self
    }

    /// Sets the data points of this dataset
    ///
    /// Points will then either be rendered as scattered points or with lines between them
    /// depending on [`Dataset::graph_type`].
    ///
    /// Data consist in an array of `f64` tuples (`(f64, f64)`), the first element being X and the
    /// second Y. It's also worth noting that, unlike the [`Rect`], here the Y axis is bottom to
    /// top, as in math.
    ///
    /// This is a fluent setter method which must be chained or used as it consumes self
    #[must_use = "method moves the value of self and returns the modified value"]
    pub const fn data(mut self, data: &'a [(f64, f64)]) -> Self {
        self.data = data;
        self
    }

    /// Sets the kind of character to use to display this dataset
    ///
    /// You can use dots (`•`), blocks (`█`), bars (`▄`), braille (`⠓`, `⣇`, `⣿`) or half-blocks
    /// (`█`, `▄`, and `▀`). See [`symbols::Marker`] for more details.
    ///
    /// Note [`Marker::Braille`](symbols::Marker::Braille) requires a font that supports Unicode
    /// Braille Patterns.
    ///
    /// This is a fluent setter method which must be chained or used as it consumes self
    #[must_use = "method moves the value of self and returns the modified value"]
    pub const fn marker(mut self, marker: symbols::Marker) -> Self {
        self.marker = marker;
        self
    }

    /// Sets the style of this dataset
    ///
    /// The given style will be used to draw the legend and the data points. Currently the legend
    /// will use the entire style whereas the data points will only use the foreground.
    ///
    /// `style` accepts any type that is convertible to [`Style`] (e.g. [`Style`], [`Color`], or
    /// your own type that implements [`Into<Style>`]).
    ///
    /// This is a fluent setter method which must be chained or used as it consumes self
    ///
    /// # Example
    ///
    /// [`Dataset`] also implements [`Stylize`](crate::style::Stylize) which mean you can style it
    /// like so
    ///
    /// ```rust
    /// # use ratatui::{prelude::*, widgets::*};
    /// let dataset = Dataset::default().red();
    /// ```
    #[must_use = "method moves the value of self and returns the modified value"]
    pub fn style<S: Into<Style>>(mut self, style: S) -> Self {
        self.style = style.into();
        self
    }

    /// Set the gradient colors of this dataset
    #[must_use = "method moves the value of self and returns the modified value"]
    pub fn gradient(mut self, gradient: (Hsl, Hsl)) -> Self {
        self.gradient = gradient;
        self
    }
}

/// A container that holds all the infos about where to display each elements of the chart (axis,
/// labels, legend, ...).
struct ChartLayout {
    /// Location of the title of the x axis
    title_x: Option<(u16, u16)>,
    /// Location of the title of the y axis
    title_y: Option<(u16, u16)>,
    /// Location of the first label of the x axis
    label_x: Option<u16>,
    /// Location of the first label of the y axis
    label_y: Option<u16>,
    /// Y coordinate of the horizontal axis
    axis_x: Option<u16>,
    /// X coordinate of the vertical axis
    axis_y: Option<u16>,
    /// Area of the legend
    legend_area: Option<Rect>,
    /// Area of the graph
    graph_area: Rect,
}

/// A widget to plot one or more [`Dataset`] in a cartesian coordinate system
///
/// To use this widget, start by creating one or more [`Dataset`]. With it, you can set the
/// [data points](Dataset::data), the [name](Dataset::name) or the
/// [chart type](Dataset::graph_type). See [`Dataset`] for a complete documentation of what is
/// possible.
///
/// Then, you'll usually want to configure the [`Axis`]. Axis [titles](Axis::title),
/// [bounds](Axis::bounds) and [labels](Axis::labels) can be configured on both axis. See [`Axis`]
/// for a complete documentation of what is possible.
///
/// Finally, you can pass all of that to the `Chart` via [`Chart::new`], [`Chart::x_axis`] and
/// [`Chart::y_axis`].
///
/// Additionally, `Chart` allows configuring the legend [position](Chart::legend_position) and
/// [hiding constraints](Chart::hidden_legend_constraints).
///
/// # Examples
///
/// ```
/// use ratatui::{prelude::*, widgets::*};
///
/// // Create the datasets to fill the chart with
/// let datasets = vec![
///     // Scatter chart
///     Dataset::default()
///         .name("data1")
///         .marker(symbols::Marker::Dot)
///         .graph_type(GraphType::Scatter)
///         .style(Style::default().cyan())
///         .data(&[(0.0, 5.0), (1.0, 6.0), (1.5, 6.434)]),
///     // Line chart
///     Dataset::default()
///         .name("data2")
///         .marker(symbols::Marker::Braille)
///         .graph_type(GraphType::Line)
///         .style(Style::default().magenta())
///         .data(&[(4.0, 5.0), (5.0, 8.0), (7.66, 13.5)]),
/// ];
///
/// // Create the X axis and define its properties
/// let x_axis = Axis::default()
///     .title("X Axis".red())
///     .style(Style::default().white())
///     .bounds([0.0, 10.0])
///     .labels(vec!["0.0".into(), "5.0".into(), "10.0".into()]);
///
/// // Create the Y axis and define its properties
/// let y_axis = Axis::default()
///     .title("Y Axis".red())
///     .style(Style::default().white())
///     .bounds([0.0, 10.0])
///     .labels(vec!["0.0".into(), "5.0".into(), "10.0".into()]);
///
/// // Create the chart and link all the parts together
/// let chart = Chart::new(dataset)
///     .block(Block::new().title("Chart"))
///     .x_axis(x_axis)
///     .y_axis(y_axis);
/// ```
#[derive(Debug, Default, Clone, PartialEq)]
pub struct HeatmapScatterPlot<'a> {
    /// A block to display around the widget eventually
    block: Option<Block<'a>>,
    /// The horizontal axis
    x_axis: Axis<'a>,
    /// The vertical axis
    y_axis: Axis<'a>,
    /// A reference to the dataset
    dataset: Dataset<'a>,
    /// The widget base style
    style: Style,
    /// Constraints used to determine whether the legend should be shown or not
    hidden_legend_constraints: (Constraint, Constraint),
    /// The position determine where the length is shown or hide regardless of
    /// `hidden_legend_constraints`
    legend_position: Option<LegendPosition>,
}

impl<'a> HeatmapScatterPlot<'a> {
    /// Creates a chart with the given [datasets](Dataset)
    ///
    /// A chart can render multiple datasets.
    ///
    /// # Example
    ///
    /// This creates a simple chart with one [`Dataset`]
    ///
    /// ```rust
    /// # use ratatui::{prelude::*, widgets::*};
    /// # let data_points = vec![];
    /// let chart = Chart::new(vec![Dataset::default().data(&data_points)]);
    /// ```
    ///
    /// This creates a chart with multiple [`Dataset`]s
    ///
    /// ```rust
    /// # use ratatui::{prelude::*, widgets::*};
    /// # let data_points = vec![];
    /// # let data_points2 = vec![];
    /// let chart = Chart::new(vec![
    ///     Dataset::default().data(&data_points),
    ///     Dataset::default().data(&data_points2),
    /// ]);
    /// ```
    pub fn new(dataset: Dataset<'a>) -> Self {
        Self {
            block: None,
            x_axis: Axis::default(),
            y_axis: Axis::default(),
            style: Style::default(),
            dataset,
            hidden_legend_constraints: (Constraint::Ratio(1, 4), Constraint::Ratio(1, 4)),
            legend_position: Some(LegendPosition::default()),
        }
    }

    /// Wraps the chart with the given [`Block`]
    ///
    /// This is a fluent setter method which must be chained or used as it consumes self
    #[must_use = "method moves the value of self and returns the modified value"]
    pub fn block(mut self, block: Block<'a>) -> Self {
        self.block = Some(block);
        self
    }

    /// Sets the style of the entire chart
    ///
    /// `style` accepts any type that is convertible to [`Style`] (e.g. [`Style`], [`Color`], or
    /// your own type that implements [`Into<Style>`]).
    ///
    /// Styles of [`Axis`] and [`Dataset`] will have priority over this style.
    ///
    /// This is a fluent setter method which must be chained or used as it consumes self
    #[must_use = "method moves the value of self and returns the modified value"]
    pub fn style<S: Into<Style>>(mut self, style: S) -> Self {
        self.style = style.into();
        self
    }

    /// Sets the X [`Axis`]
    ///
    /// The default is an empty [`Axis`], i.e. only a line.
    ///
    /// This is a fluent setter method which must be chained or used as it consumes self
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ratatui::{prelude::*, widgets::*};
    /// let chart = Chart::new(vec![]).x_axis(
    ///     Axis::default()
    ///         .title("X Axis")
    ///         .bounds([0.0, 20.0])
    ///         .labels(vec!["0".into(), "20".into()]),
    /// );
    /// ```
    #[must_use = "method moves the value of self and returns the modified value"]
    pub fn x_axis(mut self, axis: Axis<'a>) -> Self {
        self.x_axis = axis;
        self
    }

    /// Sets the Y [`Axis`]
    ///
    /// The default is an empty [`Axis`], i.e. only a line.
    ///
    /// This is a fluent setter method which must be chained or used as it consumes self
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ratatui::{prelude::*, widgets::*};
    /// let chart = Chart::new(vec![]).y_axis(
    ///     Axis::default()
    ///         .title("Y Axis")
    ///         .bounds([0.0, 20.0])
    ///         .labels(vec!["0".into(), "20".into()]),
    /// );
    /// ```
    #[must_use = "method moves the value of self and returns the modified value"]
    pub fn y_axis(mut self, axis: Axis<'a>) -> Self {
        self.y_axis = axis;
        self
    }

    /// Sets the constraints used to determine whether the legend should be shown or not.
    ///
    /// The tuple's first constraint is used for the width and the second for the height. If the
    /// legend takes more space than what is allowed by any constraint, the legend is hidden.
    /// [`Constraint::Min`] is an exception and will always show the legend.
    ///
    /// If this is not set, the default behavior is to hide the legend if it is greater than 25% of
    /// the chart, either horizontally or vertically.
    ///
    /// This is a fluent setter method which must be chained or used as it consumes self
    ///
    /// # Examples
    ///
    /// Hide the legend when either its width is greater than 33% of the total widget width or if
    /// its height is greater than 25% of the total widget height.
    ///
    /// ```
    /// # use ratatui::{prelude::*, widgets::*};
    /// let constraints = (Constraint::Ratio(1, 3), Constraint::Ratio(1, 4));
    /// let chart = Chart::new(vec![]).hidden_legend_constraints(constraints);
    /// ```
    ///
    /// Always show the legend, note the second constraint doesn't matter in this case since the
    /// first one is always true.
    ///
    /// ```
    /// # use ratatui::{prelude::*, widgets::*};
    /// let constraints = (Constraint::Min(0), Constraint::Ratio(1, 4));
    /// let chart = Chart::new(vec![]).hidden_legend_constraints(constraints);
    /// ```
    ///
    /// Always hide the legend. Note this can be accomplished more explicitly by passing `None` to
    /// [`Chart::legend_position`].
    ///
    /// ```
    /// # use ratatui::{prelude::*, widgets::*};
    /// let constraints = (Constraint::Length(0), Constraint::Ratio(1, 4));
    /// let chart = Chart::new(vec![]).hidden_legend_constraints(constraints);
    /// ```
    #[must_use = "method moves the value of self and returns the modified value"]
    pub const fn hidden_legend_constraints(
        mut self,
        constraints: (Constraint, Constraint),
    ) -> Self {
        self.hidden_legend_constraints = constraints;
        self
    }

    /// Sets the position of a legend or hide it
    ///
    /// The default is [`LegendPosition::TopRight`].
    ///
    /// If [`None`] is given, hide the legend even if [`hidden_legend_constraints`] determines it
    /// should be shown. In contrast, if `Some(...)` is given, [`hidden_legend_constraints`] might
    /// still decide whether to show the legend or not.
    ///
    /// See [`LegendPosition`] for all available positions.
    ///
    /// [`hidden_legend_constraints`]: Self::hidden_legend_constraints
    ///
    /// This is a fluent setter method which must be chained or used as it consumes self
    ///
    /// # Examples
    ///
    /// Show the legend on the top left corner.
    ///
    /// ```
    /// # use ratatui::widgets::{Chart, LegendPosition};
    /// let chart: Chart = Chart::new(vec![]).legend_position(Some(LegendPosition::TopLeft));
    /// ```
    ///
    /// Hide the legend altogether
    ///
    /// ```
    /// # use ratatui::widgets::{Chart, LegendPosition};
    /// let chart = Chart::new(vec![]).legend_position(None);
    /// ```
    #[must_use = "method moves the value of self and returns the modified value"]
    pub const fn legend_position(mut self, position: Option<LegendPosition>) -> Self {
        self.legend_position = position;
        self
    }

    /// Compute the internal layout of the chart given the area. If the area is too small some
    /// elements may be automatically hidden
    fn layout(&self, area: Rect) -> Option<ChartLayout> {
        if area.height == 0 || area.width == 0 {
            return None;
        }
        let mut x = area.left();
        let mut y = area.bottom() - 1;

        let mut label_x = None;
        if self.x_axis.labels.is_some() && y > area.top() {
            label_x = Some(y);
            y -= 1;
        }

        let label_y = self.y_axis.labels.as_ref().and(Some(x));
        x += self.max_width_of_labels_left_of_y_axis(area, self.y_axis.labels.is_some());

        let mut axis_x = None;
        if self.x_axis.labels.is_some() && y > area.top() {
            axis_x = Some(y);
            y -= 1;
        }

        let mut axis_y = None;
        if self.y_axis.labels.is_some() && x + 1 < area.right() {
            axis_y = Some(x);
            x += 1;
        }

        let graph_width = area.right().saturating_sub(x);
        let graph_height = y.saturating_sub(area.top()).saturating_add(1);
        debug_assert_ne!(
            graph_width, 0,
            "Axis and labels should have been hidden due to the small area"
        );
        debug_assert_ne!(
            graph_height, 0,
            "Axis and labels should have been hidden due to the small area"
        );
        let graph_area = Rect::new(x, area.top(), graph_width, graph_height);

        let mut title_x = None;
        if let Some(ref title) = self.x_axis.title {
            let w = title.width() as u16;
            if w < graph_area.width && graph_area.height > 2 {
                title_x = Some((x + graph_area.width - w, y));
            }
        }

        let mut title_y = None;
        if let Some(ref title) = self.y_axis.title {
            let w = title.width() as u16;
            if w + 1 < graph_area.width && graph_area.height > 2 {
                title_y = Some((x, area.top()));
            }
        }

        let legend_area = None;
        // if let Some(legend_position) = self.legend_position {
        //     let legends = self
        //         .datasets
        //         .iter()
        //         .filter_map(|d| Some(d.name.as_ref()?.width() as u16));

        //     if let Some(inner_width) = legends.clone().max() {
        //         let legend_width = inner_width + 2;
        //         let legend_height = legends.count() as u16 + 2;

        //         let [max_legend_width] = Layout::horizontal([self.hidden_legend_constraints.0])
        //             .flex(Flex::Start)
        //             .areas(graph_area);

        //         let [max_legend_height] = Layout::vertical([self.hidden_legend_constraints.1])
        //             .flex(Flex::Start)
        //             .areas(graph_area);

        //         if inner_width > 0
        //             && legend_width <= max_legend_width.width
        //             && legend_height <= max_legend_height.height
        //         {
        //             legend_area = legend_position.layout(
        //                 graph_area,
        //                 legend_width,
        //                 legend_height,
        //                 title_x
        //                     .and(self.x_axis.title.as_ref())
        //                     .map(|t| t.width() as u16)
        //                     .unwrap_or_default(),
        //                 title_y
        //                     .and(self.y_axis.title.as_ref())
        //                     .map(|t| t.width() as u16)
        //                     .unwrap_or_default(),
        //             );
        //         }
        //     }
        // }
        Some(ChartLayout {
            title_x,
            title_y,
            label_x,
            label_y,
            axis_x,
            axis_y,
            legend_area,
            graph_area,
        })
    }

    fn max_width_of_labels_left_of_y_axis(&self, area: Rect, has_y_axis: bool) -> u16 {
        let mut max_width = self
            .y_axis
            .labels
            .as_ref()
            .map(|l| l.iter().map(Span::width).max().unwrap_or_default() as u16)
            .unwrap_or_default();

        if let Some(first_x_label) = self
            .x_axis
            .labels
            .as_ref()
            .and_then(|labels| labels.first())
        {
            let first_label_width = first_x_label.content.width() as u16;
            let width_left_of_y_axis = match self.x_axis.labels_alignment {
                Alignment::Left => {
                    // The last character of the label should be below the Y-Axis when it exists,
                    // not on its left
                    let y_axis_offset = u16::from(has_y_axis);
                    first_label_width.saturating_sub(y_axis_offset)
                }
                Alignment::Center => first_label_width / 2,
                Alignment::Right => 0,
            };
            max_width = max(max_width, width_left_of_y_axis);
        }
        // labels of y axis and first label of x axis can take at most 1/3rd of the total width
        max_width.min(area.width / 3)
    }

    fn render_x_labels(
        &self,
        buf: &mut Buffer,
        layout: &ChartLayout,
        chart_area: Rect,
        graph_area: Rect,
    ) {
        let Some(y) = layout.label_x else { return };
        let labels = self.x_axis.labels.as_ref().unwrap();
        let labels_len = labels.len() as u16;
        if labels_len < 2 {
            return;
        }

        let width_between_ticks = graph_area.width / labels_len;

        let label_area = self.first_x_label_area(
            y,
            labels.first().unwrap().width() as u16,
            width_between_ticks,
            chart_area,
            graph_area,
        );

        let label_alignment = match self.x_axis.labels_alignment {
            Alignment::Left => Alignment::Right,
            Alignment::Center => Alignment::Center,
            Alignment::Right => Alignment::Left,
        };

        Self::render_label(buf, labels.first().unwrap(), label_area, label_alignment);

        for (i, label) in labels[1..labels.len() - 1].iter().enumerate() {
            // We add 1 to x (and width-1 below) to leave at least one space before each
            // intermediate labels
            let x = graph_area.left() + (i + 1) as u16 * width_between_ticks + 1;
            let label_area = Rect::new(x, y, width_between_ticks.saturating_sub(1), 1);

            Self::render_label(buf, label, label_area, Alignment::Center);
        }

        let x = graph_area.right() - width_between_ticks;
        let label_area = Rect::new(x, y, width_between_ticks, 1);
        // The last label should be aligned Right to be at the edge of the graph area
        Self::render_label(buf, labels.last().unwrap(), label_area, Alignment::Right);
    }

    fn first_x_label_area(
        &self,
        y: u16,
        label_width: u16,
        max_width_after_y_axis: u16,
        chart_area: Rect,
        graph_area: Rect,
    ) -> Rect {
        let (min_x, max_x) = match self.x_axis.labels_alignment {
            Alignment::Left => (chart_area.left(), graph_area.left()),
            Alignment::Center => (
                chart_area.left(),
                graph_area.left() + max_width_after_y_axis.min(label_width),
            ),
            Alignment::Right => (
                graph_area.left().saturating_sub(1),
                graph_area.left() + max_width_after_y_axis,
            ),
        };

        Rect::new(min_x, y, max_x - min_x, 1)
    }

    fn render_label(buf: &mut Buffer, label: &Span, label_area: Rect, alignment: Alignment) {
        let label_width = label.width() as u16;
        let bounded_label_width = label_area.width.min(label_width);

        let x = match alignment {
            Alignment::Left => label_area.left(),
            Alignment::Center => label_area.left() + label_area.width / 2 - bounded_label_width / 2,
            Alignment::Right => label_area.right() - bounded_label_width,
        };

        buf.set_span(x, label_area.top(), label, bounded_label_width);
    }

    fn render_y_labels(
        &self,
        buf: &mut Buffer,
        layout: &ChartLayout,
        chart_area: Rect,
        graph_area: Rect,
    ) {
        let Some(x) = layout.label_y else { return };
        let labels = self.y_axis.labels.as_ref().unwrap();
        let labels_len = labels.len() as u16;
        for (i, label) in labels.iter().enumerate() {
            let dy = i as u16 * (graph_area.height - 1) / (labels_len - 1);
            if dy < graph_area.bottom() {
                let label_area = Rect::new(
                    x,
                    graph_area.bottom().saturating_sub(1) - dy,
                    (graph_area.left() - chart_area.left()).saturating_sub(1),
                    1,
                );
                Self::render_label(buf, label, label_area, self.y_axis.labels_alignment);
            }
        }
    }
}

impl Widget for HeatmapScatterPlot<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        self.render_ref(area, buf);
    }
}

impl WidgetRef for HeatmapScatterPlot<'_> {
    #[allow(clippy::too_many_lines)]
    fn render_ref(&self, area: Rect, buf: &mut Buffer) {
        buf.set_style(area, self.style);

        self.block.render_ref(area, buf);
        let chart_area = self.block.inner_if_some(area);
        let Some(layout) = self.layout(chart_area) else {
            return;
        };
        let graph_area = layout.graph_area;

        // Sample the style of the entire widget. This sample will be used to reset the style of
        // the cells that are part of the components put on top of the grah area (i.e legend and
        // axis names).
        let original_style = buf.get(area.left(), area.top()).style();

        self.render_x_labels(buf, &layout, chart_area, graph_area);
        self.render_y_labels(buf, &layout, chart_area, graph_area);

        if let Some(y) = layout.axis_x {
            for x in graph_area.left()..graph_area.right() {
                buf.get_mut(x, y)
                    .set_symbol(symbols::line::HORIZONTAL)
                    .set_style(self.x_axis.style);
            }
        }

        if let Some(x) = layout.axis_y {
            for y in graph_area.top()..graph_area.bottom() {
                buf.get_mut(x, y)
                    .set_symbol(symbols::line::VERTICAL)
                    .set_style(self.y_axis.style);
            }
        }

        if let Some(y) = layout.axis_x {
            if let Some(x) = layout.axis_y {
                buf.get_mut(x, y)
                    .set_symbol(symbols::line::BOTTOM_LEFT)
                    .set_style(self.x_axis.style);
            }
        }

        Canvas::default()
            .background_color(self.style.bg.unwrap_or(Color::Reset))
            .x_bounds(self.x_axis.bounds)
            .y_bounds(self.y_axis.bounds)
            .marker(self.dataset.marker)
            .paint(|ctx| {
                ctx.draw(&Points {
                    coords: self.dataset.data,
                    gradient: self.dataset.gradient,
                });
            })
            .render(graph_area, buf);

        if let Some((x, y)) = layout.title_x {
            let title = self.x_axis.title.as_ref().unwrap();
            let width = graph_area
                .right()
                .saturating_sub(x)
                .min(title.width() as u16);
            buf.set_style(
                Rect {
                    x,
                    y,
                    width,
                    height: 1,
                },
                original_style,
            );
            buf.set_line(x, y, title, width);
        }

        if let Some((x, y)) = layout.title_y {
            let title = self.y_axis.title.as_ref().unwrap();
            let width = graph_area
                .right()
                .saturating_sub(x)
                .min(title.width() as u16);
            buf.set_style(
                Rect {
                    x,
                    y,
                    width,
                    height: 1,
                },
                original_style,
            );
            buf.set_line(x, y, title, width);
        }

        // if let Some(legend_area) = layout.legend_area {
        //     buf.set_style(legend_area, original_style);
        //     Block::bordered().render(legend_area, buf);

        //     for (i, (dataset_name, dataset_style)) in self
        //         .datasets
        //         .iter()
        //         .filter_map(|ds| Some((ds.name.as_ref()?, ds.style())))
        //         .enumerate()
        //     {
        //         let name = dataset_name.clone().patch_style(dataset_style);
        //         name.render(
        //             Rect {
        //                 x: legend_area.x + 1,
        //                 y: legend_area.y + 1 + i as u16,
        //                 width: legend_area.width - 2,
        //                 height: 1,
        //             },
        //             buf,
        //         );
        //     }
        // }
    }
}

impl<'a> Styled for Axis<'a> {
    type Item = Self;

    fn style(&self) -> Style {
        self.style
    }

    fn set_style<S: Into<Style>>(self, style: S) -> Self::Item {
        self.style(style)
    }
}

impl<'a> Styled for Dataset<'a> {
    type Item = Self;

    fn style(&self) -> Style {
        self.style
    }

    fn set_style<S: Into<Style>>(self, style: S) -> Self::Item {
        self.style(style)
    }
}

impl<'a> Styled for HeatmapScatterPlot<'a> {
    type Item = Self;

    fn style(&self) -> Style {
        self.style
    }

    fn set_style<S: Into<Style>>(self, style: S) -> Self::Item {
        self.style(style)
    }
}
