/// An implementation of a time-decaying value
pub trait Decay {
    /// Calculate value at time `t`
    fn evaluate(&self, t: f32) -> f32;
}

// TODO: better error types
fn validate(rate: f32, vi: f32, vf: f32) -> Result<(), String> {
    ((rate >= 0.0 && vi > vf) || (rate < 0.0 && vi < vf))
        .then_some(())
        .ok_or_else(|| String::from("`vi - vf` must have same sign as `rate`"))
}

/// A constant value
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Constant {
    value: f32,
}

impl Constant {
    pub fn new(value: f32) -> Self {
        Self { value }
    }
}

impl Decay for Constant {
    fn evaluate(&self, _t: f32) -> f32 {
        self.value
    }
}

/// v(t) = v<sub>f</sub> + (v<sub>i</sub> - v<sub>f</sub>) * e<sup>-rt</sup>
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Exponential {
    rate: f32,
    vi: f32,
    vf: f32,
}

impl Exponential {
    pub fn new(rate: f32, vi: f32, vf: f32) -> Result<Self, String> {
        validate(rate, vi, vf)?;
        Ok(Self { rate, vi, vf })
    }
}

impl Decay for Exponential {
    fn evaluate(&self, t: f32) -> f32 {
        let &Self { rate, vi, vf } = self;
        vf + (vi - vf) * (-rate * t).exp()
    }
}

/// v(t) = v<sub>f</sub> + (v<sub>i</sub> - v<sub>f</sub>) / (1 + rt)
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InverseTime {
    rate: f32,
    vi: f32,
    vf: f32,
}

impl InverseTime {
    pub fn new(rate: f32, vi: f32, vf: f32) -> Result<Self, String> {
        validate(rate, vi, vf)?;
        Ok(Self { rate, vi, vf })
    }
}

impl Decay for InverseTime {
    fn evaluate(&self, t: f32) -> f32 {
        let &Self { rate, vi, vf } = self;
        vf + (vi - vf) / (1.0 + rate * t)
    }
}

/// v(t) = max(v<sub>i</sub> - rt, v<sub>f</sub>)
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Linear {
    rate: f32,
    vi: f32,
    vf: f32,
}

impl Linear {
    pub fn new(rate: f32, vi: f32, vf: f32) -> Result<Self, String> {
        validate(rate, vi, vf)?;
        Ok(Self { rate, vi, vf })
    }
}

impl Decay for Linear {
    fn evaluate(&self, t: f32) -> f32 {
        let &Self { rate, vi, vf } = self;
        (vi - rate * t).max(vf)
    }
}

/// v(t) = max(v<sub>i</sub> * r<sup>floor(t/s)</sup>, v<sub>f</sub>)
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Step {
    rate: f32,
    vi: f32,
    vf: f32,
    step: f32,
}

impl Step {
    pub fn new(rate: f32, vi: f32, vf: f32, step: f32) -> Result<Self, String> {
        validate(rate, vi, vf)?;
        Ok(Self { rate, vi, vf, step })
    }
}

impl Decay for Step {
    fn evaluate(&self, t: f32) -> f32 {
        let &Self { rate, vi, vf, step } = self;
        (vi * rate.powf((t / step).floor())).max(vf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_functional() {
        assert!(validate(1.0, 1.0, 0.0).is_ok());
        assert!(validate(1.0, -1.0, 0.0).is_err());
        assert!(validate(-1.0, 1.0, 0.0).is_err());
        assert!(validate(-1.0, -1.0, 0.0).is_ok());
    }

    #[test]
    fn constant_decay() {
        let x = Constant::new(1.0);
        assert_eq!(x.evaluate(0.0), 1.0);
        assert_eq!(x.evaluate(1.0), 1.0);
    }

    #[test]
    fn exponential_decay() {
        let x = Exponential::new(2.0, 2.0, 0.5).unwrap();
        assert_eq!(x.evaluate(0.0), 2.0);
        assert_eq!(x.evaluate(1.0), 0.5 + 1.5 * f32::exp(-2.0));
    }

    #[test]
    fn inverse_time_decay() {
        let x = InverseTime::new(2.0, 2.0, 0.5).unwrap();
        assert_eq!(x.evaluate(0.0), 2.0);
        assert_eq!(x.evaluate(1.0), 1.0);
    }

    #[test]
    fn linear_decay() {
        let x = Linear::new(0.5, 2.0, 0.5).unwrap();
        assert_eq!(x.evaluate(0.0), 2.0);
        assert_eq!(x.evaluate(1.0), 1.5);
        assert_eq!(x.evaluate(10.0), 0.5);
    }

    #[test]
    fn step_decay() {
        let x = Step::new(0.5, 2.0, 0.0, 0.5).unwrap();
        assert_eq!(x.evaluate(0.25), 2.0);
        assert_eq!(x.evaluate(0.75), 1.0);
        assert_eq!(x.evaluate(1.0), 0.5);
    }
}
