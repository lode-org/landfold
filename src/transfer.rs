//! Distance transfer functions for sigmoid MDS.
//!
//! The generalised sigmoid used on high-D and low-D distances is
//!
//! `F(x; σ, a, b) = 1 - (1 + (2^{a/b} - 1) (x/σ)^a )^{-b/a}`
//!
//! which equals 1/2 at `x = σ` (Ceriotti, Tribello, Parrinello,
//! *Proc. Natl. Acad. Sci. U.S.A.* **108**, 13023 (2011),
//! <https://doi.org/10.1073/pnas.1108486108>).

use crate::error::{LandfoldError, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferMode {
    Identity,
    Sigmoid,
    Compress,
    XSigmoid,
    Gamma,
    Warp,
}

/// High-D or low-D distance transfer function with analytic derivative.
#[derive(Clone, Debug)]
pub struct Transfer {
    mode: TransferMode,
    /// Packed coefficients for the closed-form `f` / `df`.
    pars: Vec<f64>,
}

impl Default for Transfer {
    fn default() -> Self {
        Self::identity()
    }
}

impl Transfer {
    fn from_parts(mode: TransferMode, pars: Vec<f64>) -> Result<Self> {
        if pars.iter().any(|&value| !value.is_finite()) {
            return Err(LandfoldError::TransferParams(
                "transfer coefficients must be finite",
            ));
        }
        Ok(Self { mode, pars })
    }

    pub fn identity() -> Self {
        Self {
            mode: TransferMode::Identity,
            pars: Vec::new(),
        }
    }

    /// `1 - 1/(1 + (x/sigma)^2)`.
    pub fn sigmoid(sigma: f64) -> Result<Self> {
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(LandfoldError::TransferParams("sigmoid sigma must be > 0"));
        }
        let inv = 1.0 / sigma;
        Self::from_parts(TransferMode::Sigmoid, vec![inv, 2.0 * inv * inv])
    }

    /// `1 - 1/(1 + x/sigma)`.
    pub fn compress(sigma: f64) -> Result<Self> {
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(LandfoldError::TransferParams("compress sigma must be > 0"));
        }
        Self::from_parts(TransferMode::Compress, vec![1.0 / sigma])
    }

    /// Generalised sigmoid of Ceriotti 2011. Arguments `(sigma, a, b)`.
    pub fn xsigmoid(sigma: f64, a: f64, b: f64) -> Result<Self> {
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(LandfoldError::TransferParams("xsigmoid sigma must be > 0"));
        }
        if !(a.is_finite() && a > 0.0 && b.is_finite() && b > 0.0) {
            return Err(LandfoldError::TransferParams(
                "xsigmoid a and b must be finite and > 0",
            ));
        }
        Self::from_parts(
            TransferMode::XSigmoid,
            vec![1.0 / sigma, 2.0_f64.powf(a / b) - 1.0, a, b, -b / a],
        )
    }

    /// Regularised incomplete-gamma sigmoid. Arguments `(sigma, n)`.
    ///
    /// `P(n/2, (x/(sigma sqrt(2)))^2)` via a series / continued-fraction Q.
    pub fn gamma(sigma: f64, n: f64) -> Result<Self> {
        if !(sigma.is_finite() && sigma > 0.0) || !(n.is_finite() && n > 0.0) {
            return Err(LandfoldError::TransferParams(
                "gamma needs sigma > 0 and n > 0",
            ));
        }
        let normalizer = 2.0 / gamma_half(n * 0.5);
        if !normalizer.is_finite() || normalizer <= 0.0 {
            return Err(LandfoldError::TransferParams(
                "gamma shape has no finite positive normalization",
            ));
        }
        Self::from_parts(
            TransferMode::Gamma,
            vec![
                1.0 / (sigma * std::f64::consts::SQRT_2),
                n,
                normalizer,
            ],
        )
    }

    /// `F_LD^{-1}(F_HD(x))` warp. Arguments `(sigma, a_D, b_D, a_d, b_d)`.
    pub fn warp(sigma: f64, a_d: f64, b_d: f64, a_ld: f64, b_ld: f64) -> Result<Self> {
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(LandfoldError::TransferParams("warp sigma must be > 0"));
        }
        if ![a_d, b_d, a_ld, b_ld]
            .into_iter()
            .all(|p| p.is_finite() && p > 0.0)
        {
            return Err(LandfoldError::TransferParams(
                "warp shape parameters must be finite and > 0",
            ));
        }
        Self::from_parts(
            TransferMode::Warp,
            vec![
                1.0 / sigma,
                2.0_f64.powf(a_d / b_d) - 1.0,
                a_d,
                b_d,
                -b_d / a_d,
                sigma,
                2.0_f64.powf(a_ld / b_ld) - 1.0,
                1.0 / a_ld,
                b_ld,
                -a_ld / b_ld,
            ],
        )
    }

    /// Parse `identity`, `sigma`, `sigma,n`, `sigma,a,b`, or `sigma,aD,bD,ad,bd`.
    pub fn from_cli(spec: &str) -> Result<Self> {
        let spec = spec.trim();
        if spec.is_empty() || spec.eq_ignore_ascii_case("identity") {
            return Ok(Self::identity());
        }
        let parts: Vec<f64> = spec
            .split(',')
            .map(|s| {
                s.trim()
                    .parse::<f64>()
                    .map_err(|e| LandfoldError::Parse(format!("fun spec `{spec}`: {e}")))
            })
            .collect::<Result<Vec<_>>>()?;
        match parts.as_slice() {
            [s] => Self::sigmoid(*s),
            [s, n] => Self::gamma(*s, *n),
            [s, a, b] => Self::xsigmoid(*s, *a, *b),
            [s, a, b, al, bl] => Self::warp(*s, *a, *b, *al, *bl),
            _ => Err(LandfoldError::TransferParams(
                "fun spec must be identity | sigma | sigma,n | sigma,a,b | sigma,aD,bD,ad,bd",
            )),
        }
    }

    pub fn mode(&self) -> TransferMode {
        self.mode
    }

    pub fn f(&self, x: f64) -> f64 {
        self.fdf(x).0
    }

    pub fn df(&self, x: f64) -> f64 {
        self.fdf(x).1
    }

    /// Value and analytic derivative.
    pub fn fdf(&self, x: f64) -> (f64, f64) {
        match self.mode {
            TransferMode::Identity => (x, 1.0),
            TransferMode::Compress => {
                let sx = x * self.pars[0];
                let inv = 1.0 / (1.0 + sx);
                (1.0 - inv, inv * inv * self.pars[0])
            }
            TransferMode::Sigmoid => {
                let sx = x * self.pars[0];
                let inv = 1.0 / (1.0 + sx * sx);
                (1.0 - inv, x * (inv * inv) * self.pars[1])
            }
            TransferMode::XSigmoid => xsigmoid_fdf(&self.pars, x),
            TransferMode::Gamma => gamma_fdf(&self.pars, x),
            TransferMode::Warp => {
                let (fx, dfx) = xsigmoid_fdf(&self.pars, x);
                let gx = warp_g(&self.pars, fx);
                let dg = warp_dg(&self.pars, fx);
                (gx, dg * dfx)
            }
        }
    }

    /// Checked value and derivative on the nonnegative distance domain.
    pub fn try_fdf(&self, x: f64) -> Result<(f64, f64)> {
        if !x.is_finite() || x < 0.0 {
            return Err(LandfoldError::Msg(
                "transfer distance must be finite and nonnegative".into(),
            ));
        }
        let (value, derivative) = self.fdf(x);
        if !value.is_finite() || value < 0.0 || !derivative.is_finite() || derivative < 0.0 {
            return Err(LandfoldError::Msg(
                "transfer evaluation is non-finite or non-monotone".into(),
            ));
        }
        Ok((value, derivative))
    }
}

fn pow_exp(base: f64, exp: f64) -> f64 {
    let n = exp.round();
    if (exp - n).abs() <= 1e-12 && n.abs() <= 32.0 {
        if n >= 0.0 {
            base.powi(n as i32)
        } else if base != 0.0 {
            1.0 / base.powi((-n) as i32)
        } else {
            f64::INFINITY
        }
    } else {
        base.powf(exp)
    }
}

fn xsigmoid_fdf(pars: &[f64], x: f64) -> (f64, f64) {
    if x == 0.0 {
        return (0.0, 0.0);
    }
    let sx = x * pars[0];
    let sx = pars[1] * pow_exp(sx, pars[2]);
    let rf = pow_exp(1.0 + sx, pars[4]);
    let rdf = pars[3] * sx / x * rf / (1.0 + sx);
    (1.0 - rf, rdf)
}

fn warp_g(pars: &[f64], y: f64) -> f64 {
    let sx = (1.0 - y).powf(pars[9]);
    let sx = (sx - 1.0) / pars[6];
    pars[5] * sx.powf(pars[7])
}

fn warp_dg(pars: &[f64], y: f64) -> f64 {
    let sx = (1.0 - y).powf(-pars[9]);
    let sx = (sx - 1.0) * (y - 1.0) * pars[8];
    warp_g(pars, y) / sx
}

fn gamma_fdf(pars: &[f64], x: f64) -> (f64, f64) {
    let sx = x * pars[0];
    let rf = 1.0 - regularised_gamma_q(pars[1] * 0.5, sx * sx);
    let rdf = if x == 0.0 {
        0.0
    } else {
        pars[2] * sx.powf(pars[1] - 1.0) * (-sx * sx).exp() * pars[0]
    };
    (rf, rdf)
}

/// Lanczos approximation for Gamma(z) on z > 0.
fn gamma_half(z: f64) -> f64 {
    // Lanczos g=7, n=9 (Gautschi).
    const P: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if z < 0.5 {
        return std::f64::consts::PI / ((std::f64::consts::PI * z).sin() * gamma_half(1.0 - z));
    }
    let z = z - 1.0;
    let mut x = P[0];
    for (i, &p) in P.iter().enumerate().skip(1) {
        x += p / (z + i as f64);
    }
    let t = z + 7.5;
    (2.0 * std::f64::consts::PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * x
}

/// Regularised upper incomplete gamma Q(s,x) = Gamma(s,x)/Gamma(s).
fn regularised_gamma_q(s: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if x < s + 1.0 {
        1.0 - regularised_gamma_p_series(s, x)
    } else {
        regularised_gamma_q_cf(s, x)
    }
}

fn regularised_gamma_p_series(s: f64, x: f64) -> f64 {
    let mut sum = 1.0 / s;
    let mut term = sum;
    for n in 1..200 {
        term *= x / (s + n as f64);
        sum += term;
        if term.abs() < sum.abs() * 1e-15 {
            break;
        }
    }
    (-x + s * x.ln() - gamma_half(s).ln()).exp() * sum
}

fn regularised_gamma_q_cf(s: f64, x: f64) -> f64 {
    // Modified Lentz.
    const FPMIN: f64 = 1e-300;
    let mut b = x + 1.0 - s;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..200 {
        let an = -i as f64 * (i as f64 - s);
        b += 2.0;
        d = an * d + b;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = b + an / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-14 {
            break;
        }
    }
    (-x + s * x.ln() - gamma_half(s).ln()).exp() * h
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn xsigmoid_half_at_sigma() {
        let t = Transfer::xsigmoid(6.0, 8.0, 8.0).unwrap();
        assert_relative_eq!(t.f(6.0), 0.5, epsilon = 1e-14);
        assert_relative_eq!(t.f(0.0), 0.0, epsilon = 1e-14);
        assert!(t.f(100.0) > 0.999);
    }

    #[test]
    fn sigmoid_matches_closed_form() {
        let t = Transfer::sigmoid(1.0).unwrap();
        assert_relative_eq!(t.f(1.0), 0.5, epsilon = 1e-15);
        assert_relative_eq!(t.df(1.0), 0.5, epsilon = 1e-15);
        let (f, df) = t.fdf(2.0);
        assert_relative_eq!(f, 1.0 - 1.0 / 5.0, epsilon = 1e-15);
        assert_relative_eq!(df, 2.0 * (0.2_f64.powi(2)) * 2.0, epsilon = 1e-14);
    }

    #[test]
    fn compress_matches_closed_form() {
        let t = Transfer::compress(1.0).unwrap();
        assert_relative_eq!(t.f(1.0), 0.5, epsilon = 1e-15);
        assert_relative_eq!(t.df(1.0), 0.25, epsilon = 1e-15);
    }

    #[test]
    fn identity_is_passthrough() {
        let t = Transfer::identity();
        assert_eq!(t.f(3.5), 3.5);
        assert_eq!(t.df(3.5), 1.0);
    }

    #[test]
    fn checked_evaluation_rejects_invalid_domain_and_overflow() {
        let t = Transfer::xsigmoid(1.0, 8.0, 1.0).unwrap();
        assert!(t.try_fdf(-1.0).is_err());
        assert!(t.try_fdf(f64::NAN).is_err());
        assert!(t.try_fdf(1.0e200).is_err());
        assert!(Transfer::identity().try_fdf(1.0).is_ok());
    }

    #[test]
    fn finite_difference_xsigmoid() {
        let t = Transfer::xsigmoid(1.0, 4.0, 3.0).unwrap();
        let x = 1.3;
        let h = 1e-7;
        let fd = (t.f(x + h) - t.f(x - h)) / (2.0 * h);
        assert_relative_eq!(t.df(x), fd, epsilon = 1e-7);
    }

    #[test]
    fn from_cli_protein_example() {
        let hd = Transfer::from_cli("6,8,8").unwrap();
        let ld = Transfer::from_cli("6,2,8").unwrap();
        assert_eq!(hd.mode(), TransferMode::XSigmoid);
        assert_eq!(ld.mode(), TransferMode::XSigmoid);
        assert_relative_eq!(hd.f(6.0), 0.5, epsilon = 1e-14);
        assert_relative_eq!(ld.f(6.0), 0.5, epsilon = 1e-14);
    }

    #[test]
    fn gamma_is_an_increasing_distance_transfer() {
        let t = Transfer::gamma(2.0, 2.0).unwrap();
        assert_relative_eq!(t.f(0.0), 0.0, epsilon = 1e-15);
        assert!(t.f(1.0) > 0.0);
        assert!(t.f(2.0) > t.f(1.0));
        assert!(t.df(1.0) > 0.0);
    }

    #[test]
    fn rejects_gamma_shapes_without_finite_normalization() {
        assert!(Transfer::gamma(1.0, 1_000.0).is_err());
    }

    #[test]
    fn rejects_invalid_xsigmoid_parameters() {
        assert!(Transfer::xsigmoid(1.0, 2.0, 0.0).is_err());
        assert!(Transfer::xsigmoid(f64::NAN, 2.0, 1.0).is_err());
    }

    #[test]
    fn rejects_invalid_warp_parameters() {
        assert!(Transfer::warp(1.0, 0.0, 1.0, 2.0, 1.0).is_err());
        assert!(Transfer::warp(1.0, 2.0, 1.0, 2.0, 0.0).is_err());
        assert!(Transfer::warp(1.0, 2.0, f64::NAN, 2.0, 1.0).is_err());
    }

    #[test]
    fn rejects_overflowed_derived_coefficients() {
        assert!(Transfer::xsigmoid(1.0, 10_000.0, 1.0).is_err());
        assert!(Transfer::warp(1.0, 10_000.0, 1.0, 1.0, 1.0).is_err());
    }
}
