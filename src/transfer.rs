//! Distance transfer functions for sigmoid MDS.
//!
//! The generalised sigmoid used on high-D and low-D distances is
//!
//! `F(x; σ, a, b) = 1 - (1 + (2^{a/b} - 1) (x/σ)^a )^{-b/a}`
//!
//! which equals 1/2 at `x = σ` (Ceriotti, Tribello, Parrinello,
//! *Proc. Natl. Acad. Sci. U.S.A.* **108**, 13023 (2011),
//! <https://doi.org/10.1073/pnas.1108486108>).
//!
//! `imq` is the MethodsX inverse-multiquadric generator
//! `k = (c^2 + r^2)^{-1/2}` rewritten as an increasing distance
//! transfer `F(x) = 1 - c / sqrt(c^2 + x^2)` with `c = σ/√3` so
//! `F(σ) = 1/2`. Completely monotone in `r^2` (Schoenberg 1938),
//! hence strictly positive definite on every `R^d`
//! (Buhmann, *Radial Basis Functions*, 2003).

use crate::error::{LandfoldError, Result};

/// CLI / Python help for `--fun-hd` / `--fun-ld`.
///
/// Ceriotti generalised sigmoid (`sigma,a,b` or `ceriotti,sigma,a,b`)
/// is the PNAS 2011 / JCTC 2013 path. `identity`, `sigma`, `sigma,n`,
/// and warp match the C++ `dimred` parser. `imq` / `ms` stay available
/// but are not the reproduction default.
pub const FUN_SPEC_HELP: &str = "\
Ceriotti sigmoid (PNAS/JCTC): sigma,a,b or ceriotti,sigma,a,b. \
Also identity | sigma | sigma,n | sigma,aD,bD,ad,bd (C++ dimred). \
imq,sigma and ms,s1,s2,... are extra.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferMode {
    Identity,
    Sigmoid,
    Compress,
    XSigmoid,
    Gamma,
    Warp,
    Imq,
    Multiscale,
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
        if !(a.is_finite() && a >= 1.0 && b.is_finite() && b > 0.0) {
            return Err(LandfoldError::TransferParams(
                "xsigmoid a must be finite and >= 1; b must be finite and > 0",
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
        if !(sigma.is_finite() && sigma > 0.0) || !(n.is_finite() && n >= 1.0) {
            return Err(LandfoldError::TransferParams(
                "gamma needs sigma > 0 and n >= 1",
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

    /// Inverse-multiquadric transfer. Argument `sigma`.
    ///
    /// `F(x) = 1 - c / sqrt(c^2 + x^2)` with `c = σ/√3`, so `F(σ) = 1/2`.
    /// The generator `(c^2 + r^2)^{-1/2}` is the MethodsX IMQ kernel.
    pub fn imq(sigma: f64) -> Result<Self> {
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(LandfoldError::TransferParams("imq sigma must be > 0"));
        }
        let c = sigma / 3.0_f64.sqrt();
        if !c.is_finite() || c <= 0.0 {
            return Err(LandfoldError::TransferParams("imq scale is not finite"));
        }
        Self::from_parts(TransferMode::Imq, vec![c])
    }

    /// Mean of IMQ transfers at several scales (PNAS 2011 hierarchical map).
    pub fn multiscale(sigmas: &[f64]) -> Result<Self> {
        if sigmas.len() < 2 {
            return Err(LandfoldError::TransferParams(
                "ms needs at least two positive sigmas",
            ));
        }
        let mut cs = Vec::with_capacity(sigmas.len());
        for &sigma in sigmas {
            if !(sigma.is_finite() && sigma > 0.0) {
                return Err(LandfoldError::TransferParams(
                    "ms sigmas must be finite and > 0",
                ));
            }
            let c = sigma / 3.0_f64.sqrt();
            if !c.is_finite() || c <= 0.0 {
                return Err(LandfoldError::TransferParams("ms scale is not finite"));
            }
            cs.push(c);
        }
        Self::from_parts(TransferMode::Multiscale, cs)
    }

    /// `F_LD^{-1}(F_HD(x))` warp. Arguments `(sigma, a_D, b_D, a_d, b_d)`.
    pub fn warp(sigma: f64, a_d: f64, b_d: f64, a_ld: f64, b_ld: f64) -> Result<Self> {
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(LandfoldError::TransferParams("warp sigma must be > 0"));
        }
        if !(a_d.is_finite()
            && a_d >= 1.0
            && b_d.is_finite()
            && b_d > 0.0
            && a_ld.is_finite()
            && a_ld >= 1.0
            && b_ld.is_finite()
            && b_ld > 0.0)
        {
            return Err(LandfoldError::TransferParams(
                "warp a exponents must be finite and >= 1; b exponents must be finite and > 0",
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

    /// Parse a named or numeric transfer spec.
    ///
    /// Ceriotti reproduction: `sigma,a,b` or `ceriotti,sigma,a,b`.
    /// Extra MethodsX arm: `imq,sigma`.
    pub fn from_cli(spec: &str) -> Result<Self> {
        let spec = spec.trim();
        if spec.is_empty() || spec.eq_ignore_ascii_case("identity") {
            return Ok(Self::identity());
        }
        let lower = spec.to_ascii_lowercase();
        if lower == "imq" {
            return Self::imq(1.0);
        }
        if let Some(rest) = lower.strip_prefix("imq,") {
            let sigma: f64 = rest.trim().parse().map_err(|e| {
                LandfoldError::Parse(format!("fun spec `{spec}`: {e}"))
            })?;
            return Self::imq(sigma);
        }
        if lower == "ms" || lower == "multi" {
            return Err(LandfoldError::TransferParams(
                "ms needs at least two sigmas (example: ms,2.8,4.4,6.3)",
            ));
        }
        if let Some(rest) = lower
            .strip_prefix("ms,")
            .or_else(|| lower.strip_prefix("multi,"))
        {
            let sigmas: Result<Vec<f64>> = rest
                .split(',')
                .map(|s| {
                    s.trim()
                        .parse::<f64>()
                        .map_err(|e| LandfoldError::Parse(format!("fun spec `{spec}`: {e}")))
                })
                .collect();
            return Self::multiscale(&sigmas?);
        }
        if lower == "ceriotti" {
            return Err(LandfoldError::TransferParams(
                "ceriotti needs sigma,a,b (example: ceriotti,5,8,1)",
            ));
        }
        let numeric = if let Some(rest) = lower.strip_prefix("ceriotti,") {
            rest
        } else {
            spec
        };
        let parts: Vec<f64> = numeric
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
                "fun spec must be ceriotti,sigma,a,b | imq,sigma | identity | sigma | sigma,n | sigma,a,b | sigma,aD,bD,ad,bd",
            )),
        }
    }

    pub fn mode(&self) -> TransferMode {
        self.mode
    }

    /// Family name for docs and Python: `ceriotti`, `imq`, or the other arms.
    pub fn family(&self) -> &'static str {
        match self.mode {
            TransferMode::XSigmoid => "ceriotti",
            TransferMode::Imq => "imq",
            TransferMode::Identity => "identity",
            TransferMode::Sigmoid => "sigmoid",
            TransferMode::Compress => "compress",
            TransferMode::Gamma => "gamma",
            TransferMode::Warp => "warp",
            TransferMode::Multiscale => "multiscale",
        }
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
            TransferMode::Imq => {
                let c = self.pars[0];
                let inv = 1.0 / (c * c + x * x).sqrt();
                (1.0 - c * inv, c * x * inv * inv * inv)
            }
            TransferMode::Multiscale => {
                let n = self.pars.len() as f64;
                let mut value = 0.0;
                let mut deriv = 0.0;
                for &c in &self.pars {
                    let inv = 1.0 / (c * c + x * x).sqrt();
                    value += 1.0 - c * inv;
                    deriv += c * x * inv * inv * inv;
                }
                (value / n, deriv / n)
            }
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
        let derivative = if pars[2] == 1.0 {
            pars[3] * pars[1] * pars[0]
        } else {
            0.0
        };
        return (0.0, derivative);
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
        if pars[1] == 1.0 {
            pars[2] * pars[0]
        } else {
            0.0
        }
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
    fn xsigmoid_handles_the_finite_unit_exponent_slope_at_zero() {
        let t = Transfer::xsigmoid(2.0, 1.0, 2.0).unwrap();
        let expected = 2.0 * (2.0_f64.powf(0.5) - 1.0) / 2.0;
        assert_relative_eq!(t.df(0.0), expected);
        assert!(Transfer::xsigmoid(2.0, 0.5, 2.0).is_err());
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
    fn imq_half_at_sigma_and_zero_at_origin() {
        let t = Transfer::imq(6.0).unwrap();
        assert_eq!(t.mode(), TransferMode::Imq);
        assert_relative_eq!(t.f(0.0), 0.0, epsilon = 1e-15);
        assert_relative_eq!(t.f(6.0), 0.5, epsilon = 1e-14);
        assert!(t.f(100.0) > t.f(6.0));
        assert!(t.f(100.0) < 1.0);
        assert_relative_eq!(t.df(0.0), 0.0, epsilon = 1e-15);
        assert!(t.df(6.0) > 0.0);
    }

    #[test]
    fn imq_finite_difference() {
        let t = Transfer::imq(2.0).unwrap();
        let x = 1.3;
        let h = 1e-7;
        let fd = (t.f(x + h) - t.f(x - h)) / (2.0 * h);
        assert_relative_eq!(t.df(x), fd, epsilon = 1e-7);
    }

    #[test]
    fn from_cli_imq() {
        let t = Transfer::from_cli("imq,5").unwrap();
        assert_eq!(t.mode(), TransferMode::Imq);
        assert_eq!(t.family(), "imq");
        assert_relative_eq!(t.f(5.0), 0.5, epsilon = 1e-14);
        assert!(Transfer::from_cli("imq,0").is_err());
        assert!(Transfer::imq(f64::NAN).is_err());
    }

    #[test]
    fn from_cli_ceriotti_is_xsigmoid_alias() {
        let named = Transfer::from_cli("ceriotti,5,8,1").unwrap();
        let numeric = Transfer::from_cli("5,8,1").unwrap();
        assert_eq!(named.mode(), TransferMode::XSigmoid);
        assert_eq!(named.family(), "ceriotti");
        assert_eq!(numeric.family(), "ceriotti");
        assert_relative_eq!(named.f(5.0), 0.5, epsilon = 1e-14);
        assert_relative_eq!(named.f(1.0), numeric.f(1.0), epsilon = 1e-14);
        assert!(Transfer::from_cli("ceriotti").is_err());
    }

    #[test]
    fn multiscale_is_the_mean_of_imq_scales() {
        let t = Transfer::from_cli("ms,2,6").unwrap();
        assert_eq!(t.family(), "multiscale");
        let a = Transfer::imq(2.0).unwrap();
        let b = Transfer::imq(6.0).unwrap();
        let x = 3.0;
        assert_relative_eq!(t.f(x), 0.5 * (a.f(x) + b.f(x)), epsilon = 1e-14);
        assert_relative_eq!(t.df(x), 0.5 * (a.df(x) + b.df(x)), epsilon = 1e-14);
        assert!(t.f(6.0) > t.f(2.0));
        assert!(Transfer::from_cli("ms,1").is_err());
        assert!(Transfer::multiscale(&[1.0]).is_err());
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
    fn gamma_handles_the_finite_unit_shape_slope_at_zero() {
        let t = Transfer::gamma(2.0, 1.0).unwrap();
        assert_relative_eq!(t.df(0.0), (2.0 / std::f64::consts::PI).sqrt() / 2.0);
        assert!(Transfer::gamma(2.0, 0.5).is_err());
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
        assert!(Transfer::warp(1.0, 0.5, 1.0, 2.0, 1.0).is_err());
        assert!(Transfer::warp(1.0, 2.0, 1.0, 0.5, 1.0).is_err());
        assert!(Transfer::warp(1.0, 2.0, 1.0, 2.0, 0.0).is_err());
        assert!(Transfer::warp(1.0, 2.0, f64::NAN, 2.0, 1.0).is_err());
    }

    #[test]
    fn rejects_overflowed_derived_coefficients() {
        assert!(Transfer::xsigmoid(1.0, 10_000.0, 1.0).is_err());
        assert!(Transfer::warp(1.0, 10_000.0, 1.0, 1.0, 1.0).is_err());
    }
}
