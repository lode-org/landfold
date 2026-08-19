//! Polak-Ribiere conjugate gradient with Brent line search.
//!
//! Direction: `γ = (g − g_old)·g / (g_old·g_old)` (Polak and Ribiere,
//! *Rev. Fr. Inform. Rech. Opér.* **16**, 35 (1969)). Line search is
//! Brent's method (*Algorithms for Minimization without Derivatives*,
//! 1973). This is the standard full-pair solver.

use ndarray::{Array1, ArrayView1};

use crate::error::Result;
use crate::stress::Stress;

#[derive(Clone, Debug)]
pub struct CgOpts {
    pub maxiter: usize,
    pub tol: f64,
    pub ls_maxiter: usize,
    pub ls_tol: f64,
    pub istep: f64,
}

impl Default for CgOpts {
    fn default() -> Self {
        Self {
            maxiter: 100,
            tol: 1e-5,
            ls_maxiter: 5,
            ls_tol: 5e-10,
            istep: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CgReport {
    pub value: f64,
    pub coords: Array1<f64>,
    pub steps: usize,
}

pub fn minimize(stress: &Stress, init: ArrayView1<f64>, d: usize, opts: &CgOpts) -> Result<CgReport> {
    let mut pos = init.to_owned();
    let mut ev = stress.eval(pos.view(), d);
    let mut dir = ev.grad.clone();
    let mut og = dir.clone();
    let mut value = ev.value;
    let mut istep = opts.istep;

    for step in 0..opts.maxiter {
        let gnorm: f64 = ev.grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if gnorm < opts.tol {
            return Ok(CgReport {
                value,
                coords: pos,
                steps: step,
            });
        }
        let (npos, nval, lsstep) = line_search(stress, pos.view(), dir.view(), d, istep, opts);
        if nval < value {
            pos = npos;
            value = nval;
        }
        ev = stress.eval(pos.view(), d);
        let mut gg = 0.0;
        let mut gamma = 0.0;
        for i in 0..og.len() {
            gg += og[i] * og[i];
            gamma += (ev.grad[i] - og[i]) * ev.grad[i];
        }
        if gg > 0.0 {
            gamma /= gg;
        } else {
            gamma = 0.0;
        }
        for i in 0..dir.len() {
            dir[i] = gamma * dir[i] + ev.grad[i];
        }
        og.assign(&ev.grad);
        istep = if lsstep <= 0.0 {
            opts.istep
        } else {
            lsstep * 0.5
        };
    }
    Ok(CgReport {
        value,
        coords: pos,
        steps: opts.maxiter,
    })
}

fn line_search(
    stress: &Stress,
    pos: ArrayView1<f64>,
    dir: ArrayView1<f64>,
    d: usize,
    istep: f64,
    opts: &CgOpts,
) -> (Array1<f64>, f64, f64) {
    let f0 = stress.eval(pos, d).value;
    let phi = |t: f64| {
        let x = &pos + &(&dir * t);
        stress.eval(x.view(), d).value
    };
    let (a, b) = bracket(phi, 0.0, istep.max(1e-12), opts.ls_maxiter);
    let (t, ft) = brent(phi, a, b, opts.ls_tol, opts.ls_maxiter.max(20));
    if ft < f0 {
        (pos.to_owned() + &(dir.to_owned() * t), ft, t.abs())
    } else {
        (pos.to_owned(), f0, 0.0)
    }
}

fn bracket(mut phi: impl FnMut(f64) -> f64, mut a: f64, mut b: f64, maxiter: usize) -> (f64, f64) {
    let gold = 1.618_034;
    let mut fa = phi(a);
    let mut fb = phi(b);
    if fa < fb {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }
    let mut c = b + gold * (b - a);
    let mut fc = phi(c);
    let mut it = 0;
    while fb >= fc && it < maxiter {
        let u = b + gold * (c - b);
        let fu = phi(u);
        a = b;
        fa = fb;
        b = c;
        fb = fc;
        c = u;
        fc = fu;
        it += 1;
        let _ = fa;
    }
    if a < c {
        (a, c)
    } else {
        (c, a)
    }
}

fn brent(
    mut phi: impl FnMut(f64) -> f64,
    ax: f64,
    cx: f64,
    tol: f64,
    maxiter: usize,
) -> (f64, f64) {
    let cgold = 0.381_966;
    let mut a = ax.min(cx);
    let mut b = ax.max(cx);
    let mut x = 0.5 * (a + b);
    let mut w = x;
    let mut v = x;
    let mut fx = phi(x);
    let mut fw = fx;
    let mut fv = fx;
    let mut e: f64 = 0.0;
    let mut d: f64 = 0.0;
    for _ in 0..maxiter {
        let xm = 0.5 * (a + b);
        let tol1 = tol * x.abs() + 1e-12;
        let tol2 = 2.0 * tol1;
        if (x - xm).abs() <= tol2 - 0.5 * (b - a) {
            return (x, fx);
        }
        let mut u;
        if e.abs() > tol1 {
            let r = (x - w) * (fx - fv);
            let mut q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            let etemp = e;
            e = d;
            if p.abs() >= (0.5 * q * etemp).abs() || p <= q * (a - x) || p >= q * (b - x) {
                e = if x >= xm { a - x } else { b - x };
                d = cgold * e;
            } else {
                d = p / q;
                u = x + d;
                if u - a < tol2 || b - u < tol2 {
                    d = if xm - x >= 0.0 { tol1 } else { -tol1 };
                }
            }
        } else {
            e = if x >= xm { a - x } else { b - x };
            d = cgold * e;
        }
        u = if d.abs() >= tol1 {
            x + d
        } else {
            x + if d >= 0.0 { tol1 } else { -tol1 }
        };
        let fu = phi(u);
        if fu <= fx {
            if u >= x {
                a = x;
            } else {
                b = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }
    (x, fx)
}
