//! Weighted histograms and free-energy surfaces.
//!
//! `F/kT = -ln(ρ/ρ_max)` is the usual Boltzmann inversion of a histogram
//! (Ferrenberg and Swendsen, *Phys. Rev. Lett.* **63**, 1195 (1989),
//! <https://doi.org/10.1103/PhysRevLett.63.1195>). Coordination numbers
//! of Cartesian frames are an optional extra panel.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::error::{Result, LandfoldError};

#[derive(Clone, Debug)]
pub struct Histogram1d {
    pub edges: Array1<f64>,
    pub counts: Array1<f64>,
    pub below: f64,
    pub above: f64,
    pub samples: f64,
}

impl Histogram1d {
    pub fn new(lo: f64, hi: f64, n: usize) -> Result<Self> {
        if n == 0 || hi <= lo {
            return Err(LandfoldError::Shape("histogram needs n>0 and hi>lo"));
        }
        let mut edges = Array1::zeros(n + 1);
        for i in 0..=n {
            edges[i] = lo + (hi - lo) * (i as f64) / n as f64;
        }
        Ok(Self {
            edges,
            counts: Array1::zeros(n),
            below: 0.0,
            above: 0.0,
            samples: 0.0,
        })
    }

    pub fn add(&mut self, x: f64, w: f64) {
        self.samples += w;
        if x < self.edges[0] {
            self.below += w;
            return;
        }
        let last = self.edges.len() - 1;
        if x >= self.edges[last] {
            self.above += w;
            return;
        }
        let n = self.counts.len();
        let lo = self.edges[0];
        let hi = self.edges[last];
        let t = (x - lo) / (hi - lo) * n as f64;
        let i = (t as usize).min(n - 1);
        self.counts[i] += w;
    }

    pub fn add_many(&mut self, xs: ArrayView1<f64>, w: Option<ArrayView1<f64>>) {
        for (i, &x) in xs.iter().enumerate() {
            let ww = w.map(|ww| ww[i]).unwrap_or(1.0);
            self.add(x, ww);
        }
    }
}

#[derive(Clone, Debug)]
pub struct Histogram2d {
    pub x_edges: Array1<f64>,
    pub y_edges: Array1<f64>,
    pub counts: Array2<f64>,
    pub samples: f64,
}

impl Histogram2d {
    pub fn new(xlo: f64, xhi: f64, nx: usize, ylo: f64, yhi: f64, ny: usize) -> Result<Self> {
        if nx == 0 || ny == 0 || xhi <= xlo || yhi <= ylo {
            return Err(LandfoldError::Shape("2d histogram bounds"));
        }
        let mut x_edges = Array1::zeros(nx + 1);
        let mut y_edges = Array1::zeros(ny + 1);
        for i in 0..=nx {
            x_edges[i] = xlo + (xhi - xlo) * (i as f64) / nx as f64;
        }
        for i in 0..=ny {
            y_edges[i] = ylo + (yhi - ylo) * (i as f64) / ny as f64;
        }
        Ok(Self {
            x_edges,
            y_edges,
            counts: Array2::zeros((ny, nx)),
            samples: 0.0,
        })
    }

    pub fn add(&mut self, x: f64, y: f64, w: f64) {
        let nx = self.counts.ncols();
        let ny = self.counts.nrows();
        let xlo = self.x_edges[0];
        let xhi = self.x_edges[nx];
        let ylo = self.y_edges[0];
        let yhi = self.y_edges[ny];
        if x < xlo || x >= xhi || y < ylo || y >= yhi {
            return;
        }
        let ix = (((x - xlo) / (xhi - xlo) * nx as f64) as usize).min(nx - 1);
        let iy = (((y - ylo) / (yhi - ylo) * ny as f64) as usize).min(ny - 1);
        self.counts[(iy, ix)] += w;
        self.samples += w;
    }

    pub fn add_points(&mut self, xy: ArrayView2<f64>, w: Option<ArrayView1<f64>>) {
        for i in 0..xy.nrows() {
            let ww = w.map(|ww| ww[i]).unwrap_or(1.0);
            self.add(xy[(i, 0)], xy[(i, 1)], ww);
        }
    }
}

#[derive(Clone, Debug)]
pub struct FreeEnergy {
    pub x_centers: Array1<f64>,
    pub y_centers: Array1<f64>,
    /// F / kT = -ln(rho / rho_max). Empty bins are `None`.
    pub f: Array2<Option<f64>>,
    pub rho: Array2<f64>,
}

impl FreeEnergy {
    /// Build FES from a 2-D histogram. `kT` scales the output (`F/eps` when
    /// the user passes `kT = 1` and thinks in units of epsilon).
    pub fn from_histogram(h: &Histogram2d, kt: f64) -> Self {
        let nx = h.counts.ncols();
        let ny = h.counts.nrows();
        let mut x_centers = Array1::zeros(nx);
        let mut y_centers = Array1::zeros(ny);
        for i in 0..nx {
            x_centers[i] = 0.5 * (h.x_edges[i] + h.x_edges[i + 1]);
        }
        for i in 0..ny {
            y_centers[i] = 0.5 * (h.y_edges[i] + h.y_edges[i + 1]);
        }
        let mut rho = h.counts.clone();
        let area = ((h.x_edges[nx] - h.x_edges[0]) / nx as f64)
            * ((h.y_edges[ny] - h.y_edges[0]) / ny as f64);
        if h.samples > 0.0 && area > 0.0 {
            rho.mapv_inplace(|c| c / (h.samples * area));
        }
        let mut rmax = 0.0;
        for v in rho.iter() {
            if *v > rmax {
                rmax = *v;
            }
        }
        let mut f = Array2::from_elem((ny, nx), None);
        if rmax > 0.0 {
            for iy in 0..ny {
                for ix in 0..nx {
                    let r = rho[(iy, ix)];
                    if r > 0.0 {
                        f[(iy, ix)] = Some(-kt * (r / rmax).ln());
                    }
                }
            }
        }
        Self {
            x_centers,
            y_centers,
            f,
            rho,
        }
    }

    pub fn write_csv(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        writeln!(w, "# x y F rho")?;
        let ny = self.f.nrows();
        let nx = self.f.ncols();
        for iy in 0..ny {
            for ix in 0..nx {
                match self.f[(iy, ix)] {
                    Some(fv) => writeln!(
                        w,
                        "{:.8} {:.8} {:.8} {:.8e}",
                        self.x_centers[ix],
                        self.y_centers[iy],
                        fv,
                        self.rho[(iy, ix)]
                    )?,
                    None => writeln!(
                        w,
                        "{:.8} {:.8} nan {:.8e}",
                        self.x_centers[ix],
                        self.y_centers[iy],
                        self.rho[(iy, ix)]
                    )?,
                }
            }
            writeln!(w)?;
        }
        Ok(())
    }

    /// Minimal SVG heatmap (no isoline tracer). Darker = lower F.
    pub fn write_svg(&self, w: &mut impl std::io::Write, width: u32, height: u32) -> std::io::Result<()> {
        let ny = self.f.nrows();
        let nx = self.f.ncols();
        let cw = width as f64 / nx as f64;
        let ch = height as f64 / ny as f64;
        writeln!(
            w,
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">"#
        )?;
        writeln!(w, r##"<rect width="100%" height="100%" fill="#f7fbff"/>"##)?;
        for iy in 0..ny {
            for ix in 0..nx {
                if let Some(fv) = self.f[(iy, ix)] {
                    let t = (fv / 2.0).clamp(0.0, 1.0);
                    let (r, g, b) = fes_color(t);
                    let x = ix as f64 * cw;
                    let y = (ny - 1 - iy) as f64 * ch;
                    writeln!(
                        w,
                        r#"<rect x="{x:.2}" y="{y:.2}" width="{cw:.2}" height="{ch:.2}" fill="rgb({r},{g},{b})" stroke="none"/>"#
                    )?;
                }
            }
        }
        writeln!(w, "</svg>")?;
        Ok(())
    }
}

/// Orange (low F) to ice-blue (high F), matching the reference figure palette.
fn fes_color(t: f64) -> (u8, u8, u8) {
    // t=0 orange, t=0.4 purple, t=1 light blue.
    let stops = [
        (0.0, (230.0, 80.0, 20.0)),
        (0.2, (90.0, 20.0, 90.0)),
        (0.45, (40.0, 30.0, 140.0)),
        (0.7, (50.0, 110.0, 210.0)),
        (1.0, (220.0, 235.0, 250.0)),
    ];
    let t = t.clamp(0.0, 1.0);
    for w in stops.windows(2) {
        if t <= w[1].0 {
            let u = (t - w[0].0) / (w[1].0 - w[0].0);
            let r = w[0].1.0 + u * (w[1].1.0 - w[0].1.0);
            let g = w[0].1.1 + u * (w[1].1.1 - w[0].1.1);
            let b = w[0].1.2 + u * (w[1].1.2 - w[0].1.2);
            return (r as u8, g as u8, b as u8);
        }
    }
    (220, 235, 250)
}

/// Coordination number: neighbours with Euclidean distance `< cutoff`.
pub fn coordination_numbers(pos: ArrayView2<f64>, cutoff: f64) -> Array1<f64> {
    let n = pos.nrows();
    let mut cn = Array1::<f64>::zeros(n);
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let mut acc = 0.0;
            for h in 0..pos.ncols() {
                let d = pos[(i, h)] - pos[(j, h)];
                acc += d * d;
            }
            if acc.sqrt() < cutoff {
                cn[i] += 1.0;
            }
        }
    }
    cn
}

pub fn coordination_histogram(pos: ArrayView2<f64>, cutoff: f64, max_cn: usize) -> Histogram1d {
    let cn = coordination_numbers(pos, cutoff);
    let mut h = Histogram1d::new(-0.5, max_cn as f64 + 0.5, max_cn + 1).unwrap();
    h.add_many(cn.view(), None);
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn fes_min_at_dense_bin() {
        let mut h = Histogram2d::new(-1.0, 1.0, 4, -1.0, 1.0, 4).unwrap();
        for _ in 0..50 {
            h.add(0.1, 0.1, 1.0);
        }
        h.add(-0.8, -0.8, 1.0);
        let fes = FreeEnergy::from_histogram(&h, 1.0);
        let mut min_f = f64::INFINITY;
        let mut at = (0, 0);
        for iy in 0..4 {
            for ix in 0..4 {
                if let Some(f) = fes.f[(iy, ix)] {
                    if f < min_f {
                        min_f = f;
                        at = (ix, iy);
                    }
                }
            }
        }
        assert_eq!(min_f, 0.0);
        assert_eq!(at, (2, 2));
    }

    #[test]
    fn cn_of_dimer() {
        let pos = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let cn = coordination_numbers(pos.view(), 1.5);
        assert_eq!(cn[0], 1.0);
        assert_eq!(cn[1], 1.0);
        assert_eq!(cn[2], 0.0);
    }
}
