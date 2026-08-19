//! Weighted histograms and free-energy surfaces.
//!
//! `F/kT = -ln(ρ/ρ_max)` is the usual Boltzmann inversion of a histogram
//! (Ferrenberg and Swendsen, *Phys. Rev. Lett.* **63**, 1195 (1989),
//! <https://doi.org/10.1103/PhysRevLett.63.1195>). The cluster-FES figure
//! class of Ceriotti, Tribello and Parrinello, *J. Chem. Theory Comput.*
//! **9**, 1521 (2013), <https://doi.org/10.1021/ct3010563> is a 2-D invert
//! plus integer coordination histograms of the labelled basins.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::error::{LandfoldError, Result};

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
        if n == 0 || !lo.is_finite() || !hi.is_finite() || hi <= lo {
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

    pub fn add(&mut self, x: f64, w: f64) -> Result<()> {
        if !x.is_finite() || !w.is_finite() || w < 0.0 {
            return Err(LandfoldError::Msg(
                "histogram samples and weights must be finite; weights must be nonnegative".into(),
            ));
        }
        self.samples += w;
        if x < self.edges[0] {
            self.below += w;
            return Ok(());
        }
        let last = self.edges.len() - 1;
        if x >= self.edges[last] {
            self.above += w;
            return Ok(());
        }
        let n = self.counts.len();
        let lo = self.edges[0];
        let hi = self.edges[last];
        let t = (x - lo) / (hi - lo) * n as f64;
        let i = (t as usize).min(n - 1);
        self.counts[i] += w;
        Ok(())
    }

    pub fn add_many(&mut self, xs: ArrayView1<f64>, w: Option<ArrayView1<f64>>) -> Result<()> {
        if w.is_some_and(|weights| weights.len() != xs.len()) {
            return Err(LandfoldError::Shape("histogram weight length"));
        }
        for (i, &x) in xs.iter().enumerate() {
            let ww = w.map(|ww| ww[i]).unwrap_or(1.0);
            self.add(x, ww)?;
        }
        Ok(())
    }

    /// Integer-bin CSV (`# cn count` then `i count` per row).
    pub fn write_csv(&self, w: &mut impl std::io::Write, header: &str) -> std::io::Result<()> {
        writeln!(w, "{header}")?;
        for i in 0..self.counts.len() {
            writeln!(w, "{i} {}", self.counts[i])?;
        }
        Ok(())
    }

    pub fn write_cn_csv(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        self.write_csv(w, "# cn count")
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
        if nx == 0
            || ny == 0
            || !xlo.is_finite()
            || !xhi.is_finite()
            || !ylo.is_finite()
            || !yhi.is_finite()
            || xhi <= xlo
            || yhi <= ylo
        {
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

    pub fn add(&mut self, x: f64, y: f64, w: f64) -> Result<()> {
        if !x.is_finite() || !y.is_finite() || !w.is_finite() || w < 0.0 {
            return Err(LandfoldError::Msg(
                "histogram samples and weights must be finite; weights must be nonnegative".into(),
            ));
        }
        let nx = self.counts.ncols();
        let ny = self.counts.nrows();
        let xlo = self.x_edges[0];
        let xhi = self.x_edges[nx];
        let ylo = self.y_edges[0];
        let yhi = self.y_edges[ny];
        if x < xlo || x >= xhi || y < ylo || y >= yhi {
            return Ok(());
        }
        let ix = (((x - xlo) / (xhi - xlo) * nx as f64) as usize).min(nx - 1);
        let iy = (((y - ylo) / (yhi - ylo) * ny as f64) as usize).min(ny - 1);
        self.counts[(iy, ix)] += w;
        self.samples += w;
        Ok(())
    }

    pub fn add_points(&mut self, xy: ArrayView2<f64>, w: Option<ArrayView1<f64>>) -> Result<()> {
        if xy.ncols() < 2 {
            return Err(LandfoldError::Shape("2d histogram needs n x 2 points"));
        }
        if w.is_some_and(|weights| weights.len() != xy.nrows()) {
            return Err(LandfoldError::Shape("histogram weight length"));
        }
        for i in 0..xy.nrows() {
            let ww = w.map(|ww| ww[i]).unwrap_or(1.0);
            self.add(xy[(i, 0)], xy[(i, 1)], ww)?;
        }
        Ok(())
    }

    /// Separable Gaussian blur of the count field (binned KDE).
    pub fn blur(&self, sigma_bins: f64) -> Array2<f64> {
        blur_separable(&self.counts, sigma_bins)
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
    pub fn from_histogram(h: &Histogram2d, kt: f64) -> Result<Self> {
        validate_fes_kt(kt)?;
        Self::from_counts(h, &h.counts, kt)
    }

    /// Same invert after a separable Gaussian blur of the counts.
    pub fn from_histogram_blurred(h: &Histogram2d, kt: f64, sigma_bins: f64) -> Result<Self> {
        validate_fes_kt(kt)?;
        if !sigma_bins.is_finite() || sigma_bins < 0.0 {
            return Err(LandfoldError::Msg(
                "fes blur sigma must be finite and nonnegative".into(),
            ));
        }
        let counts = h.blur(sigma_bins);
        Self::from_counts(h, &counts, kt)
    }

    fn from_counts(h: &Histogram2d, counts: &Array2<f64>, kt: f64) -> Result<Self> {
        let nx = counts.ncols();
        let ny = counts.nrows();
        let mut x_centers = Array1::zeros(nx);
        let mut y_centers = Array1::zeros(ny);
        for i in 0..nx {
            x_centers[i] = 0.5 * (h.x_edges[i] + h.x_edges[i + 1]);
        }
        for i in 0..ny {
            y_centers[i] = 0.5 * (h.y_edges[i] + h.y_edges[i + 1]);
        }
        let mut rho = counts.clone();
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
        Ok(Self {
            x_centers,
            y_centers,
            f,
            rho,
        })
    }

    /// Clip finite F to `[0, fmax]` (JCTC 2013 panel is `fmax = 2`).
    pub fn clip(&mut self, fmax: f64) {
        for f in self.f.iter_mut().flatten() {
            *f = f.clamp(0.0, fmax);
        }
    }

    /// Drop detached islands and fill interior holes so the map is one body.
    pub fn connected_body(&mut self, floor: f64) {
        let ny = self.f.nrows();
        let nx = self.f.ncols();
        let mut mask = vec![false; ny * nx];
        let mut rmax = 0.0;
        for r in self.rho.iter() {
            if *r > rmax {
                rmax = *r;
            }
        }
        let thresh = floor * rmax;
        for iy in 0..ny {
            for ix in 0..nx {
                mask[iy * nx + ix] = self.rho[(iy, ix)] > thresh;
            }
        }
        fill_holes(&mut mask, ny, nx);
        keep_largest(&mut mask, ny, nx);
        for iy in 0..ny {
            for ix in 0..nx {
                if !mask[iy * nx + ix] {
                    self.f[(iy, ix)] = None;
                } else if self.f[(iy, ix)].is_none() {
                    self.f[(iy, ix)] = Some(0.0);
                }
            }
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
                        plus_zero(fv),
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

    /// Heatmap SVG. `fmax` is the colour scale (JCTC 2013 panel uses 2).
    pub fn write_svg(
        &self,
        w: &mut impl std::io::Write,
        width: u32,
        height: u32,
    ) -> std::io::Result<()> {
        self.write_svg_scaled(w, width, height, 2.0)
    }

    pub fn write_svg_scaled(
        &self,
        w: &mut impl std::io::Write,
        width: u32,
        height: u32,
        fmax: f64,
    ) -> std::io::Result<()> {
        let ny = self.f.nrows();
        let nx = self.f.ncols();
        let bar_h = (height as f64 * 0.08).max(16.0);
        let plot_h = (height as f64 - bar_h).max(1.0);
        let cw = width as f64 / nx as f64;
        let ch = plot_h / ny as f64;
        let fmax = if fmax > 0.0 { fmax } else { 2.0 };
        writeln!(
            w,
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">"#
        )?;
        writeln!(w, r##"<rect width="100%" height="100%" fill="#ffffff"/>"##)?;
        for iy in 0..ny {
            for ix in 0..nx {
                if let Some(fv) = self.f[(iy, ix)] {
                    let t = (fv / fmax).clamp(0.0, 1.0);
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
        let nstop = 64usize;
        let bar_y = plot_h + 4.0;
        let bar_w = width as f64 * 0.6;
        let bar_x = width as f64 * 0.2;
        let sw = bar_w / nstop as f64;
        for i in 0..nstop {
            let t = i as f64 / (nstop - 1) as f64;
            let (r, g, b) = fes_color(t);
            let x = bar_x + i as f64 * sw;
            writeln!(
                w,
                r#"<rect x="{x:.2}" y="{bar_y:.2}" width="{sw:.2}" height="{:.2}" fill="rgb({r},{g},{b})" stroke="none"/>"#,
                bar_h - 8.0
            )?;
        }
        writeln!(
            w,
            r##"<text x="{:.2}" y="{:.2}" font-size="11" fill="#222">0</text>"##,
            bar_x,
            height as f64 - 2.0
        )?;
        writeln!(
            w,
            r##"<text x="{:.2}" y="{:.2}" font-size="11" fill="#222" text-anchor="end">F/kT = {fmax}</text>"##,
            bar_x + bar_w,
            height as f64 - 2.0
        )?;
        writeln!(w, "</svg>")?;
        Ok(())
    }
}

/// Histogram the embedded (x, y), invert to F = -kT ln(ρ/ρ_max).
pub fn fes_from_points(
    xy: ArrayView2<f64>,
    nx: usize,
    ny: usize,
    kt: f64,
    pad: f64,
    weights: Option<ArrayView1<f64>>,
) -> Result<FreeEnergy> {
    if xy.ncols() < 2 || xy.nrows() == 0 {
        return Err(LandfoldError::Shape("fes needs n x 2 points"));
    }
    if !kt.is_finite() || kt <= 0.0 {
        return Err(LandfoldError::Msg("fes kT must be finite and > 0".into()));
    }
    if !pad.is_finite() || pad < 0.0 {
        return Err(LandfoldError::Msg("fes pad must be finite and >= 0".into()));
    }
    if xy.iter().any(|&value| !value.is_finite()) {
        return Err(LandfoldError::Msg("fes coordinates must be finite".into()));
    }
    if weights.is_some_and(|w| {
        w.len() != xy.nrows() || w.iter().any(|&value| !value.is_finite() || value < 0.0)
    }) {
        return Err(LandfoldError::Msg(
            "fes weights must be finite and nonnegative".into(),
        ));
    }
    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymin = f64::INFINITY;
    let mut ymax = f64::NEG_INFINITY;
    for i in 0..xy.nrows() {
        xmin = xmin.min(xy[(i, 0)]);
        xmax = xmax.max(xy[(i, 0)]);
        ymin = ymin.min(xy[(i, 1)]);
        ymax = ymax.max(xy[(i, 1)]);
    }
    let px = pad * (xmax - xmin).max(1e-6);
    let py = pad * (ymax - ymin).max(1e-6);
    let mut h = Histogram2d::new(xmin - px, xmax + px, nx, ymin - py, ymax + py, ny)?;
    h.add_points(xy, weights)?;
    FreeEnergy::from_histogram(&h, kt)
}

fn validate_fes_kt(kt: f64) -> Result<()> {
    if !kt.is_finite() || kt <= 0.0 {
        return Err(LandfoldError::Msg("fes kT must be finite and > 0".into()));
    }
    Ok(())
}

fn plus_zero(x: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x }
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

fn gauss1d(sigma: f64) -> Vec<f64> {
    let radius = ((3.0 * sigma).ceil() as usize).max(1);
    let s2 = 2.0 * sigma * sigma.max(1e-12);
    let mut k = Vec::with_capacity(2 * radius + 1);
    let mut sum = 0.0;
    for i in 0..=2 * radius {
        let x = i as f64 - radius as f64;
        let v = (-x * x / s2).exp();
        k.push(v);
        sum += v;
    }
    if sum > 0.0 {
        for v in &mut k {
            *v /= sum;
        }
    }
    k
}

fn blur_separable(z: &Array2<f64>, sigma: f64) -> Array2<f64> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return z.clone();
    }
    let k = gauss1d(sigma);
    let radius = k.len() / 2;
    let ny = z.nrows();
    let nx = z.ncols();
    let mut tmp = Array2::<f64>::zeros((ny, nx));
    for iy in 0..ny {
        for ix in 0..nx {
            let mut acc = 0.0;
            for (t, &kv) in k.iter().enumerate() {
                let j = ix as isize + t as isize - radius as isize;
                if j >= 0 && (j as usize) < nx {
                    acc += z[(iy, j as usize)] * kv;
                }
            }
            tmp[(iy, ix)] = acc;
        }
    }
    let mut out = Array2::<f64>::zeros((ny, nx));
    for iy in 0..ny {
        for ix in 0..nx {
            let mut acc = 0.0;
            for (t, &kv) in k.iter().enumerate() {
                let i = iy as isize + t as isize - radius as isize;
                if i >= 0 && (i as usize) < ny {
                    acc += tmp[(i as usize, ix)] * kv;
                }
            }
            out[(iy, ix)] = acc;
        }
    }
    out
}

fn fill_holes(mask: &mut [bool], ny: usize, nx: usize) {
    let mut reach = vec![false; ny * nx];
    let mut stack = Vec::new();
    for i in 0..ny {
        if !mask[i * nx] {
            stack.push((i, 0));
        }
        if !mask[i * nx + nx - 1] {
            stack.push((i, nx - 1));
        }
    }
    for j in 0..nx {
        if !mask[j] {
            stack.push((0, j));
        }
        if !mask[(ny - 1) * nx + j] {
            stack.push((ny - 1, j));
        }
    }
    while let Some((i, j)) = stack.pop() {
        if i >= ny || j >= nx {
            continue;
        }
        let idx = i * nx + j;
        if reach[idx] || mask[idx] {
            continue;
        }
        reach[idx] = true;
        if i > 0 {
            stack.push((i - 1, j));
        }
        if i + 1 < ny {
            stack.push((i + 1, j));
        }
        if j > 0 {
            stack.push((i, j - 1));
        }
        if j + 1 < nx {
            stack.push((i, j + 1));
        }
    }
    for i in 0..ny * nx {
        if !mask[i] && !reach[i] {
            mask[i] = true;
        }
    }
}

fn keep_largest(mask: &mut [bool], ny: usize, nx: usize) {
    let mut seen = vec![false; ny * nx];
    let mut best: Vec<(usize, usize)> = Vec::new();
    for i0 in 0..ny {
        for j0 in 0..nx {
            if !mask[i0 * nx + j0] || seen[i0 * nx + j0] {
                continue;
            }
            let mut stack = vec![(i0, j0)];
            let mut cells = Vec::new();
            while let Some((i, j)) = stack.pop() {
                if i >= ny || j >= nx {
                    continue;
                }
                let idx = i * nx + j;
                if seen[idx] || !mask[idx] {
                    continue;
                }
                seen[idx] = true;
                cells.push((i, j));
                if i > 0 {
                    stack.push((i - 1, j));
                }
                if i + 1 < ny {
                    stack.push((i + 1, j));
                }
                if j > 0 {
                    stack.push((i, j - 1));
                }
                if j + 1 < nx {
                    stack.push((i, j + 1));
                }
            }
            if cells.len() > best.len() {
                best = cells;
            }
        }
    }
    mask.fill(false);
    for (i, j) in best {
        mask[i * nx + j] = true;
    }
}

/// Coordination number: neighbours with Euclidean distance `< cutoff`.
pub fn coordination_numbers(pos: ArrayView2<f64>, cutoff: f64) -> Result<Array1<f64>> {
    if !cutoff.is_finite() || cutoff < 0.0 {
        return Err(LandfoldError::Msg(
            "coordination cutoff must be finite and nonnegative".into(),
        ));
    }
    if pos.iter().any(|&value| !value.is_finite()) {
        return Err(LandfoldError::Msg(
            "coordination coordinates must be finite".into(),
        ));
    }
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
    Ok(cn)
}

pub fn coordination_histogram(
    pos: ArrayView2<f64>,
    cutoff: f64,
    max_cn: usize,
) -> Result<Histogram1d> {
    let cn = coordination_numbers(pos, cutoff)?;
    let mut h = Histogram1d::new(-0.5, max_cn as f64 + 0.5, max_cn + 1)?;
    h.add_many(cn.view(), None)?;
    Ok(h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn fes_min_at_dense_bin() {
        let mut h = Histogram2d::new(-1.0, 1.0, 4, -1.0, 1.0, 4).unwrap();
        for _ in 0..50 {
            h.add(0.1, 0.1, 1.0).unwrap();
        }
        h.add(-0.8, -0.8, 1.0).unwrap();
        let fes = FreeEnergy::from_histogram(&h, 1.0).unwrap();
        let mut min_f = f64::INFINITY;
        let mut at = (0, 0);
        for iy in 0..4 {
            for ix in 0..4 {
                match fes.f[(iy, ix)] {
                    Some(f) if f < min_f => {
                        min_f = f;
                        at = (ix, iy);
                    }
                    _ => {}
                }
            }
        }
        assert_eq!(min_f, 0.0);
        assert_eq!(at, (2, 2));
    }

    #[test]
    fn rejects_nonfinite_fes_inputs() {
        assert!(Histogram1d::new(f64::NAN, 1.0, 4).is_err());
        assert!(Histogram2d::new(0.0, f64::NAN, 4, 0.0, 1.0, 4).is_err());
        let points = array![[0.0, f64::NAN], [1.0, 0.0]];
        assert!(fes_from_points(points.view(), 4, 4, 1.0, 0.05, None).is_err());
        let points = array![[0.0, 0.0], [1.0, 1.0]];
        assert!(fes_from_points(points.view(), 4, 4, 0.0, 0.05, None).is_err());
        assert!(fes_from_points(points.view(), 4, 4, 1.0, -0.1, None).is_err());
    }

    #[test]
    fn rejects_invalid_fes_constructor_parameters() {
        let h = Histogram2d::new(0.0, 1.0, 2, 0.0, 1.0, 2).unwrap();
        for kt in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(FreeEnergy::from_histogram(&h, kt).is_err());
        }
        assert!(FreeEnergy::from_histogram_blurred(&h, 1.0, -1.0).is_err());
        assert!(FreeEnergy::from_histogram_blurred(&h, 1.0, f64::NAN).is_err());
    }

    #[test]
    fn rejects_mismatched_histogram_weights() {
        let mut h1 = Histogram1d::new(0.0, 1.0, 2).unwrap();
        assert!(
            h1.add_many(array![0.1, 0.2].view(), Some(array![1.0].view()))
                .is_err()
        );

        let mut h2 = Histogram2d::new(0.0, 1.0, 2, 0.0, 1.0, 2).unwrap();
        assert!(
            h2.add_points(
                array![[0.1, 0.1], [0.2, 0.2]].view(),
                Some(array![1.0].view())
            )
            .is_err()
        );
        assert!(h2.add_points(array![[0.1]].view(), None).is_err());
    }

    #[test]
    fn rejects_invalid_histogram_samples_and_weights() {
        let mut h1 = Histogram1d::new(0.0, 1.0, 2).unwrap();
        assert!(h1.add(f64::NAN, 1.0).is_err());
        assert!(h1.add(0.5, -1.0).is_err());
        assert!(
            h1.add_many(array![0.5].view(), Some(array![f64::NAN].view()))
                .is_err()
        );

        let mut h2 = Histogram2d::new(0.0, 1.0, 2, 0.0, 1.0, 2).unwrap();
        assert!(h2.add(0.5, f64::INFINITY, 1.0).is_err());
        assert!(
            h2.add_points(array![[0.5, 0.5]].view(), Some(array![-1.0].view()))
                .is_err()
        );
    }

    #[test]
    fn cn_of_dimer() {
        let pos = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let cn = coordination_numbers(pos.view(), 1.5).unwrap();
        assert_eq!(cn[0], 1.0);
        assert_eq!(cn[1], 1.0);
        assert_eq!(cn[2], 0.0);
        let h = coordination_histogram(pos.view(), 1.5, 12).unwrap();
        let mut buf = Vec::new();
        h.write_cn_csv(&mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.starts_with("# cn count\n"));
        assert!(s.contains("0 1\n"));
        assert!(s.contains("1 2\n"));
    }

    #[test]
    fn rejects_invalid_coordination_inputs() {
        let pos = array![[0.0, 0.0], [f64::NAN, 1.0]];
        assert!(coordination_numbers(pos.view(), 1.0).is_err());
        assert!(coordination_numbers(array![[0.0], [1.0]].view(), -1.0).is_err());
        assert!(coordination_numbers(array![[0.0], [1.0]].view(), f64::NAN).is_err());
    }

    #[test]
    fn blur_spreads_mass_into_empty_bins() {
        let mut h = Histogram2d::new(0.0, 2.0, 2, 0.0, 2.0, 2).unwrap();
        h.add(0.5, 0.5, 1.0).unwrap();
        let sharp = FreeEnergy::from_histogram(&h, 1.0).unwrap();
        let soft = FreeEnergy::from_histogram_blurred(&h, 1.0, 1.0).unwrap();
        assert_eq!(sharp.f[(0, 1)], None);
        assert!(soft.rho[(0, 1)] > 0.0);
        assert!(soft.f[(0, 1)].is_some());
    }

    #[test]
    fn connected_body_drops_a_detached_island() {
        let mut h = Histogram2d::new(0.0, 3.0, 3, 0.0, 3.0, 3).unwrap();
        h.add(0.5, 0.5, 10.0).unwrap();
        h.add(0.5, 1.5, 10.0).unwrap();
        h.add(2.5, 2.5, 1.0).unwrap();
        let mut fes = FreeEnergy::from_histogram(&h, 1.0).unwrap();
        assert!(fes.f[(2, 2)].is_some());
        fes.connected_body(0.05);
        assert!(fes.f[(0, 0)].is_some());
        assert!(fes.f[(1, 0)].is_some());
        assert_eq!(fes.f[(2, 2)], None);
    }

    #[test]
    fn half_density_invert() {
        let mut h = Histogram2d::new(0.0, 2.0, 2, 0.0, 2.0, 2).unwrap();
        h.add(0.5, 0.5, 1.0).unwrap();
        h.add(0.5, 0.5, 1.0).unwrap();
        h.add(1.5, 1.5, 1.0).unwrap();
        let fes = FreeEnergy::from_histogram(&h, 1.0).unwrap();
        assert_eq!(fes.f[(0, 0)], Some(0.0));
        assert_eq!(fes.f[(0, 1)], None);
        assert_eq!(fes.f[(1, 0)], None);
        let f11 = fes.f[(1, 1)].unwrap();
        assert!((f11 - 0.5_f64.ln().abs()).abs() < 1e-12);
        assert!((fes.rho[(0, 0)] - 2.0 / 3.0).abs() < 1e-12);
        assert!((fes.rho[(1, 1)] - 1.0 / 3.0).abs() < 1e-12);
    }
}
