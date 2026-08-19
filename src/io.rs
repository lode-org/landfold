//! Whitespace-separated point tables: `x1 ... xD [weight]` per row.

use std::io::{BufRead, Write};
use std::path::Path;

use ndarray::{Array1, Array2};

use crate::error::{LandfoldError, Result};

#[derive(Clone, Debug)]
pub struct PointSet {
    pub points: Array2<f64>,
    pub weights: Option<Array1<f64>>,
}

pub fn read_points<R: BufRead>(r: R, dim: usize, weighted: bool) -> Result<PointSet> {
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();
    for (lineno, line) in r.lines().enumerate() {
        let line = line?;
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let nums: Vec<f64> = t
            .split_whitespace()
            .map(|s| {
                s.parse::<f64>()
                    .map_err(|e| LandfoldError::Parse(format!("line {}: {e}", lineno + 1)))
            })
            .collect::<Result<Vec<_>>>()?;
        let need = dim + usize::from(weighted);
        if nums.len() < need {
            return Err(LandfoldError::Parse(format!(
                "line {}: expected {need} columns, got {}",
                lineno + 1,
                nums.len()
            )));
        }
        rows.push(nums[..dim].to_vec());
        if weighted {
            weights.push(nums[dim]);
        }
    }
    if rows.is_empty() {
        return Err(LandfoldError::Empty);
    }
    let n = rows.len();
    let mut points = Array2::<f64>::zeros((n, dim));
    for (i, row) in rows.iter().enumerate() {
        for j in 0..dim {
            points[(i, j)] = row[j];
        }
    }
    Ok(PointSet {
        points,
        weights: if weighted {
            Some(Array1::from(weights))
        } else {
            None
        },
    })
}

pub fn read_points_path(path: &Path, dim: usize, weighted: bool) -> Result<PointSet> {
    let f = std::fs::File::open(path)?;
    read_points(std::io::BufReader::new(f), dim, weighted)
}

pub fn write_points<W: Write>(
    w: &mut W,
    pts: &Array2<f64>,
    weights: Option<&Array1<f64>>,
) -> Result<()> {
    for i in 0..pts.nrows() {
        for j in 0..pts.ncols() {
            if j > 0 {
                write!(w, " ")?;
            }
            write!(w, "{:.12}", pts[(i, j)])?;
        }
        if let Some(ww) = weights {
            write!(w, " {:.12}", ww[i])?;
        }
        writeln!(w)?;
    }
    Ok(())
}
