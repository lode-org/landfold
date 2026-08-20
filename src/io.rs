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
    if dim == 0 {
        return Err(LandfoldError::Shape("point dimension must be > 0"));
    }
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
        if nums.iter().any(|&value| !value.is_finite()) {
            return Err(LandfoldError::Parse(format!(
                "line {}: values must be finite",
                lineno + 1
            )));
        }
        let need = dim
            .checked_add(usize::from(weighted))
            .ok_or(LandfoldError::Shape("point table column count overflow"))?;
        if nums.len() != need {
            return Err(LandfoldError::Parse(format!(
                "line {}: expected exactly {need} columns, got {}",
                lineno + 1,
                nums.len()
            )));
        }
        rows.push(nums[..dim].to_vec());
        if weighted {
            if nums[dim] < 0.0 {
                return Err(LandfoldError::Parse(format!(
                    "line {}: weight must be nonnegative",
                    lineno + 1
                )));
            }
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
    if pts.nrows() == 0 || pts.ncols() == 0 {
        return Err(LandfoldError::Shape(
            "point table must be nonempty and have positive dimension",
        ));
    }
    if let Some(weights) = weights {
        if weights.len() != pts.nrows() {
            return Err(LandfoldError::Shape("point weight length"));
        }
        if weights
            .iter()
            .any(|&value| !value.is_finite() || value < 0.0)
        {
            return Err(LandfoldError::Msg(
                "point weights must be finite and nonnegative".into(),
            ));
        }
    }
    if pts.iter().any(|&value| !value.is_finite()) {
        return Err(LandfoldError::Msg(
            "point coordinates must be finite".into(),
        ));
    }
    for i in 0..pts.nrows() {
        for j in 0..pts.ncols() {
            if j > 0 {
                write!(w, " ")?;
            }
            write!(w, "{}", pts[(i, j)])?;
        }
        if let Some(ww) = weights {
            write!(w, " {}", ww[i])?;
        }
        writeln!(w)?;
    }
    Ok(())
}

/// C++ `dimred -plumed` landmark dump for PLUMED bespoke CVs.
pub fn write_plumed<W: Write>(
    w: &mut W,
    high: &Array2<f64>,
    low: &Array2<f64>,
    weights: Option<&Array1<f64>>,
) -> Result<()> {
    if high.nrows() != low.nrows() {
        return Err(LandfoldError::Shape("plumed HD/LD count mismatch"));
    }
    if high.nrows() == 0 {
        return Err(LandfoldError::Empty);
    }
    writeln!(w, "DESCRIPTION: results from sketch-map analysis in PLUMED compatible format")?;
    for i in 0..high.nrows() {
        let wt = weights.map(|ww| ww[i]).unwrap_or(1.0);
        write!(w, "REMARK WEIGHT={wt}")?;
        for h in 0..low.ncols() {
            write!(w, " SKETCHMAP.{}={}", h + 1, low[(i, h)])?;
        }
        writeln!(w)?;
        write!(w, "REMARK ARG=")?;
        for h in 0..high.ncols() {
            if h > 0 {
                write!(w, ",")?;
            }
            write!(w, "d.{}", h + 1)?;
        }
        writeln!(w)?;
        write!(w, "REMARK")?;
        for h in 0..high.ncols() {
            write!(w, " d.{}={}", h + 1, high[(i, h)])?;
        }
        writeln!(w)?;
        writeln!(w, "END")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::io::Cursor;

    #[test]
    fn rejects_nonfinite_input_values() {
        assert!(read_points(Cursor::new("0 NaN\n"), 2, false).is_err());
        assert!(read_points(Cursor::new("0 1 2\n"), 0, false).is_err());
        assert!(read_points(Cursor::new("0 1 2\n"), 2, false).is_err());
        assert!(read_points(Cursor::new("0\n"), usize::MAX, true).is_err());
    }

    #[test]
    fn rejects_negative_weights() {
        assert!(read_points(Cursor::new("0 1 -1\n"), 2, true).is_err());
        let points = Array2::zeros((1, 2));
        let mut output = Vec::new();
        assert!(write_points(&mut output, &points, Some(&Array1::from(vec![-1.0]))).is_err());
    }

    #[test]
    #[test]
    fn plumed_dump_has_ceriotti_blocks() {
        let high = array![[1.0, 2.0], [3.0, 4.0]];
        let low = array![[0.1, 0.2], [0.3, 0.4]];
        let w = array![1.0, 2.0];
        let mut buf = Vec::new();
        write_plumed(&mut buf, &high, &low, Some(&w)).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("DESCRIPTION: results from sketch-map analysis"));
        assert!(s.contains("SKETCHMAP.1=0.1"));
        assert!(s.contains("WEIGHT=2"));
        assert!(s.contains("END"));
    }

    #[test]
    fn rejects_empty_point_tables_on_write() {
        let mut output = Vec::new();
        assert!(write_points(&mut output, &Array2::zeros((0, 2)), None).is_err());
        assert!(write_points(&mut output, &Array2::zeros((1, 0)), None).is_err());
    }

    #[test]
    fn point_tables_round_trip_f64_values() {
        let points = array![[0.123_456_789_012_345_67, -9.876_543_210_987_654e123]];
        let weights = Array1::from_vec(vec![0.333_333_333_333_333_3]);
        let mut output = Vec::new();
        write_points(&mut output, &points, Some(&weights)).unwrap();
        let parsed = read_points(Cursor::new(output), 2, true).unwrap();
        assert_eq!(parsed.points, points);
        assert_eq!(parsed.weights.unwrap(), weights);
    }
}
