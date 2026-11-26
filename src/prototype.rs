//! Weighted Gaussian partition of centered descriptor rows.
//!
//! Near-duplicate rows collapse to prototypes. A Gaussian mean shift then
//! runs at a caller-supplied fraction of the cloud radius, and each input
//! row receives the label of its prototype.

use ndarray::{Array2, ArrayView1, ArrayView2};

use crate::error::{LandfoldError, Result};

/// Distances within this absolute tolerance are the same medoid distance.
pub const SUBSET_MEDOID_TIE_TOL: f64 = 1.0e-12;

const MEAN_SHIFT_ITERS: usize = 120;
const MEAN_SHIFT_TOLERANCE: f64 = 1.0e-6;
const MEAN_SHIFT_MERGE_FACTOR: f64 = 0.5;
const MEAN_SHIFT_KERNEL_CUTOFF: f64 = 32.0;

/// Partition centered descriptor rows.
///
/// Columns are centered first. The cloud scale is the largest distance from
/// that centroid. Rows within `dedup_fraction` of that scale collapse to one
/// prototype and their weights add. Gaussian mean shift then uses bandwidth
/// `bandwidth_fraction` of the same scale, kernel `exp(-1/2 d²/h²)`, and
/// merges modes that finish within half a bandwidth. Each input row gets one
/// label. `weights` defaults to one per row. A zero scale, or a zero
/// bandwidth fraction, returns the prototype labels without shifting.
pub fn gaussian_prototype_partition(
    points: ArrayView2<f64>,
    weights: Option<&[f64]>,
    dedup_fraction: f64,
    bandwidth_fraction: f64,
) -> Result<Vec<usize>> {
    let n = points.nrows();
    let width = points.ncols();
    if n == 0 {
        return Err(LandfoldError::Empty);
    }
    if width == 0 {
        return Err(LandfoldError::Shape(
            "prototype rows must have at least one feature",
        ));
    }
    if points.iter().any(|value| !value.is_finite()) {
        return Err(LandfoldError::Msg("prototype rows must be finite".into()));
    }
    if !dedup_fraction.is_finite() || dedup_fraction < 0.0 {
        return Err(LandfoldError::Msg(
            "prototype dedup fraction must be finite and nonnegative".into(),
        ));
    }
    if !bandwidth_fraction.is_finite() || bandwidth_fraction < 0.0 {
        return Err(LandfoldError::Msg(
            "prototype bandwidth fraction must be finite and nonnegative".into(),
        ));
    }
    let weights = match weights {
        Some(values) if values.len() == n => {
            if values
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            {
                return Err(LandfoldError::Msg(
                    "prototype weights must be finite and nonnegative".into(),
                ));
            }
            values.to_vec()
        }
        Some(_) => {
            return Err(LandfoldError::Shape(
                "prototype weights must have one value per row",
            ));
        }
        None => vec![1.0; n],
    };

    let mut centered = Array2::zeros((n, width));
    for axis in 0..width {
        let mean = points.column(axis).sum() / n as f64;
        for row in 0..n {
            centered[(row, axis)] = points[(row, axis)] - mean;
        }
    }
    let mut scale = 0.0;
    for row in centered.rows() {
        let norm = row.iter().map(|value| value * value).sum::<f64>().sqrt();
        if norm > scale {
            scale = norm;
        }
    }
    if scale == 0.0 {
        return Ok(vec![0; n]);
    }

    let (prototypes, prototype_weights, row_to_prototype) =
        collapse_prototypes(&centered, &weights, dedup_fraction * scale);
    let bandwidth = bandwidth_fraction * scale;
    if bandwidth == 0.0 {
        return Ok(row_to_prototype);
    }
    if !(bandwidth * bandwidth).is_normal() {
        return Err(LandfoldError::Msg(
            "prototype bandwidth square must be a normal positive finite value".into(),
        ));
    }
    let prototype_labels = weighted_gaussian_shift(&prototypes, &prototype_weights, bandwidth)?;
    Ok(row_to_prototype
        .into_iter()
        .map(|prototype| prototype_labels[prototype])
        .collect())
}

/// Medoid of a caller-supplied subset.
///
/// The medoid is the subset row closest to the subset centroid. Distances
/// within [`SUBSET_MEDOID_TIE_TOL`] are tied. The higher score wins a tie,
/// and a remaining tie keeps the smaller row index. With no scores, the
/// smaller index wins every tie. A zero-width matrix uses scores only.
/// `scores`, when supplied, has one value per row of `points`.
pub fn subset_medoid(
    points: ArrayView2<f64>,
    indices: &[usize],
    scores: Option<&[f64]>,
) -> Result<usize> {
    if indices.is_empty() {
        return Err(LandfoldError::Empty);
    }
    if points.iter().any(|value| !value.is_finite()) {
        return Err(LandfoldError::Msg("medoid points must be finite".into()));
    }
    if let Some(scores) = scores {
        if scores.len() != points.nrows() {
            return Err(LandfoldError::Shape(
                "medoid scores must have one value per row",
            ));
        }
        if scores.iter().any(|score| !score.is_finite()) {
            return Err(LandfoldError::Msg("medoid scores must be finite".into()));
        }
    }
    let mut seen = vec![false; points.nrows()];
    for &index in indices {
        if index >= points.nrows() {
            return Err(LandfoldError::Msg(format!(
                "medoid index {index} is outside {} rows",
                points.nrows()
            )));
        }
        if seen[index] {
            return Err(LandfoldError::Msg(format!(
                "medoid index {index} is repeated"
            )));
        }
        seen[index] = true;
    }

    let mut ordered = indices.to_vec();
    ordered.sort_unstable();
    let mut best = ordered[0];
    if points.ncols() == 0 {
        for &member in &ordered[1..] {
            if score_is_higher(scores, member, best) {
                best = member;
            }
        }
        return Ok(best);
    }

    let mut centroid = vec![0.0; points.ncols()];
    let inv_count = 1.0 / ordered.len() as f64;
    for &member in &ordered {
        for axis in 0..points.ncols() {
            centroid[axis] += points[(member, axis)] * inv_count;
        }
    }
    let mut best_distance = euclidean(points.row(best), &centroid);
    for &member in &ordered[1..] {
        let distance = euclidean(points.row(member), &centroid);
        let closer = distance < best_distance - SUBSET_MEDOID_TIE_TOL;
        let tied = (distance - best_distance).abs() <= SUBSET_MEDOID_TIE_TOL;
        if closer || (tied && score_is_higher(scores, member, best)) {
            best_distance = distance;
            best = member;
        }
    }
    Ok(best)
}

fn score_is_higher(scores: Option<&[f64]>, member: usize, best: usize) -> bool {
    scores.is_some_and(|scores| scores[member] > scores[best])
}

fn squared(left: ArrayView1<f64>, right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| {
            let delta = a - b;
            delta * delta
        })
        .sum()
}

fn euclidean(point: ArrayView1<f64>, centroid: &[f64]) -> f64 {
    squared(point, centroid).sqrt()
}

fn collapse_prototypes(
    points: &Array2<f64>,
    weights: &[f64],
    tolerance: f64,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<usize>) {
    let n = points.nrows();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&left, &right| {
        points[(left, 0)]
            .total_cmp(&points[(right, 0)])
            .then(left.cmp(&right))
    });
    let tolerance_sq = tolerance * tolerance;
    let mut prototypes: Vec<Vec<f64>> = Vec::new();
    let mut prototype_weights = Vec::new();
    let mut row_to_prototype = vec![0usize; n];
    let mut window_start = 0usize;
    for row in order {
        let leading = points[(row, 0)];
        while window_start < prototypes.len() && leading - prototypes[window_start][0] > tolerance {
            window_start += 1;
        }
        let mut matched = None;
        for prototype in window_start..prototypes.len() {
            if squared(points.row(row), &prototypes[prototype]) <= tolerance_sq {
                matched = Some(prototype);
                break;
            }
        }
        if let Some(prototype) = matched {
            prototype_weights[prototype] += weights[row];
            row_to_prototype[row] = prototype;
        } else {
            let prototype = prototypes.len();
            prototypes.push(points.row(row).to_vec());
            prototype_weights.push(weights[row]);
            row_to_prototype[row] = prototype;
        }
    }
    (prototypes, prototype_weights, row_to_prototype)
}

fn weighted_gaussian_shift(
    prototypes: &[Vec<f64>],
    weights: &[f64],
    bandwidth: f64,
) -> Result<Vec<usize>> {
    let count = prototypes.len();
    let width = prototypes.first().map_or(0, Vec::len);
    let inv_two_h2 = 0.5 / (bandwidth * bandwidth);
    if !inv_two_h2.is_finite() {
        return Err(LandfoldError::Msg(
            "prototype bandwidth square must be a normal positive finite value".into(),
        ));
    }
    let convergence_sq = (MEAN_SHIFT_TOLERANCE * bandwidth).powi(2);
    let merge_sq = (MEAN_SHIFT_MERGE_FACTOR * bandwidth).powi(2);
    let mut seeds: Vec<usize> = (0..count).collect();
    seeds.sort_by(|&left, &right| {
        weights[right]
            .total_cmp(&weights[left])
            .then(left.cmp(&right))
    });

    let mut centers: Vec<Vec<f64>> = Vec::new();
    let mut labels = vec![0usize; count];
    for &seed in &seeds {
        let mut position = prototypes[seed].clone();
        for _ in 0..MEAN_SHIFT_ITERS {
            let mut numerator = vec![0.0; width];
            let mut denominator = 0.0;
            for (index, prototype) in prototypes.iter().enumerate() {
                let mut distance_sq = 0.0;
                for axis in 0..width {
                    let delta = position[axis] - prototype[axis];
                    distance_sq += delta * delta;
                }
                let argument = distance_sq * inv_two_h2;
                if argument <= MEAN_SHIFT_KERNEL_CUTOFF {
                    let weight = weights[index] * (-argument).exp();
                    denominator += weight;
                    for axis in 0..width {
                        numerator[axis] += weight * prototype[axis];
                    }
                }
            }
            if denominator <= 0.0 || !denominator.is_finite() {
                break;
            }
            let mut shift_sq = 0.0;
            for axis in 0..width {
                let next = numerator[axis] / denominator;
                let delta = next - position[axis];
                shift_sq += delta * delta;
                position[axis] = next;
            }
            if !shift_sq.is_finite() {
                return Err(LandfoldError::Msg(
                    "prototype mean shift left the finite range".into(),
                ));
            }
            if shift_sq <= convergence_sq {
                break;
            }
        }
        let mut matched = None;
        for (label, center) in centers.iter().enumerate() {
            let mut distance_sq = 0.0;
            for axis in 0..width {
                let delta = position[axis] - center[axis];
                distance_sq += delta * delta;
            }
            if distance_sq <= merge_sq {
                matched = Some(label);
                break;
            }
        }
        if let Some(label) = matched {
            labels[seed] = label;
        } else {
            labels[seed] = centers.len();
            centers.push(position);
        }
    }
    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn duplicates_share_a_label_and_a_distant_row_does_not() {
        let points = array![[0.0, 0.0], [0.0, 0.0], [4.0, 0.0]];
        let labels = gaussian_prototype_partition(points.view(), None, 0.0, 0.05).unwrap();
        assert_eq!(labels[0], labels[1]);
        assert_ne!(labels[0], labels[2]);

        let shifted = points.map(|value| value + 3.0);
        let shifted_labels = gaussian_prototype_partition(shifted.view(), None, 0.0, 0.05).unwrap();
        assert_eq!(labels, shifted_labels);
    }

    #[test]
    fn near_duplicates_collapse_when_the_bandwidth_fraction_is_zero() {
        let points = array![[0.0], [1.0e-6], [1.0]];
        let collapsed = gaussian_prototype_partition(points.view(), None, 1.0e-4, 0.0).unwrap();
        assert_eq!(collapsed[0], collapsed[1]);
        assert_ne!(collapsed[0], collapsed[2]);

        let distinct = gaussian_prototype_partition(points.view(), None, 0.0, 0.0).unwrap();
        assert_ne!(distinct[0], distinct[1]);
        assert_ne!(distinct[0], distinct[2]);
    }

    #[test]
    fn a_heavier_neighbor_pulls_the_midpoint_into_one_class() {
        let points = array![[0.0], [1.0], [0.35]];
        let equal = gaussian_prototype_partition(points.view(), None, 0.0, 0.35).unwrap();
        assert_eq!(equal[0], equal[2]);
        assert_ne!(equal[0], equal[1]);

        let heavy_right =
            gaussian_prototype_partition(points.view(), Some(&[1.0, 80.0, 1.0]), 0.0, 0.35)
                .unwrap();
        assert!(heavy_right.iter().all(|label| *label == heavy_right[0]));
    }

    #[test]
    fn identical_rows_are_one_class() {
        let points = Array2::from_elem((4, 2), 1.5);
        let labels = gaussian_prototype_partition(points.view(), None, 1.0e-9, 0.025).unwrap();
        assert_eq!(labels, vec![0, 0, 0, 0]);
    }

    #[test]
    fn partition_rejects_bad_shapes_and_weights() {
        let points = array![[0.0], [1.0]];
        assert!(gaussian_prototype_partition(points.view(), None, -0.1, 0.1).is_err());
        assert!(gaussian_prototype_partition(points.view(), Some(&[1.0]), 0.0, 0.1).is_err());
        assert!(
            gaussian_prototype_partition(points.view(), Some(&[1.0, f64::NAN]), 0.0, 0.1).is_err()
        );
        let empty = Array2::<f64>::zeros((0, 2));
        assert!(gaussian_prototype_partition(empty.view(), None, 0.0, 0.1).is_err());
    }

    #[test]
    fn subset_medoid_breaks_distance_ties_by_score_then_index() {
        let points = array![[0.0, 0.0], [2.0, 0.0], [1.0, 0.0]];
        assert_eq!(subset_medoid(points.view(), &[0, 1, 2], None).unwrap(), 2);
        assert_eq!(subset_medoid(points.view(), &[1, 0], None).unwrap(), 0);
        assert_eq!(
            subset_medoid(points.view(), &[0, 1], Some(&[0.0, 1.0, 0.0])).unwrap(),
            1
        );
        assert_eq!(
            subset_medoid(points.view(), &[1, 0], Some(&[0.5, 0.5, 9.0])).unwrap(),
            0
        );
        assert_eq!(
            subset_medoid(points.view(), &[2, 0, 1], Some(&[10.0, 10.0, 0.0])).unwrap(),
            2
        );

        let scores_only = Array2::<f64>::zeros((3, 0));
        assert_eq!(
            subset_medoid(scores_only.view(), &[2, 0, 1], Some(&[1.0, 3.0, 2.0])).unwrap(),
            1
        );
        assert_eq!(subset_medoid(scores_only.view(), &[2, 0], None).unwrap(), 0);
    }

    #[test]
    fn subset_medoid_rejects_empty_repeated_and_out_of_range_indices() {
        let points = array![[0.0], [1.0]];
        assert!(subset_medoid(points.view(), &[], None).is_err());
        assert!(subset_medoid(points.view(), &[0, 0], None).is_err());
        assert!(subset_medoid(points.view(), &[2], None).is_err());
        assert!(subset_medoid(points.view(), &[0], Some(&[1.0])).is_err());
    }
}
