//! Descriptor Embedding and Clustering for Atomistic-environment Framework.
//!
//! This module is descriptor-agnostic. It implements the classical MDS,
//! broken-stick, bandwidth, and mean-shift stages of Lai et al., J. Chem.
//! Phys. 159, 024129 (2023), DOI 10.1063/5.0160369. Callers provide either
//! descriptor rows or a symmetric dissimilarity matrix.

use ndarray::{Array2, ArrayView2};

use crate::error::{LandfoldError, Result};
use crate::mds::{ClassicalMdsModel, MdsDimension};

const MAX_BANDWIDTH_BINS: usize = 1_000_000;
/// A gap has to fall to at most half the mode on its left. A shallower dip is
/// a shoulder ripple of that same mode.
const VALLEY_MAX_DEPTH: f64 = 0.5;
/// Density after a gap has to recover to at least this fraction of the left mode.
const VALLEY_MIN_RECOVERY: f64 = 0.2;
/// At least this share of pairwise distances has to lie strictly beyond a gap.
const VALLEY_MIN_FAR_FRACTION: f64 = 0.05;

/// MDS dimension selection for DECAF.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DimensionChoice {
    /// Eq. 5 broken-stick selection.
    #[default]
    BrokenStick,
    /// Explicit embedding dimension. A nonuniform fit errors when this exceeds
    /// numerical rank; an exactly zero-dissimilarity fit is represented as one
    /// class with zero embedding axes.
    Fixed(usize),
}

/// Mean-shift bandwidth selection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BandwidthChoice {
    /// First gap of the Gaussian-smeared pair-distance distribution from the
    /// paper, using the supplied smearing width. A dip counts only when it
    /// falls to half the mode on its left, a later mode recovers, and at
    /// least five percent of pairs lie beyond it. A gapless cloud uses the
    /// median nearest-neighbor distance, unless the distances are a single
    /// shell at the diameter.
    Paper { sigma: f64 },
    /// Explicit mean-shift radius.
    Fixed(f64),
}

impl Default for BandwidthChoice {
    fn default() -> Self {
        Self::Paper { sigma: 0.0092 }
    }
}

/// Generic DECAF fit options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecafOptions {
    /// MDS dimension rule.
    pub dimensions: DimensionChoice,
    /// Mean-shift bandwidth rule.
    pub bandwidth: BandwidthChoice,
    /// Maximum mean-shift updates per seed.
    pub max_iterations: usize,
    /// Convergence tolerance relative to the bandwidth.
    pub tolerance: f64,
}

impl Default for DecafOptions {
    fn default() -> Self {
        Self {
            dimensions: DimensionChoice::BrokenStick,
            bandwidth: BandwidthChoice::default(),
            max_iterations: 800,
            tolerance: 1.0e-6,
        }
    }
}

/// Fitted generic DECAF classification.
#[derive(Clone, Debug)]
pub struct Decaf {
    descriptors: Option<Array2<f64>>,
    mds: Option<ClassicalMdsModel>,
    embedded: Array2<f64>,
    centers: Array2<f64>,
    labels: Vec<usize>,
    memberships: Array2<f64>,
    representatives: Vec<usize>,
    bandwidth: f64,
}

/// Out-of-sample DECAF projection and fuzzy classification.
#[derive(Clone, Debug)]
pub struct DecafPrediction {
    embedded: Array2<f64>,
    labels: Vec<usize>,
    memberships: Array2<f64>,
}

impl DecafPrediction {
    /// Projected coordinates at the paper/reference package's two-times
    /// conventional classical-MDS scale.
    pub fn embedded(&self) -> &Array2<f64> {
        &self.embedded
    }

    /// Hard nearest-center class labels.
    pub fn labels(&self) -> &[usize] {
        &self.labels
    }

    /// Row-normalized fuzzy affinities to all fitted class centers.
    pub fn memberships(&self) -> &Array2<f64> {
        &self.memberships
    }
}

impl Decaf {
    /// Number of identified classes.
    pub fn n_classes(&self) -> usize {
        self.centers.nrows().max(1)
    }

    /// Hard nearest-mode class labels.
    pub fn labels(&self) -> &[usize] {
        &self.labels
    }

    /// Row-normalized fuzzy affinities to all class modes.
    pub fn memberships(&self) -> &Array2<f64> {
        &self.memberships
    }

    /// One observed medoid index per class.
    pub fn representatives(&self) -> &[usize] {
        &self.representatives
    }

    /// Embedded training coordinates.
    pub fn embedded(&self) -> &Array2<f64> {
        &self.embedded
    }

    /// Fitted mean-shift class centers in embedded space.
    pub fn centers(&self) -> &Array2<f64> {
        &self.centers
    }

    /// Mean-shift bandwidth used for this fit.
    pub fn bandwidth(&self) -> f64 {
        self.bandwidth
    }

    /// Fitted classical MDS model, absent only for a zero-dissimilarity set.
    pub fn mds(&self) -> Option<&ClassicalMdsModel> {
        self.mds.as_ref()
    }

    /// Project new descriptor vectors through the fitted Eq. 3-4 operator.
    pub fn project_descriptors(&self, points: ArrayView2<f64>) -> Result<Array2<f64>> {
        let training = self.descriptors.as_ref().ok_or_else(|| {
            LandfoldError::Msg(
                "descriptor projection is unavailable for a dissimilarity-only DECAF fit".into(),
            )
        })?;
        if points.ncols() != training.ncols() {
            return Err(LandfoldError::Shape(
                "new descriptors must have the training feature width",
            ));
        }
        let distances = cross_distances(training.view(), points)?;
        self.project_distances(distances.view())
    }

    /// Project new points from distances to all training points.
    pub fn project_distances(&self, distances: ArrayView2<f64>) -> Result<Array2<f64>> {
        self.mds
            .as_ref()
            .ok_or_else(|| LandfoldError::Msg("uniform DECAF fit has no MDS projector".into()))?
            .project_distances(distances)
    }

    /// Classify already projected points against the fitted mean-shift centers.
    pub fn classify(&self, embedded: ArrayView2<f64>) -> Result<DecafPrediction> {
        if embedded.ncols() != self.centers.ncols() {
            return Err(LandfoldError::Shape(
                "projected points must have the fitted DECAF dimension",
            ));
        }
        if embedded.iter().any(|value| !value.is_finite()) {
            return Err(LandfoldError::Msg(
                "projected DECAF points must be finite".into(),
            ));
        }
        Ok(DecafPrediction {
            embedded: embedded.to_owned(),
            labels: nearest_labels(embedded, self.centers.view()),
            memberships: fuzzy_memberships(embedded, self.centers.view(), self.bandwidth)?,
        })
    }

    /// Project and classify new descriptor vectors.
    pub fn predict_descriptors(&self, points: ArrayView2<f64>) -> Result<DecafPrediction> {
        let embedded = self.project_descriptors(points)?;
        self.classify(embedded.view())
    }

    /// Project and classify new points from distances to all training points.
    pub fn predict_distances(&self, distances: ArrayView2<f64>) -> Result<DecafPrediction> {
        let embedded = self.project_distances(distances)?;
        self.classify(embedded.view())
    }
}

/// Fit DECAF to descriptor vectors using Euclidean dissimilarity.
pub fn decaf_from_descriptors(
    descriptors: ArrayView2<f64>,
    options: &DecafOptions,
) -> Result<Decaf> {
    if descriptors.nrows() == 0 {
        return Err(LandfoldError::Empty);
    }
    if descriptors.ncols() == 0 {
        return Err(LandfoldError::Shape(
            "DECAF descriptors must have at least one feature",
        ));
    }
    if descriptors.iter().any(|value| !value.is_finite()) {
        return Err(LandfoldError::Msg(
            "DECAF descriptors must be finite".into(),
        ));
    }
    let distances = cross_distances(descriptors, descriptors)?;
    let mut fitted = decaf_from_dissimilarity(distances.view(), options)?;
    fitted.descriptors = Some(descriptors.to_owned());
    Ok(fitted)
}

/// Fit DECAF to a symmetric dissimilarity matrix.
pub fn decaf_from_dissimilarity(
    distances: ArrayView2<f64>,
    options: &DecafOptions,
) -> Result<Decaf> {
    validate_options(options)?;
    if distances.nrows() == 0 || distances.ncols() != distances.nrows() {
        return Err(LandfoldError::Shape(
            "DECAF dissimilarity matrix must be non-empty and square",
        ));
    }
    validate_dissimilarity(distances)?;
    let n = distances.nrows();
    let max_distance = distances.iter().copied().fold(0.0_f64, f64::max);
    if max_distance == 0.0 {
        return Ok(Decaf {
            descriptors: None,
            mds: None,
            embedded: Array2::zeros((n, 0)),
            centers: Array2::zeros((1, 0)),
            labels: vec![0; n],
            memberships: Array2::ones((n, 1)),
            representatives: vec![0],
            bandwidth: match options.bandwidth {
                BandwidthChoice::Fixed(value) => value,
                BandwidthChoice::Paper { sigma } => sigma,
            },
        });
    }
    let squared_scale = max_distance * max_distance;
    if squared_scale == 0.0 || squared_scale.is_subnormal() {
        return Err(LandfoldError::Msg(
            "DECAF dissimilarity scale is too small for squared-distance MDS".into(),
        ));
    }

    let dimensions = match options.dimensions {
        DimensionChoice::BrokenStick => MdsDimension::BrokenStick,
        DimensionChoice::Fixed(value) => MdsDimension::Fixed(value),
    };
    let mds = ClassicalMdsModel::fit(distances, dimensions)?;
    let embedded = mds.embedding().clone();
    let bandwidth = match options.bandwidth {
        BandwidthChoice::Fixed(value) => value,
        BandwidthChoice::Paper { sigma } => paper_bandwidth(embedded.view(), sigma)?,
    };
    validate_effective_bandwidth(bandwidth)?;
    let centers = mean_shift(
        embedded.view(),
        bandwidth,
        options.max_iterations,
        options.tolerance,
    )?;
    let labels = nearest_labels(embedded.view(), centers.view());
    let memberships = fuzzy_memberships(embedded.view(), centers.view(), bandwidth)?;
    let representatives = medoid_representatives(embedded.view(), &labels, centers.nrows())?;
    Ok(Decaf {
        descriptors: None,
        mds: Some(mds),
        embedded,
        centers,
        labels,
        memberships,
        representatives,
        bandwidth,
    })
}

fn validate_dissimilarity(distances: ArrayView2<f64>) -> Result<()> {
    if distances
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(LandfoldError::Msg(
            "DECAF dissimilarities must be finite and nonnegative".into(),
        ));
    }
    for i in 0..distances.nrows() {
        if distances[(i, i)] != 0.0 {
            return Err(LandfoldError::Msg(
                "DECAF dissimilarity diagonal must be zero".into(),
            ));
        }
        for j in 0..i {
            let left = distances[(i, j)];
            let right = distances[(j, i)];
            let scale = left.abs().max(right.abs()).max(1.0);
            if (left - right).abs() > 64.0 * f64::EPSILON * scale {
                return Err(LandfoldError::Msg(
                    "DECAF dissimilarities must be symmetric".into(),
                ));
            }
        }
    }
    Ok(())
}

fn validate_options(options: &DecafOptions) -> Result<()> {
    if matches!(options.dimensions, DimensionChoice::Fixed(0)) {
        return Err(LandfoldError::LowDim { low: 0, high: 0 });
    }
    let bandwidth_value = match options.bandwidth {
        BandwidthChoice::Paper { sigma } => sigma,
        BandwidthChoice::Fixed(value) => value,
    };
    if !bandwidth_value.is_finite() || bandwidth_value <= 0.0 {
        return Err(LandfoldError::Msg(
            "DECAF bandwidth or smearing width must be finite and positive".into(),
        ));
    }
    if matches!(options.bandwidth, BandwidthChoice::Fixed(_))
        && !(bandwidth_value * bandwidth_value).is_normal()
    {
        return Err(LandfoldError::Msg(
            "DECAF fixed bandwidth square must be a normal positive finite value".into(),
        ));
    }
    if options.max_iterations == 0 || !options.tolerance.is_finite() || options.tolerance <= 0.0 {
        return Err(LandfoldError::Msg(
            "DECAF mean-shift controls must be positive".into(),
        ));
    }
    Ok(())
}

fn validate_effective_bandwidth(bandwidth: f64) -> Result<()> {
    if !bandwidth.is_finite() || bandwidth <= 0.0 || !(bandwidth * bandwidth).is_normal() {
        return Err(LandfoldError::Msg(
            "DECAF mean-shift bandwidth square must be a normal positive finite value".into(),
        ));
    }
    Ok(())
}

fn cross_distances(left: ArrayView2<f64>, right: ArrayView2<f64>) -> Result<Array2<f64>> {
    if left.ncols() != right.ncols() {
        return Err(LandfoldError::Shape(
            "descriptor sets must have the same feature width",
        ));
    }
    let mut distances = Array2::zeros((left.nrows(), right.nrows()));
    for i in 0..left.nrows() {
        for j in 0..right.nrows() {
            let mut distance = 0.0_f64;
            for feature in 0..left.ncols() {
                let delta = left[(i, feature)] - right[(j, feature)];
                if !delta.is_finite() {
                    return Err(LandfoldError::Msg(
                        "DECAF descriptor difference overflowed".into(),
                    ));
                }
                distance = distance.hypot(delta);
            }
            if !distance.is_finite() {
                return Err(LandfoldError::Msg(
                    "DECAF descriptor distance overflowed".into(),
                ));
            }
            distances[(i, j)] = distance;
        }
    }
    Ok(distances)
}

fn squared_distance(
    left: ndarray::ArrayView1<'_, f64>,
    right: ndarray::ArrayView1<'_, f64>,
) -> f64 {
    left.iter().zip(right).map(|(a, b)| (a - b).powi(2)).sum()
}

fn pair_distances(points: ArrayView2<f64>) -> Vec<f64> {
    let mut values = Vec::with_capacity(points.nrows().saturating_mul(points.nrows()) / 2);
    for i in 0..points.nrows() {
        for j in 0..i {
            values.push(squared_distance(points.row(i), points.row(j)).sqrt());
        }
    }
    values
}

/// The reference Gaussian rule needs at least 21 bins to resolve a gap.
fn bandwidth_grid_is_too_coarse(bins: usize) -> bool {
    bins < 21
}

fn first_density_minimum(density: &[f64]) -> Option<(usize, usize)> {
    let mut index = 1;
    while index + 1 < density.len() {
        if density[index] < density[index - 1] {
            let mut plateau_end = index;
            while plateau_end + 1 < density.len() && density[plateau_end + 1] == density[index] {
                plateau_end += 1;
            }
            if plateau_end + 1 < density.len() && density[plateau_end] < density[plateau_end + 1] {
                return Some((index, plateau_end));
            }
            index = plateau_end;
        }
        index += 1;
    }
    None
}

fn gaussian_kernel_cutoff(sigma: f64) -> f64 {
    6.0 * sigma
}

fn kernel_window(distances: &[f64], lower: f64, upper: f64) -> std::ops::Range<usize> {
    let start = distances.partition_point(|distance| *distance < lower);
    let end = distances.partition_point(|distance| *distance <= upper);
    start..end
}

fn paper_bandwidth(points: ArrayView2<f64>, sigma: f64) -> Result<f64> {
    let mut distances = pair_distances(points);
    distances.sort_by(f64::total_cmp);
    let max_distance = distances.iter().copied().fold(0.0_f64, f64::max);
    if max_distance == 0.0 {
        return Ok(sigma);
    }
    let bin_width = sigma / 21.0;
    if !bin_width.is_finite() || bin_width <= 0.0 {
        return Err(LandfoldError::Msg(
            "DECAF bandwidth grid spacing is not representable".into(),
        ));
    }
    let requested_bins = (max_distance / bin_width).ceil();
    if !requested_bins.is_finite() || requested_bins > MAX_BANDWIDTH_BINS as f64 {
        return Err(LandfoldError::Msg(format!(
            "DECAF bandwidth grid requires more than {MAX_BANDWIDTH_BINS} bins"
        )));
    }
    let bins = requested_bins as usize;
    if bandwidth_grid_is_too_coarse(bins) {
        return Ok(max_distance);
    }
    let step = (bin_width * bins as f64) / (bins - 1) as f64;
    let mut density = vec![0.0; bins];
    // Six standard deviations retain each Gaussian down to exp(-18) while
    // avoiding evaluations whose contribution cannot affect the first
    // histogram minimum at the fixture precision.
    let kernel_cutoff = gaussian_kernel_cutoff(sigma);
    for (index, value) in density.iter_mut().enumerate() {
        let x = index as f64 * step;
        let lower = x - kernel_cutoff;
        let upper = x + kernel_cutoff;
        *value = distances[kernel_window(&distances, lower, upper)]
            .iter()
            .map(|distance| (-0.5 * ((x - distance) / sigma).powi(2)).exp())
            .sum();
    }
    if let Some(bandwidth) = separating_valley(&density, &distances, step, bin_width) {
        return Ok(bandwidth);
    }
    // A single shell, including one pair, has its mode at the diameter. The
    // kernel cutoff is the same width the density sum already uses.
    let peak = density.iter().copied().fold(0.0_f64, f64::max);
    let mode_index = density.iter().position(|value| *value == peak).unwrap_or(0);
    let mode = mode_index as f64 * step;
    if max_distance - mode <= gaussian_kernel_cutoff(sigma) {
        return Ok(max_distance);
    }
    Ok(median_positive_nearest_neighbor(points, max_distance))
}

/// First density minimum that separates two modes with mass on both sides.
fn separating_valley(density: &[f64], distances: &[f64], step: f64, bin_width: f64) -> Option<f64> {
    let mut remaining = density.to_vec();
    while let Some((index, plateau_end)) = first_density_minimum(&remaining) {
        let left_peak = density[..=index].iter().copied().fold(0.0, f64::max);
        let right_peak = density[plateau_end + 1..]
            .iter()
            .copied()
            .fold(0.0, f64::max);
        let minimum = (index + plateau_end) as f64 * 0.5;
        let bandwidth = minimum * step + 0.5 * bin_width;
        let beyond = distances
            .iter()
            .filter(|distance| **distance > bandwidth)
            .count();
        let far_fraction = beyond as f64 / distances.len() as f64;
        if left_peak > 0.0
            && density[index] <= VALLEY_MAX_DEPTH * left_peak
            && right_peak >= VALLEY_MIN_RECOVERY * left_peak
            && far_fraction >= VALLEY_MIN_FAR_FRACTION
        {
            return Some(bandwidth);
        }
        let shoulder = remaining[index - 1];
        for value in &mut remaining[index..=plateau_end] {
            *value = shoulder;
        }
    }
    None
}

fn median_positive_nearest_neighbor(points: ArrayView2<f64>, max_distance: f64) -> f64 {
    let n = points.nrows();
    let mut nearest = Vec::with_capacity(n);
    for i in 0..n {
        let mut best = f64::INFINITY;
        for j in 0..n {
            if i == j {
                continue;
            }
            let distance = squared_distance(points.row(i), points.row(j)).sqrt();
            if distance > 0.0 && distance < best {
                best = distance;
            }
        }
        if best.is_finite() {
            nearest.push(best);
        }
    }
    if nearest.is_empty() {
        return max_distance;
    }
    nearest.sort_by(f64::total_cmp);
    nearest[(nearest.len() - 1) / 2]
}

fn convergence_threshold_sq(tolerance: f64, bandwidth: f64) -> f64 {
    (tolerance * bandwidth).powi(2)
}

fn has_converged(shift: f64, convergence_sq: f64) -> bool {
    shift <= convergence_sq
}

fn centers_are_distinct(distance: f64, radius_sq: f64) -> bool {
    distance > radius_sq
}

fn mean_shift(
    points: ArrayView2<f64>,
    bandwidth: f64,
    max_iterations: usize,
    tolerance: f64,
) -> Result<Array2<f64>> {
    let radius_sq = bandwidth.powi(2);
    let convergence_sq = convergence_threshold_sq(tolerance, bandwidth);
    let mut candidates: Vec<(usize, Vec<f64>)> = Vec::with_capacity(points.nrows());
    for seed in points.rows() {
        let mut centre = seed.to_vec();
        let mut support = 0;
        for _ in 0..max_iterations {
            let mut next = vec![0.0; points.ncols()];
            support = 0;
            for point in points.rows() {
                if point
                    .iter()
                    .zip(&centre)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    <= radius_sq
                {
                    support += 1;
                    for axis in 0..points.ncols() {
                        next[axis] += point[axis];
                    }
                }
            }
            if support == 0 {
                break;
            }
            for value in &mut next {
                *value /= support as f64;
            }
            let shift = squared_distance(
                ndarray::ArrayView1::from(next.as_slice()),
                ndarray::ArrayView1::from(centre.as_slice()),
            );
            centre = next;
            if has_converged(shift, convergence_sq) {
                break;
            }
        }
        candidates.push((support, centre));
    }
    candidates.sort_by(|a, b| {
        b.0.cmp(&a.0).then_with(|| {
            a.1.iter()
                .zip(&b.1)
                .find_map(|(left, right)| {
                    let ordering = left.total_cmp(right);
                    ordering.is_ne().then_some(ordering)
                })
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });
    let mut accepted: Vec<Vec<f64>> = Vec::new();
    for (_, candidate) in candidates {
        if accepted.iter().all(|centre| {
            let distance = squared_distance(
                ndarray::ArrayView1::from(centre.as_slice()),
                ndarray::ArrayView1::from(candidate.as_slice()),
            );
            centers_are_distinct(distance, radius_sq)
        }) {
            accepted.push(candidate);
        }
    }
    Array2::from_shape_vec(
        (accepted.len(), points.ncols()),
        accepted.into_iter().flatten().collect(),
    )
    .map_err(|_| LandfoldError::Shape("mean-shift center matrix"))
}

fn nearest_labels(points: ArrayView2<f64>, centers: ArrayView2<f64>) -> Vec<usize> {
    points
        .rows()
        .into_iter()
        .map(|point| {
            centers
                .rows()
                .into_iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| {
                    squared_distance(point, *left).total_cmp(&squared_distance(point, *right))
                })
                .map_or(0, |(index, _)| index)
        })
        .collect()
}

fn fuzzy_memberships(
    points: ArrayView2<f64>,
    centers: ArrayView2<f64>,
    bandwidth: f64,
) -> Result<Array2<f64>> {
    let mut memberships = Array2::zeros((points.nrows(), centers.nrows()));
    for i in 0..points.nrows() {
        let mut total = 0.0;
        for class in 0..centers.nrows() {
            let scaled_distance = points
                .row(i)
                .iter()
                .zip(centers.row(class))
                .fold(0.0_f64, |norm, (point, center)| {
                    norm.hypot((point - center) / bandwidth)
                });
            let weight = (-0.5 * scaled_distance.powi(2)).exp();
            memberships[(i, class)] = weight;
            total += weight;
        }
        if total == 0.0 {
            memberships[(
                i,
                nearest_labels(points.slice(ndarray::s![i..=i, ..]), centers)[0],
            )] = 1.0;
        } else {
            for class in 0..centers.nrows() {
                memberships[(i, class)] /= total;
            }
        }
    }
    if memberships.iter().any(|value| !value.is_finite()) {
        return Err(LandfoldError::Msg(
            "DECAF fuzzy memberships must be finite".into(),
        ));
    }
    Ok(memberships)
}

fn medoid_representatives(
    points: ArrayView2<f64>,
    labels: &[usize],
    class_count: usize,
) -> Result<Vec<usize>> {
    if labels.len() != points.nrows() {
        return Err(LandfoldError::Shape(
            "DECAF labels must have one entry per embedded point",
        ));
    }
    if let Some(label) = labels.iter().copied().find(|label| *label >= class_count) {
        return Err(LandfoldError::Msg(format!(
            "DECAF class label {label} is outside {class_count} centers"
        )));
    }
    (0..class_count)
        .map(|class| {
            labels
                .iter()
                .enumerate()
                .filter(|(_, label)| **label == class)
                .min_by(|(left, _), (right, _)| {
                    let left_sum = labels
                        .iter()
                        .enumerate()
                        .filter(|(_, label)| **label == class)
                        .map(|(other, _)| {
                            squared_distance(points.row(*left), points.row(other)).sqrt()
                        })
                        .sum::<f64>();
                    let right_sum = labels
                        .iter()
                        .enumerate()
                        .filter(|(_, label)| **label == class)
                        .map(|(other, _)| {
                            squared_distance(points.row(*right), points.row(other)).sqrt()
                        })
                        .sum::<f64>();
                    left_sum.total_cmp(&right_sum).then_with(|| left.cmp(right))
                })
                .map(|(index, _)| index)
                .ok_or_else(|| {
                    LandfoldError::Msg(format!("DECAF class {class} has no observations"))
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn rejects_each_invalid_shape_and_option() {
        let empty = Array2::zeros((0, 0));
        let empty_error =
            decaf_from_dissimilarity(empty.view(), &DecafOptions::default()).unwrap_err();
        assert!(matches!(
            empty_error,
            LandfoldError::Shape("DECAF dissimilarity matrix must be non-empty and square")
        ));

        let nonsquare = array![[0.0, 1.0, 2.0], [1.0, 0.0, 1.0]];
        let shape_error =
            decaf_from_dissimilarity(nonsquare.view(), &DecafOptions::default()).unwrap_err();
        assert!(matches!(
            shape_error,
            LandfoldError::Shape("DECAF dissimilarity matrix must be non-empty and square")
        ));

        let distances = array![[0.0, 1.0], [1.0, 0.0]];
        for options in [
            DecafOptions {
                dimensions: DimensionChoice::Fixed(0),
                ..DecafOptions::default()
            },
            DecafOptions {
                bandwidth: BandwidthChoice::Fixed(f64::NAN),
                ..DecafOptions::default()
            },
            DecafOptions {
                bandwidth: BandwidthChoice::Fixed(0.0),
                ..DecafOptions::default()
            },
            DecafOptions {
                max_iterations: 0,
                ..DecafOptions::default()
            },
            DecafOptions {
                tolerance: f64::NAN,
                ..DecafOptions::default()
            },
            DecafOptions {
                tolerance: 0.0,
                ..DecafOptions::default()
            },
        ] {
            assert!(decaf_from_dissimilarity(distances.view(), &options).is_err());
        }
    }

    fn fixed_bandwidth_options(value: f64) -> DecafOptions {
        DecafOptions {
            dimensions: DimensionChoice::Fixed(1),
            bandwidth: BandwidthChoice::Fixed(value),
            ..DecafOptions::default()
        }
    }

    fn assert_ok_fit_has_no_nan(result: &Result<Decaf>) {
        if let Ok(fitted) = result {
            assert!(!fitted.bandwidth().is_nan());
            assert!(fitted.embedded().iter().all(|value| !value.is_nan()));
            assert!(fitted.centers().iter().all(|value| !value.is_nan()));
            assert!(fitted.memberships().iter().all(|value| !value.is_nan()));
        }
    }

    #[test]
    fn fixed_bandwidth_rejects_an_underflowing_square() {
        let distances = array![[0.0, 1.0], [1.0, 0.0]];
        let error = decaf_from_dissimilarity(distances.view(), &fixed_bandwidth_options(1.0e-170))
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "DECAF fixed bandwidth square must be a normal positive finite value"
        );
    }

    #[test]
    fn tiny_valid_fixed_bandwidth_has_finite_normalized_memberships() {
        let distances = array![[0.0, 1.0], [1.0, 0.0]];
        let bandwidth = 2.0 * f64::MIN_POSITIVE.sqrt();
        assert!(bandwidth.powi(2).is_normal());
        let fitted =
            decaf_from_dissimilarity(distances.view(), &fixed_bandwidth_options(bandwidth))
                .unwrap();

        for memberships in fitted.memberships().rows() {
            assert!(memberships.iter().all(|value| value.is_finite()));
            assert!((memberships.sum() - 1.0).abs() < 1.0e-15);
        }
    }

    #[test]
    fn a_successful_fit_never_contains_nan_values() {
        let distances = array![[0.0, 1.0], [1.0, 0.0]];
        for bandwidth in [1.0e-170, 2.0 * f64::MIN_POSITIVE.sqrt(), 0.5] {
            let result =
                decaf_from_dissimilarity(distances.view(), &fixed_bandwidth_options(bandwidth));
            assert_ok_fit_has_no_nan(&result);
        }
    }

    #[test]
    fn rejects_a_dissimilarity_scale_too_small_for_squared_mds() {
        let distances = array![[0.0, 1.0e-162], [1.0e-162, 0.0]];
        let error =
            decaf_from_dissimilarity(distances.view(), &fixed_bandwidth_options(0.5)).unwrap_err();
        assert_eq!(
            error.to_string(),
            "DECAF dissimilarity scale is too small for squared-distance MDS"
        );
    }

    #[test]
    fn nonzero_dissimilarity_does_not_take_the_uniform_shortcut() {
        let distances = array![[0.0, 1.0], [1.0, 0.0]];
        let fitted = decaf_from_dissimilarity(
            distances.view(),
            &DecafOptions {
                dimensions: DimensionChoice::Fixed(1),
                bandwidth: BandwidthChoice::Fixed(0.5),
                ..DecafOptions::default()
            },
        )
        .unwrap();
        assert_eq!(fitted.embedded().ncols(), 1);
        assert!(fitted.mds().is_some());
    }

    #[test]
    fn oversized_bandwidth_grids_return_errors_instead_of_panicking() {
        let huge = array![[0.0, 1.0e20], [1.0e20, 0.0]];
        let huge_error = decaf_from_dissimilarity(
            huge.view(),
            &DecafOptions {
                dimensions: DimensionChoice::Fixed(1),
                bandwidth: BandwidthChoice::Paper { sigma: 0.0092 },
                ..DecafOptions::default()
            },
        )
        .unwrap_err();
        assert!(huge_error.to_string().contains("bandwidth grid"));

        let ordinary = array![[0.0, 1.0], [1.0, 0.0]];
        let tiny_sigma_error = decaf_from_dissimilarity(
            ordinary.view(),
            &DecafOptions {
                dimensions: DimensionChoice::Fixed(1),
                bandwidth: BandwidthChoice::Paper {
                    sigma: f64::MIN_POSITIVE,
                },
                ..DecafOptions::default()
            },
        )
        .unwrap_err();
        assert!(tiny_sigma_error.to_string().contains("bandwidth grid"));

        let subnormal_sigma_error = decaf_from_dissimilarity(
            ordinary.view(),
            &DecafOptions {
                dimensions: DimensionChoice::Fixed(1),
                bandwidth: BandwidthChoice::Paper {
                    sigma: f64::from_bits(1),
                },
                ..DecafOptions::default()
            },
        )
        .unwrap_err();
        assert_eq!(
            subnormal_sigma_error.to_string(),
            "DECAF bandwidth grid spacing is not representable"
        );
    }

    #[test]
    fn maximum_bandwidth_grid_size_is_inclusive() {
        let points = array![[0.0], [MAX_BANDWIDTH_BINS as f64]];
        assert!(paper_bandwidth(points.view(), 21.0).is_ok());
    }

    #[test]
    fn descriptor_distances_use_a_scaled_norm_and_report_true_overflow() {
        let large = array![[1.0e200, 1.0e200]];
        let origin = array![[0.0, 0.0]];
        let distances = cross_distances(large.view(), origin.view()).unwrap();
        assert!(distances[(0, 0)].is_finite());
        assert!((distances[(0, 0)] / 1.0e200 - 2.0_f64.sqrt()).abs() < 1.0e-15);

        let positive = array![[f64::MAX]];
        let negative = array![[-f64::MAX]];
        assert!(cross_distances(positive.view(), negative.view()).is_err());
    }

    #[test]
    fn bandwidth_of_a_zero_distance_set_is_the_smearing_width() {
        let sigma = 0.0092;
        let uniform = Array2::zeros((3, 2));
        assert_eq!(paper_bandwidth(uniform.view(), sigma).unwrap(), sigma);

        let separated = array![[0.0], [0.1], [2.0], [2.1]];
        let valley = paper_bandwidth(separated.view(), 0.05).unwrap();
        assert!(valley > 0.9 && valley < 1.1, "valley={valley}");

        let one_pair = array![[0.0], [1.0]];
        assert_eq!(paper_bandwidth(one_pair.view(), 0.05).unwrap(), 1.0);
    }

    #[test]
    fn bandwidth_without_a_minimum_falls_back_after_a_full_coarse_grid() {
        let sigma = 0.05_f64;
        let points = array![[0.0], [1.0]];
        let bins = (1.0 / (sigma / 21.0)).ceil() as usize;
        assert!(bins >= 21);
        assert_eq!(paper_bandwidth(points.view(), sigma).unwrap(), 1.0);
    }

    #[test]
    fn equally_spaced_line_keeps_the_first_shell_gap() {
        let points = Array2::from_shape_fn((40, 1), |(index, _)| index as f64);
        let bandwidth = paper_bandwidth(points.view(), 0.0092).unwrap();
        assert!((1.0..2.0).contains(&bandwidth), "bandwidth={bandwidth}");
    }

    #[test]
    fn irregular_line_uses_nearest_neighbor_scale_not_the_diameter() {
        let points = Array2::from_shape_fn((40, 1), |(index, _)| {
            let step = index as f64;
            step + 0.37 * (step * 1.6180339887).sin()
        });
        let bandwidth = paper_bandwidth(points.view(), 0.0092).unwrap();
        let diameter = (points[(39, 0)] - points[(0, 0)]).abs();
        assert!(
            bandwidth < 0.25 * diameter,
            "bandwidth={bandwidth} diameter={diameter}"
        );
    }

    #[test]
    fn bandwidth_grid_is_coarse_below_twenty_one_bins() {
        assert!(bandwidth_grid_is_too_coarse(20));
        assert!(!bandwidth_grid_is_too_coarse(21));
    }

    #[test]
    fn density_minimum_requires_a_strict_left_drop_and_accepts_a_right_tie() {
        assert_eq!(first_density_minimum(&[2.0, 1.0, 1.0, 2.0]), Some((1, 2)));
        assert_eq!(first_density_minimum(&[1.0, 1.0, 2.0]), None);
        assert_eq!(first_density_minimum(&[3.0, 2.0, 1.0, 2.0]), Some((2, 2)));
        assert_eq!(first_density_minimum(&[3.0, 2.0, 2.0]), None);
        assert_eq!(first_density_minimum(&[3.0, 2.0, 1.0]), None);
    }

    #[test]
    fn truncated_kernel_window_includes_both_cutoff_boundaries() {
        let distances = [0.0, 1.0, 2.0, 3.0];
        assert_eq!(gaussian_kernel_cutoff(0.5), 3.0);
        assert_eq!(kernel_window(&distances, 1.0, 2.0), 1..3);
    }

    #[test]
    fn centers_at_exactly_one_bandwidth_are_not_distinct() {
        assert!(!centers_are_distinct(0.04, 0.04));
        assert!(centers_are_distinct(
            f64::from_bits(0.04_f64.to_bits() + 1),
            0.04
        ));
    }

    #[test]
    fn convergence_threshold_is_inclusive_and_uses_tolerance_times_bandwidth() {
        let threshold = convergence_threshold_sq(0.25, 0.4);
        assert!((threshold - 0.01).abs() < 1.0e-15);
        assert!(has_converged(threshold, threshold));
        assert!(!has_converged(
            f64::from_bits(threshold.to_bits() + 1),
            threshold
        ));
    }

    #[test]
    fn mean_shift_stops_as_soon_as_the_tolerance_is_met() {
        let points = array![[0.0], [0.1], [0.2], [0.4]];
        let stopped = mean_shift(points.view(), 0.2, 20, 1.0).unwrap();
        let one_iteration = mean_shift(points.view(), 0.2, 1, 1.0e-12).unwrap();
        let fully_converged = mean_shift(points.view(), 0.2, 20, 1.0e-12).unwrap();

        assert_eq!(stopped, one_iteration);
        assert_ne!(stopped, fully_converged);
        assert!((stopped[(0, 0)] - 0.175).abs() < 1.0e-15);
        assert!((fully_converged[(0, 0)] - 0.1).abs() < 1.0e-15);
    }

    #[test]
    fn flat_kernel_mean_shift_converges_to_local_arithmetic_means() {
        let points = array![[0.0], [0.2], [0.8], [1.0]];
        let centers = mean_shift(points.view(), 0.3, 20, 1.0e-12).unwrap();
        assert_eq!(centers.nrows(), 2);
        assert!((centers[(0, 0)] - 0.1).abs() < 1.0e-12);
        assert!((centers[(1, 0)] - 0.9).abs() < 1.0e-12);

        let chained = array![[0.0], [0.2], [0.35]];
        let converged = mean_shift(chained.view(), 0.25, 20, 1.0e-12).unwrap();
        assert_eq!(converged.nrows(), 1);
        assert!((converged[(0, 0)] - 0.183_333_333_333_333_35).abs() < 1.0e-12);
    }

    #[test]
    fn mean_shift_uses_the_squared_bandwidth_as_its_flat_kernel_radius() {
        let points = array![[0.0], [0.1], [0.5], [0.6]];
        let centers = mean_shift(points.view(), 0.2, 20, 1.0e-12).unwrap();
        assert_eq!(centers, array![[0.05], [0.55]]);
    }

    #[test]
    fn published_pd211_embedding_matches_reference_bandwidth_and_partition() {
        let coordinates: Vec<f64> = include_str!("../tests/goldens/decaf_pd211_embedded.txt")
            .split_whitespace()
            .map(|value| value.parse().unwrap())
            .collect();
        let points = Array2::from_shape_vec((coordinates.len(), 1), coordinates).unwrap();
        let reference: Vec<usize> = include_str!("../tests/goldens/decaf_pd211_labels.txt")
            .split_whitespace()
            .map(|value| value.parse().unwrap())
            .collect();

        let bandwidth = paper_bandwidth(points.view(), 0.0092).unwrap();
        assert!((bandwidth - 0.029_146_398_984_356_506).abs() < 1.0e-15);
        let centers = mean_shift(points.view(), bandwidth, 800, 1.0e-6).unwrap();
        let labels = nearest_labels(points.view(), centers.view());
        for i in 0..labels.len() {
            for j in 0..labels.len() {
                assert_eq!(labels[i] == labels[j], reference[i] == reference[j]);
            }
        }
    }

    #[test]
    fn fuzzy_weights_and_underflow_fallback_are_exact() {
        let points = array![[0.0]];
        let centers = array![[0.0], [1.0]];
        let memberships = fuzzy_memberships(points.view(), centers.view(), 2.0).unwrap();
        let distant_weight = (-0.125_f64).exp();
        assert!((memberships[(0, 0)] - 1.0 / (1.0 + distant_weight)).abs() < 1.0e-15);
        assert!((memberships[(0, 1)] - distant_weight / (1.0 + distant_weight)).abs() < 1.0e-15);

        let far_centers = array![[1.0e200], [2.0e200]];
        let fallback = fuzzy_memberships(points.view(), far_centers.view(), 1.0).unwrap();
        assert_eq!(fallback, array![[1.0, 0.0]]);
    }

    #[test]
    fn representatives_are_within_class_medoids() {
        let points = array![[0.0], [0.2], [2.0], [5.0]];
        assert_eq!(
            medoid_representatives(points.view(), &[0, 0, 0, 1], 2).unwrap(),
            vec![1, 3]
        );
    }

    #[test]
    fn representatives_reject_empty_and_out_of_range_classes() {
        let points = array![[0.0], [2.0]];
        let empty = medoid_representatives(points.view(), &[0, 2], 3).unwrap_err();
        assert!(empty.to_string().contains("class 1 has no observations"));

        let out_of_range = medoid_representatives(points.view(), &[0, 2], 2).unwrap_err();
        assert!(out_of_range.to_string().contains("outside 2 centers"));
    }
}
