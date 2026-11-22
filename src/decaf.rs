//! Descriptor Embedding and Clustering for Atomistic-environment Framework.
//!
//! This module is descriptor-agnostic. It implements the classical MDS,
//! broken-stick, bandwidth, and mean-shift stages of Lai et al., J. Chem.
//! Phys. 159, 024129 (2023), DOI 10.1063/5.0160369. Callers provide either
//! descriptor rows or a symmetric dissimilarity matrix.

use ndarray::{Array2, ArrayView2};

use crate::error::{LandfoldError, Result};
use crate::mds::{ClassicalMdsModel, MdsDimension};

/// MDS dimension selection for DECAF.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DimensionChoice {
    /// Eq. 5 broken-stick selection.
    #[default]
    BrokenStick,
    /// Explicit embedding dimension.
    Fixed(usize),
}

/// Mean-shift bandwidth selection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BandwidthChoice {
    /// First minimum of the Gaussian-smeared pair-distance distribution from
    /// the paper, using the supplied smearing width.
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
    let centers = mean_shift(
        embedded.view(),
        bandwidth,
        options.max_iterations,
        options.tolerance,
    )?;
    let labels = nearest_labels(embedded.view(), centers.view());
    let memberships = fuzzy_memberships(embedded.view(), centers.view(), bandwidth);
    let representatives = medoid_representatives(embedded.view(), centers.view(), &labels);
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
    if options.max_iterations == 0 || !options.tolerance.is_finite() || options.tolerance <= 0.0 {
        return Err(LandfoldError::Msg(
            "DECAF mean-shift controls must be positive".into(),
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
            let mut squared = 0.0;
            for feature in 0..left.ncols() {
                let delta = left[(i, feature)] - right[(j, feature)];
                squared += delta * delta;
            }
            distances[(i, j)] = squared.sqrt();
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

fn paper_bandwidth(points: ArrayView2<f64>, sigma: f64) -> Result<f64> {
    let distances = pair_distances(points);
    let max_distance = distances.iter().copied().fold(0.0_f64, f64::max);
    if max_distance == 0.0 {
        return Ok(sigma);
    }
    let bin_width = sigma / 21.0;
    let bins = (max_distance / bin_width).ceil() as usize;
    if bins < 3 {
        return Ok(max_distance);
    }
    let step = (bin_width * bins as f64) / (bins - 1) as f64;
    let mut density = vec![0.0; bins];
    for (index, value) in density.iter_mut().enumerate() {
        let x = index as f64 * step;
        *value = distances
            .iter()
            .map(|distance| (-0.5 * ((x - distance) / sigma).powi(2)).exp())
            .sum();
    }
    for index in 1..(bins - 1) {
        if density[index] < density[index - 1] && density[index] <= density[index + 1] {
            return Ok((index as f64 + 0.5) * bin_width);
        }
    }
    Ok(max_distance)
}

fn mean_shift(
    points: ArrayView2<f64>,
    bandwidth: f64,
    max_iterations: usize,
    tolerance: f64,
) -> Result<Array2<f64>> {
    let radius_sq = bandwidth * bandwidth;
    let convergence_sq = (tolerance * bandwidth).powi(2);
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
            let shift: f64 = next.iter().zip(&centre).map(|(a, b)| (a - b).powi(2)).sum();
            centre = next;
            if shift <= convergence_sq {
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
            centre
                .iter()
                .zip(&candidate)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                > radius_sq
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
) -> Array2<f64> {
    let mut memberships = Array2::zeros((points.nrows(), centers.nrows()));
    for i in 0..points.nrows() {
        let mut total = 0.0;
        for class in 0..centers.nrows() {
            let weight = (-0.5 * squared_distance(points.row(i), centers.row(class))
                / bandwidth.powi(2))
            .exp();
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
    memberships
}

fn medoid_representatives(
    points: ArrayView2<f64>,
    centers: ArrayView2<f64>,
    labels: &[usize],
) -> Vec<usize> {
    (0..centers.nrows())
        .map(|class| {
            labels
                .iter()
                .enumerate()
                .filter(|(_, label)| **label == class)
                .min_by(|(left, _), (right, _)| {
                    squared_distance(points.row(*left), centers.row(class))
                        .total_cmp(&squared_distance(points.row(*right), centers.row(class)))
                        .then_with(|| left.cmp(right))
                })
                .map_or(0, |(index, _)| index)
        })
        .collect()
}
