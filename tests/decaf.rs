use landfold::{
    BandwidthChoice, DecafOptions, DimensionChoice, broken_stick_dimension, decaf_from_descriptors,
};
use ndarray::{Array2, array};

fn parse_matrix(text: &str) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            line.split_whitespace()
                .map(|value| value.parse().unwrap())
                .collect()
        })
        .collect();
    let width = rows[0].len();
    Array2::from_shape_vec((rows.len(), width), rows.into_iter().flatten().collect()).unwrap()
}

fn same_partition(labels: &[usize]) -> Vec<Vec<bool>> {
    labels
        .iter()
        .map(|left| labels.iter().map(|right| left == right).collect())
        .collect()
}

#[test]
fn broken_stick_matches_hand_computed_four_eigenvalue_example() {
    // Normalized eigenvalues are [0.6, 0.3, 0.1, 0.0]. The four-segment
    // broken-stick expectations are [25/48, 13/48, 7/48, 3/48], so exactly
    // the first two components survive Eq. 5.
    assert_eq!(broken_stick_dimension(&[6.0, 3.0, 1.0, 0.0]), 2);

    // Eq. 5 counts every component above its corresponding broken-stick
    // expectation; it does not stop at the first component below expectation.
    assert_eq!(broken_stick_dimension(&[0.55, 0.20, 0.18, 0.07]), 3);
}

#[test]
fn decaf_is_permutation_invariant_and_returns_normalized_memberships() {
    let points = array![
        [0.0, 0.0],
        [0.0, 0.1],
        [4.0, 4.0],
        [4.0, 4.1],
        [9.0, 0.0],
        [9.0, 0.1],
    ];
    let options = DecafOptions {
        dimensions: DimensionChoice::Fixed(2),
        bandwidth: BandwidthChoice::Fixed(0.5),
        ..DecafOptions::default()
    };
    let fitted = decaf_from_descriptors(points.view(), &options).unwrap();
    assert_eq!(fitted.n_classes(), 3);
    assert_eq!(fitted.representatives().len(), 3);
    for row in fitted.memberships().rows() {
        assert!((row.sum() - 1.0).abs() < 1.0e-12);
    }

    let permutation = [4usize, 2, 0, 5, 3, 1];
    let permuted = Array2::from_shape_fn(points.raw_dim(), |(i, j)| points[(permutation[i], j)]);
    let fitted_permuted = decaf_from_descriptors(permuted.view(), &options).unwrap();
    let original_partition = same_partition(fitted.labels());
    let permuted_partition = same_partition(fitted_permuted.labels());
    for i in 0..permutation.len() {
        for j in 0..permutation.len() {
            assert_eq!(
                original_partition[permutation[i]][permutation[j]],
                permuted_partition[i][j]
            );
        }
    }
}

#[test]
fn perfect_crystal_is_one_class_and_vacancy_shells_are_distinct() {
    let crystal = Array2::from_elem((32, 4), 0.25);
    let options = DecafOptions {
        dimensions: DimensionChoice::BrokenStick,
        bandwidth: BandwidthChoice::Fixed(0.1),
        ..DecafOptions::default()
    };
    let perfect = decaf_from_descriptors(crystal.view(), &options).unwrap();
    assert_eq!(perfect.n_classes(), 1);
    assert!(perfect.labels().iter().all(|&label| label == 0));

    // Synthetic descriptor-space analogue of bulk, first vacancy shell, and
    // second vacancy shell. Multiplicities mirror a crystalline majority and
    // two coordination shells; DECAF itself remains descriptor-agnostic.
    let shells = Array2::from_shape_fn((32, 3), |(i, j)| {
        let centre = if i < 20 {
            [0.0, 0.0, 0.0]
        } else if i < 28 {
            [2.0, 0.0, 0.0]
        } else {
            [4.0, 0.0, 0.0]
        };
        centre[j]
    });
    let vacancy = decaf_from_descriptors(
        shells.view(),
        &DecafOptions {
            dimensions: DimensionChoice::Fixed(1),
            bandwidth: BandwidthChoice::Fixed(0.5),
            ..DecafOptions::default()
        },
    )
    .unwrap();
    assert_eq!(vacancy.n_classes(), 3);
    assert_ne!(vacancy.labels()[0], vacancy.labels()[20]);
    assert_ne!(vacancy.labels()[20], vacancy.labels()[28]);
}

#[test]
fn mds_model_projects_new_descriptors_with_the_training_projector() {
    let training = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let fitted = decaf_from_descriptors(
        training.view(),
        &DecafOptions {
            dimensions: DimensionChoice::Fixed(2),
            bandwidth: BandwidthChoice::Fixed(0.25),
            ..DecafOptions::default()
        },
    )
    .unwrap();
    let projected = fitted
        .project_descriptors(array![[0.5, 0.5]].view())
        .unwrap();
    assert_eq!(projected.nrows(), 1);
    assert_eq!(projected.ncols(), 2);
    assert!(projected.iter().all(|value| value.abs() < 1.0e-12));
}

#[test]
fn published_pah_example_matches_reference_partition() {
    let distances = parse_matrix(include_str!("goldens/decaf_pah_distances.txt"));
    let reference: Vec<usize> = include_str!("goldens/decaf_pah_labels.txt")
        .split_whitespace()
        .map(|value| value.parse().unwrap())
        .collect();
    let fitted = landfold::decaf_from_dissimilarity(
        distances.view(),
        &DecafOptions {
            dimensions: DimensionChoice::Fixed(2),
            bandwidth: BandwidthChoice::Fixed(0.0241),
            ..DecafOptions::default()
        },
    )
    .unwrap();
    assert_eq!(fitted.n_classes(), 11);
    assert_eq!(same_partition(fitted.labels()), same_partition(&reference));

    let default_fit =
        landfold::decaf_from_dissimilarity(distances.view(), &DecafOptions::default()).unwrap();
    assert_eq!(default_fit.n_classes(), 11);
    assert_eq!(
        same_partition(default_fit.labels()),
        same_partition(&reference)
    );
    assert!((default_fit.bandwidth() - 0.024_402_923_154_637_893).abs() < 1.0e-15);
}
