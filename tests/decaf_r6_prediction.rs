use approx::assert_abs_diff_eq;
use landfold::{
    BandwidthChoice, DecafOptions, DimensionChoice, decaf_from_descriptors,
    decaf_from_dissimilarity,
};
use ndarray::{Array2, ArrayView2, array};

fn pair_distances(left: ArrayView2<'_, f64>, right: ArrayView2<'_, f64>) -> Array2<f64> {
    Array2::from_shape_fn((left.nrows(), right.nrows()), |(i, j)| {
        left.row(i)
            .iter()
            .zip(right.row(j))
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    })
}

fn normalized_reference_training() -> Array2<f64> {
    array![
        [1.0, 0.0, 0.0],
        [0.8, 0.6, 0.0],
        [0.8, 0.0, 0.6],
        [-1.0, 0.0, 0.0],
        [-0.8, -0.6, 0.0],
        [-0.8, 0.0, -0.6],
        [0.0, 1.0, 0.0],
    ]
}

#[test]
fn out_of_sample_prediction_matches_reference_projection() {
    let training = normalized_reference_training();
    let queries = array![[0.6, 0.8, 0.0], [0.6, 0.0, 0.8]];
    let fitted = decaf_from_descriptors(
        training.view(),
        &DecafOptions {
            dimensions: DimensionChoice::Fixed(2),
            bandwidth: BandwidthChoice::Fixed(0.75),
            ..DecafOptions::default()
        },
    )
    .unwrap();

    let prediction = fitted.predict_descriptors(queries.view()).unwrap();
    let projected_distances =
        pair_distances(fitted.embedded().view(), prediction.embedded().view());
    let reference = super_parse_matrix(include_str!("goldens/decaf_oos_reference_distances.txt"));
    assert_eq!(projected_distances.dim(), reference.dim());
    for (actual, expected) in projected_distances.iter().zip(&reference) {
        assert_abs_diff_eq!(actual, expected, epsilon = 2.0e-12);
    }
    assert_eq!(prediction.labels().len(), queries.nrows());
    assert_eq!(prediction.memberships().nrows(), queries.nrows());
    for row in prediction.memberships().rows() {
        assert_abs_diff_eq!(row.sum(), 1.0, epsilon = 1.0e-12);
    }
}

#[test]
fn projecting_training_points_reproduces_the_fitted_embedding() {
    let training = normalized_reference_training();
    let distances = pair_distances(training.view(), training.view());
    let fitted = decaf_from_dissimilarity(
        distances.view(),
        &DecafOptions {
            dimensions: DimensionChoice::Fixed(2),
            bandwidth: BandwidthChoice::Fixed(0.75),
            ..DecafOptions::default()
        },
    )
    .unwrap();

    let prediction = fitted.predict_distances(distances.view()).unwrap();
    assert_eq!(prediction.embedded().dim(), fitted.embedded().dim());
    for (actual, expected) in prediction.embedded().iter().zip(fitted.embedded()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 2.0e-12);
    }
}

#[test]
fn projected_classification_exposes_labels_and_memberships() {
    let training = array![[0.0], [0.1], [4.0], [4.1]];
    let fitted = decaf_from_descriptors(
        training.view(),
        &DecafOptions {
            dimensions: DimensionChoice::Fixed(1),
            bandwidth: BandwidthChoice::Fixed(0.5),
            ..DecafOptions::default()
        },
    )
    .unwrap();

    let prediction = fitted.classify(fitted.centers().view()).unwrap();
    assert_eq!(prediction.labels(), &[0, 1]);
    assert_eq!(
        prediction.memberships().dim(),
        (fitted.n_classes(), fitted.n_classes())
    );
    for class in 0..fitted.n_classes() {
        assert!(
            prediction.memberships()[(class, class)] > prediction.memberships()[(class, 1 - class)]
        );
    }
}

#[test]
fn representative_is_a_medoid_not_the_observation_nearest_the_class_mean() {
    let observations = array![[0.0], [1.0], [2.0], [4.0]];
    let fitted = decaf_from_descriptors(
        observations.view(),
        &DecafOptions {
            dimensions: DimensionChoice::Fixed(1),
            bandwidth: BandwidthChoice::Fixed(20.0),
            ..DecafOptions::default()
        },
    )
    .unwrap();
    assert_eq!(fitted.n_classes(), 1);

    let center = fitted.centers().row(0);
    let nearest_to_center = fitted
        .embedded()
        .rows()
        .into_iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| {
            pair_distances(
                left.insert_axis(ndarray::Axis(0)),
                center.insert_axis(ndarray::Axis(0)),
            )[(0, 0)]
                .total_cmp(
                    &pair_distances(
                        right.insert_axis(ndarray::Axis(0)),
                        center.insert_axis(ndarray::Axis(0)),
                    )[(0, 0)],
                )
        })
        .unwrap()
        .0;
    assert_eq!(fitted.representatives(), &[1]);
    assert_eq!(nearest_to_center, 2);
    assert_ne!(fitted.representatives()[0], nearest_to_center);
}

fn super_parse_matrix(text: &str) -> Array2<f64> {
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
