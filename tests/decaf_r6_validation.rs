use landfold::{
    BandwidthChoice, ClassicalMdsModel, DecafOptions, DimensionChoice, MdsDimension,
    decaf_from_dissimilarity,
};
use ndarray::array;

fn fixed_options() -> DecafOptions {
    DecafOptions {
        dimensions: DimensionChoice::Fixed(1),
        bandwidth: BandwidthChoice::Fixed(0.5),
        ..DecafOptions::default()
    }
}

#[test]
fn rejects_negative_all_zero_shortcut_input() {
    let invalid = array![[0.0, -1.0], [-1.0, 0.0]];
    assert!(decaf_from_dissimilarity(invalid.view(), &fixed_options()).is_err());
}

#[test]
fn rejects_nan_all_zero_shortcut_input() {
    let invalid = array![[0.0, f64::NAN], [f64::NAN, 0.0]];
    assert!(decaf_from_dissimilarity(invalid.view(), &fixed_options()).is_err());
}

#[test]
fn rejects_asymmetric_all_zero_shortcut_input() {
    let invalid = array![[0.0, 0.0], [-1.0, 0.0]];
    assert!(decaf_from_dissimilarity(invalid.view(), &fixed_options()).is_err());
}

#[test]
fn fixed_dimension_above_numerical_rank_is_rejected() {
    let collinear = array![[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]];
    assert!(ClassicalMdsModel::fit(collinear.view(), MdsDimension::Fixed(2)).is_err());
}
