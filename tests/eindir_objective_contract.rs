use eindir_core::Objective;
use landfold::{ChiObjective, Stress, Transfer, pairwise_euclid};
use landfold::provenance::Provenance;
use ndarray::array;

#[test]
fn xtsci_consumes_landfold_objective_with_rgpot_provenance() {
    let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let distances = pairwise_euclid(points.view()).expect("finite source distances");
    let stress = Stress::new(
        distances.clone(),
        distances,
        Transfer::identity(),
        0.0,
        None,
        None,
    )
    .expect("valid stress state");
    let objective = ChiObjective::new(&stress, 2);
    let provenance = Provenance::new_with_eindir_revision(
        "rgpot-run-42",
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "rgpot",
        "rgpot.potentials",
        1,
        0,
        1,
        1,
        0,
        "f3c42130bb389ba6cd6e4cfdc8b2e182f4a764e9",
    )
    .expect("valid rgpot provenance");
    let initial = ndarray::Array1::from_iter(points.iter().copied());
    let report = xtsci_optimize::minimize_method(
        &objective,
        initial,
        &xtsci_optimize::Control {
            maxiter: 40,
            gtol: 1e-10,
            istep: 0.1,
            maxmove: None,
        },
        xtsci_optimize::Method::lbfgs(),
        xtsci_optimize::LineSearch::Brent {
            maxiter: 40,
            tol: 1e-12,
        },
    )
    .expect("xtsci should consume the eindir objective contract");

    assert_eq!(Objective::dim(&objective), 6);
    assert!(report.value.is_finite());
    assert!(report.value < 1e-10);
    assert_eq!(provenance.engine_id, "rgpot");
    assert_eq!(
        provenance.eindir_revision.as_deref(),
        Some("f3c42130bb389ba6cd6e4cfdc8b2e182f4a764e9")
    );
}
