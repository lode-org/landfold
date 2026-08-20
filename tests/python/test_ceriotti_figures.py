"""Regression tests for the public Ceriotti figure helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

SCRIPT = Path(__file__).parents[2] / "scripts" / "ceriotti_figures.py"
SPEC = importlib.util.spec_from_file_location("ceriotti_figures", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
FIGURES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIGURES)


def test_fes_hist_scales_energy_by_kt() -> None:
    points = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])

    _, _, fes = FIGURES.fes_hist(points, 2, 2, 0.5)
    _, _, unit_fes = FIGURES.fes_hist(points, 2, 2, 1.0)

    assert fes == pytest.approx(0.5 * unit_fes, nan_ok=True)


def test_fes_hist_supports_constant_coordinates() -> None:
    edges_x, edges_y, fes = FIGURES.fes_hist(np.ones((3, 2)), 4, 4, 1.0)

    assert np.all(np.diff(edges_x) > 0)
    assert np.all(np.diff(edges_y) > 0)
    assert np.isfinite(fes).any()


@pytest.mark.parametrize("kt", [0.0, -1.0, np.nan, np.inf])
def test_fes_hist_rejects_invalid_kt(kt: float) -> None:
    with pytest.raises(ValueError, match="kt"):
        FIGURES.fes_hist(np.zeros((1, 2)), 2, 2, kt)
