"""Stability check and buckling factor give the same answers on the fast
symmetric eigen-solvers as on the general ones."""

import math

import numpy as np
import pytest

from anastruct import SystemElements
from anastruct.basic import FEMException
from anastruct.fem.system_components.solver import det_linear_buckling

EI = 1810.0
EA = 460000.0


def _beam(supports: str, axial: float = 0.0, n: int = 12) -> SystemElements:
    ss = SystemElements()
    xs = np.linspace(0, 6, n + 1)
    for a, b in zip(xs[:-1], xs[1:]):
        ss.add_element([[a, 0], [b, 0]], EA=EA, EI=EI)
    left, right = supports
    for node, kind in ((1, left), (n + 1, right)):
        if kind == "p":
            ss.add_support_hinged(node)
        elif kind == "f":
            ss.add_support_fixed(node)
        else:
            ss.add_support_roll(node, direction=2)
    for e in range(1, n + 1):
        ss.q_load(q=-5.0, element_id=e, direction="element")
    if axial:
        ss.point_load(n + 1, Fx=axial)
    return ss


@pytest.mark.parametrize("supports", ["pr", "fr", "ff", "pp"])
def test_a_stable_beam_solves(supports):
    _beam(supports).solve()


def test_a_mechanism_is_rejected():
    """Two rollers: nothing stops the beam sliding along its axis."""
    with pytest.raises(FEMException, match="StabilityError"):
        _beam("rr").solve()


@pytest.mark.parametrize("supports, k", [("pr", 1.0), ("fr", 0.6992)])
def test_buckling_factor_matches_the_euler_load(supports, k):
    """Euler load π²EI/(kL)²: pinned–pinned k = 1, fixed–pinned k = 0.6992."""
    n_cr = math.pi**2 * EI / (k * 6) ** 2
    ss = _beam(supports, axial=-100.0, n=24)
    assert det_linear_buckling(ss) == pytest.approx(n_cr / 100.0, rel=2e-3)
