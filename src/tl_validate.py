# src/tl_validate.py
import numpy as np
from .tl_core import LineInputs, basic_params

def approx(a, b, tol=1e-3):
    """Relative tolerance check: |a-b| <= tol*(1+|b|)."""
    return abs(a - b) <= tol * (1 + abs(b))

def test_matched():
    """
    Lossless benchmark: choose L and C so that Z0 = sqrt(L/C) = 50 ohm.
    With R=G=0 and ZL=50 Ω (purely real), we must have Γ≈0 and VSWR≈1.
    """
    # Pick L and compute C so that Z0 = sqrt(L/C) = 50 Ω
    L = 300e-9                 # H/m
    Z0_target = 50.0           # ohm
    C = L / (Z0_target**2)     # -> 120e-12 F/m

    inp = LineInputs(
        R=0.0, L=L, G=0.0, C=C,
        f=1e9, l=0.2,
        ZL=50.0 + 0j
    )
    d = basic_params(inp)

    # Sanity: Z0 should be (very nearly) 50 Ω and purely real for lossless
    assert abs(d.Z0.real - Z0_target) < 1e-6, f"Expected Z0≈50Ω, got {d.Z0}"
    assert abs(d.Z0.imag) < 1e-9, f"Expected lossless Z0 to be ~real, got {d.Z0}"

    # Perfect match checks
    assert abs(d.Gamma_L) < 1e-8, f"Not matched: Γ={d.Gamma_L}"
    assert d.VSWR < 1.00001, f"VSWR not ~1: VSWR={d.VSWR}"

def test_half_lambda_periodicity():
    """
    For any given line, Z_in(l + λ/2) ≈ Z_in(l) (impedance repeats every half-wavelength).
    We test real parts equality within a small tolerance.
    """
    inp = LineInputs(
        R=0.0, L=300e-9, G=0.0, C=80e-12,
        f=1e9, l=0.1,
        ZL=100.0 + 0j
    )
    d1 = basic_params(inp)
    lamb = d1.lamb

    inp2 = LineInputs(inp.R, inp.L, inp.G, inp.C, inp.f, inp.l + 0.5 * lamb, inp.ZL)
    d2 = basic_params(inp2)

    assert approx(np.real(d1.Zin), np.real(d2.Zin), 1e-3), \
        f"Zin periodicity failed: {d1.Zin} vs {d2.Zin}"

if __name__ == '__main__':
    test_matched()
    test_half_lambda_periodicity()
    print("OK")
