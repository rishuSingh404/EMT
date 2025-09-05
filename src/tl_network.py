# src/tl_network.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class ABCD:
    A: complex
    B: complex
    C: complex
    D: complex

    def __matmul__(self, other: "ABCD") -> "ABCD":
        """Cascade: self ∘ other  (self applied to the left of other)."""
        A = self.A*other.A + self.B*other.C
        B = self.A*other.B + self.B*other.D
        C = self.C*other.A + self.D*other.C
        D = self.C*other.B + self.D*other.D
        return ABCD(A, B, C, D)

def abcd_of_shunt_admittance(Y: complex) -> ABCD:
    """Shunt element: Z=0 branch with admittance Y to ground."""
    return ABCD(A=1.0, B=0.0, C=Y, D=1.0)

def abcd_of_series_impedance(Z: complex) -> ABCD:
    """Series element of impedance Z."""
    return ABCD(A=1.0, B=Z, C=0.0, D=1.0)

def abcd_of_tline(gamma: complex, Z0: complex, length: float) -> ABCD:
    """
    AB CD of a transmission-line section of length l with propagation constant gamma and Z0.
    Uses exact hyperbolic formulas (works for lossy lines).
    """
    gl = gamma * length
    cosh_gl = np.cosh(gl)
    sinh_gl = np.sinh(gl)
    A = cosh_gl
    B = Z0 * sinh_gl
    C = (1.0 / Z0) * sinh_gl
    D = cosh_gl
    return ABCD(A, B, C, D)

def z_in_from_abcd(abcd: ABCD, ZL: complex) -> complex:
    """Input impedance of a 2-port terminated by ZL."""
    A, B, C, D = abcd.A, abcd.B, abcd.C, abcd.D
    return (A*ZL + B) / (C*ZL + D)

def gamma_of_impedance(Zin: complex, Z0: complex) -> complex:
    """Reflection coefficient Γ at a port with reference Z0."""
    return (Zin - Z0) / (Zin + Z0 + 1e-30)

def vswr_from_gamma(Gamma: complex) -> float:
    g = abs(Gamma)
    return (1 + g) / (1 - g + 1e-15)

def shunt_stub_admittance_short(Z0: complex, beta: float, l_stub: float) -> complex:
    """Short-circuited shunt stub: B = (1/Z0) * tan(beta * l_stub)."""
    return 1j * (1.0 / Z0) * np.tan(beta * l_stub)

def shunt_stub_admittance_open(Z0: complex, beta: float, l_stub: float) -> complex:
    """Open-circuited shunt stub: B = -(1/Z0) * cot(beta * l_stub) = -(1/Z0)/tan(beta*l)."""
    return -1j * (1.0 / Z0) / (np.tan(beta * l_stub) + 1e-30)
