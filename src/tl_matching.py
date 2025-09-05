import numpy as np
from typing import Literal
from dataclasses import dataclass
from scipy.optimize import brentq, minimize_scalar

# ----------------------------
# Dataclasses for results
# ----------------------------

@dataclass
class StubResult:
    d_opt: float
    l_stub: float
    topology: str
    Zin_src: complex
    Gamma_src: complex
    VSWR_src: float
    notes: str

@dataclass
class QWTResult:
    l_qw: float
    Zt: complex
    Zin_src: complex
    Gamma_src: complex
    VSWR_src: float

# ----------------------------
# Core TL functions
# ----------------------------

def gamma_Z0(R: float, L: float, G: float, C: float, w: float):
    """Compute propagation constant γ and characteristic impedance Z0."""
    Z = R + 1j*w*L
    Y = G + 1j*w*C
    gamma = np.sqrt(Z*Y)
    Z0 = np.sqrt(Z/Y)
    return gamma, Z0

def Zin_at_distance(ZL: complex, Z0: complex, gamma: complex, d: float) -> complex:
    """Input impedance looking into line of length d terminated with ZL."""
    num = ZL + Z0*np.tanh(gamma*d)
    den = Z0 + ZL*np.tanh(gamma*d)
    return Z0 * (num/den)

def abcd_of_tline(gamma: complex, Z0: complex, l: float):
    """ABCD matrix of transmission line section of length l."""
    gl = gamma*l
    A = np.cosh(gl)
    B = Z0*np.sinh(gl)
    C = (1/Z0)*np.sinh(gl)
    D = np.cosh(gl)
    return np.array([[A, B],[C, D]], dtype=complex)

def abcd_of_shunt_admittance(Y: complex):
    """ABCD matrix of a shunt element with admittance Y."""
    return np.array([[1, 0],[Y, 1]], dtype=complex)

def z_in_from_abcd(M: np.ndarray, ZL: complex) -> complex:
    """Input impedance seen at port 1 of ABCD matrix with load ZL at port 2."""
    A,B = M[0,0], M[0,1]
    C,D = M[1,0], M[1,1]
    return (A*ZL + B)/(C*ZL + D)

def gamma_of_impedance(Z: complex, Z0: complex) -> complex:
    """Reflection coefficient at interface."""
    return (Z - Z0)/(Z + Z0)

def vswr_from_gamma(Gamma: complex) -> float:
    """Voltage standing wave ratio from reflection coefficient."""
    g = abs(Gamma)
    if g >= 1: 
        return np.inf
    return (1+g)/(1-g)

# ----------------------------
# Stub helpers
# ----------------------------

def shunt_stub_admittance_short(Z0: complex, beta: float, l: float) -> complex:
    """Short-circuited shunt stub admittance: Yin = -j*cot(beta*l)/Z0."""
    bl = beta * l
    cot = np.cos(bl) / np.maximum(np.sin(bl), 1e-30)
    return -1j * cot / Z0

def shunt_stub_admittance_open(Z0: complex, beta: float, l: float) -> complex:
    """Open-circuited shunt stub admittance: Yin = +j*tan(beta*l)/Z0."""
    bl = beta * l
    tan = np.sin(bl) / np.maximum(np.cos(bl), 1e-30)
    return 1j * tan / Z0

def _solve_stub_length_short(Z0: complex, beta: float, B_needed: float, lamb: float) -> float:
    def f(l):
        return shunt_stub_admittance_short(Z0, beta, l).imag - B_needed
    a, b = 1e-6, 0.5*lamb - 1e-6
    nseg = 8
    for k in range(nseg):
        x0 = a + k*(b-a)/nseg
        x1 = a + (k+1)*(b-a)/nseg
        try:
            if np.sign(f(x0)) == np.sign(f(x1)):
                continue
            return float(brentq(f, x0, x1, maxiter=200))
        except Exception:
            pass
    res = minimize_scalar(lambda x: abs(f(x)), bounds=(a,b), method='bounded')
    return float(res.x)

def _solve_stub_length_open(Z0: complex, beta: float, B_needed: float, lamb: float) -> float:
    def f(l):
        return shunt_stub_admittance_open(Z0, beta, l).imag - B_needed
    a, b = 1e-6, 0.5*lamb - 1e-6
    nseg = 8
    for k in range(nseg):
        x0 = a + k*(b-a)/nseg
        x1 = a + (k+1)*(b-a)/nseg
        try:
            if np.sign(f(x0)) == np.sign(f(x1)):
                continue
            return float(brentq(f, x0, x1, maxiter=200))
        except Exception:
            pass
    res = minimize_scalar(lambda x: abs(f(x)), bounds=(a,b), method='bounded')
    return float(res.x)

# ----------------------------
# Matching: single-stub shunt
# ----------------------------

def single_stub_shunt(
    R: float, L: float, G: float, C: float, f: float,
    l_total: float, ZL: complex, prefer: Literal['short','open']='short'
) -> StubResult:
    w = 2*np.pi*f
    gamma, Z0 = gamma_Z0(R, L, G, C, w)
    beta = np.imag(gamma)
    lamb = 2*np.pi / max(beta, 1e-30)

    # target conductance: Re(Y0)
    Y0 = 1/Z0
    target_Gr = np.real(Y0)

    def g_residual(d):
        Zin_d = Zin_at_distance(ZL, Z0, gamma, d)
        Yd = 1/Zin_d
        return (np.real(Yd) - target_Gr)

    # Find candidate d
    candidates = []
    nseg = 8
    eps = 1e-6
    for k in range(nseg):
        a = (k / nseg) * 0.5*lamb + eps
        b = ((k+1) / nseg) * 0.5*lamb - eps
        try:
            if np.sign(g_residual(a)) == np.sign(g_residual(b)):
                continue
            d_root = brentq(g_residual, a, b, maxiter=200)
            if 0 < d_root < 0.5*lamb:
                candidates.append(d_root)
        except Exception:
            pass

    if not candidates:
        def obj(d):
            Zin_d = Zin_at_distance(ZL, Z0, gamma, d)
            return abs((1/Zin_d).real - target_Gr)
        res = minimize_scalar(obj, bounds=(eps, 0.5*lamb-eps), method='bounded')
        d_opt = float(res.x)
    else:
        def imag_at(d):
            Zin_d = Zin_at_distance(ZL, Z0, gamma, d)
            return abs((1/Zin_d).imag)
        d_opt = float(min(candidates, key=imag_at))

    # Compute required stub susceptance
    Zin_d = Zin_at_distance(ZL, Z0, gamma, d_opt)
    Yd = 1/Zin_d
    B_needed = -Yd.imag

    if prefer == 'short':
        l_stub = _solve_stub_length_short(Z0, beta, B_needed, lamb)
        Y_stub = shunt_stub_admittance_short(Z0, beta, l_stub)
        topology = 'shunt-short'
    else:
        l_stub = _solve_stub_length_open(Z0, beta, B_needed, lamb)
        Y_stub = shunt_stub_admittance_open(Z0, beta, l_stub)
        topology = 'shunt-open'

    # Cascade: load → line(d) → shunt → line(l_total-d) → source
    abcd_d   = abcd_of_tline(gamma, Z0, d_opt)
    abcd_sh  = abcd_of_shunt_admittance(Y_stub)
    abcd_bk  = abcd_of_tline(gamma, Z0, max(l_total - d_opt, 0.0))
    abcd_net = abcd_bk @ (abcd_sh @ abcd_d)
    Zin_src  = z_in_from_abcd(abcd_net, ZL)
    Gamma_src = gamma_of_impedance(Zin_src, Z0)
    VSWR_src  = vswr_from_gamma(Gamma_src)

    notes = (
        f"Placed at d s.t. Re(Y)=Re(1/Z0). Y(d)={Yd:.4g}, B_needed={B_needed:.4g}, "
        f"beta={beta:.4g}, lambda={lamb:.4g}"
    )
    return StubResult(
        d_opt=float(d_opt),
        l_stub=float(l_stub),
        topology=topology,
        Zin_src=Zin_src,
        Gamma_src=Gamma_src,
        VSWR_src=float(VSWR_src),
        notes=notes
    )

# ----------------------------
# Matching: Quarter-wave transformer
# ----------------------------

def quarter_wave_transform(
    R: float, L: float, G: float, C: float, f: float, ZL: complex, Zt: float | None = None
) -> QWTResult:
    w = 2*np.pi*f
    gamma_host, Z0 = gamma_Z0(R, L, G, C, w)
    beta = np.imag(gamma_host)
    lamb = 2*np.pi / max(beta, 1e-30)
    l_qw = lamb/4.0

    if Zt is None:
        Zt = np.sqrt(max(np.real(Z0),1e-9) * max(np.real(ZL),1e-9))

    gamma_tr = 1j*beta
    abcd_tr = abcd_of_tline(gamma_tr, Zt, l_qw)

    Zin_at_input = z_in_from_abcd(abcd_tr, ZL)
    Gamma_src = gamma_of_impedance(Zin_at_input, Z0)
    VSWR_src  = vswr_from_gamma(Gamma_src)

    return QWTResult(
        l_qw=float(l_qw),
        Zt=complex(Zt,0),
        Zin_src=Zin_at_input,
        Gamma_src=Gamma_src,
        VSWR_src=float(VSWR_src)
    )
