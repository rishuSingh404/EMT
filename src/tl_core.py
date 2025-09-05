# src/tl_core.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class LineInputs:
    R: float      # ohm/m
    L: float      # H/m
    G: float      # S/m
    C: float      # F/m
    f: float      # Hz
    l: float      # m
    ZL: complex   # load impedance (ohm)
    Vplus: float = 1.0  # forward wave amplitude (arbitrary)

@dataclass
class LineDerived:
    omega: float
    gamma: complex  # alpha + j*beta
    alpha: float
    beta: float
    Z0: complex
    vp: float
    lamb: float
    tau: float
    Gamma_L: complex
    VSWR: float
    RL_dB: float
    ML_dB: float
    Zin: complex

def gamma_Z0(R,L,G,C,omega):
    z = R + 1j*omega*L
    y = G + 1j*omega*C
    gamma = np.sqrt(z*y)
    Z0 = np.sqrt(z/y)
    return gamma, Z0

def basic_params(inp: LineInputs) -> LineDerived:
    w = 2*np.pi*inp.f
    gamma, Z0 = gamma_Z0(inp.R, inp.L, inp.G, inp.C, w)
    alpha, beta = np.real(gamma), np.imag(gamma)
    vp = w/np.maximum(beta, 1e-30)
    lamb = 2*np.pi/np.maximum(beta, 1e-30)
    tau = inp.l/np.maximum(vp, 1e-30)
    Gamma_L = (inp.ZL - Z0)/(inp.ZL + Z0)
    VSWR = (1+np.abs(Gamma_L))/(1-np.abs(Gamma_L) + 1e-15)
    RL_dB = -20*np.log10(np.abs(Gamma_L) + 1e-15)        # return loss
    ML_dB = -10*np.log10(1 - np.abs(Gamma_L)**2 + 1e-15) # mismatch loss
    # Input impedance for lossy line:
    tanh_gl = np.tanh(gamma*inp.l)
    Zin = Z0*(inp.ZL + Z0*tanh_gl)/(Z0 + inp.ZL*tanh_gl)
    return LineDerived(
        omega=w, gamma=gamma, alpha=float(alpha), beta=float(beta),
        Z0=Z0, vp=float(vp), lamb=float(lamb), tau=float(tau),
        Gamma_L=Gamma_L, VSWR=float(VSWR), RL_dB=float(RL_dB),
        ML_dB=float(ML_dB), Zin=Zin
    )

def waves_along_line(inp: LineInputs, npts:int=200):
    """Return z-grid and complex V(z), I(z) along 0..l.
       Convention: z=0 at source, z=l at load. """
    d = basic_params(inp)
    z = np.linspace(0, inp.l, npts)
    # reflect at load:
    Vp = inp.Vplus
    Vm_over_Vp = d.Gamma_L * np.exp(-2*d.gamma*inp.l)
    V = Vp*np.exp(-d.gamma*z) + (Vp*Vm_over_Vp)*np.exp(d.gamma*z)
    I = (Vp/d.Z0)*np.exp(-d.gamma*z) - (Vp*Vm_over_Vp/d.Z0)*np.exp(d.gamma*z)
    return z, V, I, d

def Zin_at_distance(ZL, Z0, gamma, d):
    """Input impedance looking into a segment of length d terminated by ZL."""
    tanh_gd = np.tanh(gamma*d)
    return Z0*(ZL + Z0*tanh_gd)/(Z0 + ZL*tanh_gd)
