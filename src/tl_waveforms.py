# src/tl_waveforms.py
import numpy as np
import matplotlib.pyplot as plt
from .tl_core import waves_along_line

def plot_envelopes(inp, savepath):
    z, V, I, d = waves_along_line(inp, npts=600)
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(z, np.abs(V)/np.max(np.abs(V)), label='|V(z)| (norm)')
    ax.plot(z, np.abs(I)/np.max(np.abs(I)), label='|I(z)| (norm)')
    ax.set_xlabel('z (m)')
    ax.set_ylabel('Magnitude (normalized)')
    ax.set_title(f'Envelopes (Voltage & Current) | VSWR={d.VSWR:.2f}')
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120)
    plt.close(fig)

def plot_vswr_vs_freq(base_inputs, f_array_hz, savepath):
    vs = []
    for f in f_array_hz:
        bi = base_inputs; bi = type(bi)(**{**bi.__dict__, 'f':f})
        from .tl_core import basic_params
        vs.append(basic_params(bi).VSWR)
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(np.array(f_array_hz)*1e-9, vs, linewidth=2)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('VSWR')
    ax.set_title('VSWR vs Frequency')
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(savepath, dpi=120)
    plt.close(fig)
