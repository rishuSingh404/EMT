# src/tl_cli.py
import argparse, numpy as np
from .tl_core import LineInputs, basic_params
from .tl_waveforms import plot_envelopes, plot_vswr_vs_freq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--R', type=float, default=0.05)
    ap.add_argument('--L', type=float, default=300e-9)
    ap.add_argument('--G', type=float, default=1e-8)
    ap.add_argument('--C', type=float, default=80e-12)
    ap.add_argument('--f', type=float, default=1e9)
    ap.add_argument('--l', type=float, default=0.25)
    ap.add_argument('--ZLre', type=float, default=50.0)
    ap.add_argument('--ZLim', type=float, default=0.0)
    ap.add_argument('--plots_dir', default='figures')
    args = ap.parse_args()

    inp = LineInputs(args.R,args.L,args.G,args.C,args.f,args.l, complex(args.ZLre,args.ZLim))
    d = basic_params(inp)
    print(f'Z0={d.Z0:.3f}, gamma={d.gamma:.3e}, VSWR={d.VSWR:.3f}, Zin={d.Zin:.3f}')

    plot_envelopes(inp, f'{args.plots_dir}/envelopes.png')
    freqs = np.linspace(args.f*0.5, args.f*1.5, 200)
    plot_vswr_vs_freq(inp, freqs, f'{args.plots_dir}/vswr_vs_f.png')

if __name__ == '__main__':
    main()
