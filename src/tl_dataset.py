# src/tl_dataset.py
import numpy as np
import pandas as pd
from .tl_core import LineInputs, basic_params

def _balance_binary(X_list, y_list, target_min_frac=0.45):
    """Undersample the majority class so both classes are close to 50/50."""
    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=int)
    cls0_idx = np.where(y == 0)[0]
    cls1_idx = np.where(y == 1)[0]
    n0, n1 = len(cls0_idx), len(cls1_idx)
    if n0 == 0 or n1 == 0:
        return X, y  # nothing to balance
    # undersample majority to match minority (or slightly higher)
    if n0 > n1:
        keep0 = np.random.choice(cls0_idx, size=max(n1, int(n0*target_min_frac)), replace=False)
        keep1 = cls1_idx
    else:
        keep1 = np.random.choice(cls1_idx, size=max(n0, int(n1*target_min_frac)), replace=False)
        keep0 = cls0_idx
    keep = np.concatenate([keep0, keep1])
    np.random.shuffle(keep)
    return X[keep], y[keep]

def make_dataset(n=1500, task='cls', seed=42, balance=True, log_vswr=False):
    """
    Generate synthetic Tx-line data using your physics solver.
    Defaults chosen to be realistic yet learnable for the classifier.

    task='cls': label = 1 if VSWR <= 2.0 else 0
    task='reg': target = |Zin| (or log10(VSWR) if log_vswr=True)
    """
    rng = np.random.default_rng(seed)

    # Keep losses small (learnable), but not zero; keep ranges physically plausible.
    R = rng.uniform(0.0, 0.2, n)            # ohm/m
    L = rng.uniform(200e-9, 400e-9, n)      # H/m
    G = rng.uniform(0.0, 5e-8, n)           # S/m
    C = rng.uniform(60e-12, 120e-12, n)     # F/m

    # Frequencies around 1 GHz (keeps β, λ in a consistent regime)
    f = rng.uniform(0.6e9, 1.4e9, n)        # Hz

    # Practical line lengths
    l = rng.uniform(0.05, 0.5, n)           # m

    # Loads: mostly real 10–150 Ω, with a bit of reactance to add diversity
    RL = rng.uniform(10, 150, n)
    XL_choices = np.array([0.0, -25.0, 25.0, -50.0, 50.0])
    XL = rng.choice(XL_choices, size=n, p=[0.6, 0.1, 0.1, 0.1, 0.1])
    ZL = RL + 1j*XL

    # Build feature matrix X and targets
    X_list, y_list = [], []
    for i in range(n):
        inp = LineInputs(R[i], L[i], G[i], C[i], f[i], l[i], complex(ZL[i]))
        d = basic_params(inp)
        feats = [R[i], L[i], G[i], C[i], f[i], l[i], np.real(ZL[i]), np.imag(ZL[i])]

        if task == 'cls':
            label = 1 if d.VSWR <= 2.0 else 0
            y_list.append(label)
        else:
            if log_vswr:
                y_list.append(np.log10(d.VSWR + 1e-12))
            else:
                # regress on |Zin| by default
                y_list.append(abs(d.Zin))
        X_list.append(feats)

    X = np.array(X_list, dtype=float)

    if task == 'cls' and balance:
        X, y = _balance_binary(X_list, y_list)
        cols = ['R','L','G','C','f','l','ZL_re','ZL_im']
        return pd.DataFrame(X, columns=cols), pd.Series(y, name='label')
    else:
        cols = ['R','L','G','C','f','l','ZL_re','ZL_im']
        return pd.DataFrame(X, columns=cols), pd.Series(y_list, name=('label' if task=='cls' else 'y'))
