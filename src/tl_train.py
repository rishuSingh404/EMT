class EnsembleAvg:
    def __init__(self, models, weights=None, scaler=None):
        self.models = models
        self.weights = weights if weights is not None else [1.0/len(models)]*len(models)
        self.scaler = scaler
    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self
    def predict_proba(self, X):
        probs = []
        for w, m in zip(self.weights, self.models):
            p = m.predict_proba(X)[:, 1]
            probs.append(w * p)
        p_avg = np.sum(probs, axis=0)
        p0 = 1.0 - p_avg
        return np.vstack([p0, p_avg]).T
    def predict(self, X, threshold=0.5):
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= threshold).astype(int)
# src/tl_train.py
import argparse, joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from .tl_dataset import make_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['cls','reg'], default='cls',
                    help="Classification (default) or regression.")
    ap.add_argument('-n', type=int, default=10000, help="Number of synthetic samples to generate.")
    ap.add_argument('--out', default='models/model.pkl', help="Where to save the trained model.")
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-balance', action='store_true', help="Disable class balancing for classification.")
    ap.add_argument('--log-vswr', action='store_true', help="Regression target = log10(VSWR) instead of |Zin|.")
    args = ap.parse_args()

    if args.task == 'cls':
        X, y = make_dataset(n=args.n, task='cls', seed=args.seed, balance=(not args.no_balance))
        # --- balance dataset ---
        from sklearn.utils import resample
        n_min = min((y==0).sum(), (y==1).sum())
        X_balanced = np.vstack([
            resample(X[y==0], n_samples=n_min, random_state=42),
            resample(X[y==1], n_samples=n_min, random_state=42)
        ])
        y_balanced = np.hstack([
            np.zeros(n_min, dtype=int),
            np.ones(n_min, dtype=int)
        ])
        X, y = X_balanced, y_balanced

        # --- scale features (optional) ---
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)

        # --- train ensemble RF + LightGBM ---
        from sklearn.ensemble import RandomForestClassifier
        from lightgbm import LGBMClassifier

        rf = RandomForestClassifier(
            n_estimators=1500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )
        lgbm = LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.05,
            num_leaves=64,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        weights = [0.4, 0.6]
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
        ens = EnsembleAvg(models=[rf, lgbm], weights=weights, scaler=scaler)

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        ens.fit(Xtr, ytr)

        from sklearn.metrics import accuracy_score, roc_auc_score
        y_pred = ens.predict(Xte, threshold=0.5)
        y_proba = ens.predict_proba(Xte)[:, 1]
        acc = accuracy_score(yte, y_pred)
        auc = roc_auc_score(yte, y_proba)
        print(f"Ensemble accuracy: {acc:.4f}")
        print(f"Ensemble ROC AUC:  {auc:.4f}")

        # 5-fold Stratified CV
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs, aucs = [], []
        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            rf_cv = RandomForestClassifier(
                n_estimators=1500, max_depth=None, min_samples_split=2, min_samples_leaf=2,
                max_features='sqrt', n_jobs=-1, random_state=42
            )
            lgbm_cv = LGBMClassifier(
                n_estimators=2000, learning_rate=0.05, num_leaves=64, max_depth=-1,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
            )
            ens_cv = EnsembleAvg([rf_cv, lgbm_cv], weights=weights)
            ens_cv.fit(X_tr, y_tr)
            y_pred_cv = ens_cv.predict(X_te)
            y_proba_cv = ens_cv.predict_proba(X_te)[:, 1]
            accs.append(accuracy_score(y_te, y_pred_cv))
            aucs.append(roc_auc_score(y_te, y_proba_cv))
        print(f"5-fold CV accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"5-fold CV ROC AUC:  {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

        assert acc >= 0.98, "Accuracy < 98% — consider increasing n or tuning weights."
        joblib.dump(ens, args.out)

    else:
        # Regression
        X, y = make_dataset(n=args.n, task='reg', seed=args.seed, log_vswr=args.log_vswr)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
        model = RandomForestRegressor(
            n_estimators=1200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=0
        )
        model.fit(Xtr, ytr)
        p = model.predict(Xte)

        if args.log_vswr:
            # Evaluate R^2 on the original VSWR scale for honesty:
            vswr_true = np.power(10.0, yte.to_numpy())
            vswr_pred = np.power(10.0, p)
            r2 = r2_score(vswr_true, vswr_pred)
            print(f'Regression R2 (back on VSWR): {r2:.4f}')
        else:
            r2 = r2_score(yte, p)
            print(f'Regression R2: {r2:.4f}')

        # No hard assert for regression (assignment wants accuracy; regression is optional)
        joblib.dump({'task':'reg','model':model,'log_vswr':args.log_vswr}, args.out)

if __name__ == '__main__':
    main()
