# -*- coding: utf-8 -*-
"""
Logistic Cluster Elastic Net (Logistic CEN) on DOROTHEA  (real data only)
=========================================================================
Objective (guideline Eq. 3.4):
    F = (1/n) sum[log(1+e^eta) - y*eta] + lam/n * sum_k sum_{j in C_k} ||f_j - fbar_k||^2 + delta*||beta||_1
    f_j = X_j * beta_j  (standardised columns),  K = number of predictor clusters.
Solver : alternating  (2a) cyclic coordinate descent on the global MM surrogate w = 1/4 (Option 1)
                      (2b) k-means on {f_j} with strict-decrease acceptance rule (eps_C)
Baselines (same solver, same objective, same folds):
    Lasso (lam=0) | Ridge | Elastic Net | CEN K=1 | CEN K=2,3,5 | CEN-multi (K tuned)
Protocol: leak-free. Screening, mu_j, s_j, EN init, k-means, tuning, tau_hat are all computed
          inside the training part only.  Outer 5-fold CV on dorothea_train (n=800)
          + (optional) one-shot external evaluation on dorothea_valid (n=350) if the files exist.

Run:   python logistic_cen_dorothea.py --selftest        (unit tests, ~1 min)
       python logistic_cen_dorothea.py --preset tiny     (1-2 min smoke test)
       python logistic_cen_dorothea.py --preset quick --n_jobs 5   (full run)
       python logistic_cen_dorothea.py --preset quick --skip_cv    (reuse existing per_fold_results.csv, ~6.5 min)
"""
import os, sys, time, argparse, warnings
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))   
os.environ["PYTHONWARNINGS"] = "ignore"   # joblib worker process-এও warning বন্ধ করে
import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve

try:
    from numba import njit
except ImportError:                      # slow fallback (works, but very slow)
    def njit(*a, **k):
        if len(a) == 1 and callable(a[0]):
            return a[0]
        return lambda f: f
    print("WARNING: numba not found -> run  pip install numba  (otherwise extremely slow)")

warnings.filterwarnings("ignore")

# =====================================================================================
# 0. YOUR PATHS
# =====================================================================================
TRAIN_DATA   = r"E:\Anaconda Projects\Project( Jupyter NB)\Porject\dorothea_train.data"
TRAIN_LABELS = r"E:\Anaconda Projects\Project( Jupyter NB)\Porject\dorothea_train.labels"
VALID_DATA   = r"E:\Anaconda Projects\Project( Jupyter NB)\Porject\dorothea_valid.data"
VALID_LABELS = r"E:\Anaconda Projects\Project( Jupyter NB)\Porject\dorothea_valid.labels"
OUT_DIR      = os.path.join(os.path.dirname(TRAIN_DATA), "logistic_cen_results")
N_FEATURES   = 100000
PRESETS = {
    "quick": dict(m=1000, min_nnz=3, n_delta=8, delta_min_ratio=1e-2, lam_grid=[0.01, 0.1, 1.0],
                  K_grid=[2, 3, 5], outer_folds=5, inner_folds=3, kmeans_ninit=5),
    "tiny":  dict(m=300, min_nnz=3, n_delta=4, delta_min_ratio=3e-2, lam_grid=[0.1, 1.0],
                  K_grid=[2, 3], outer_folds=2, inner_folds=2, kmeans_ninit=3),   # smoke test only
    "full":  dict(m=1000, min_nnz=3, n_delta=15, delta_min_ratio=1e-3, lam_grid=[1e-3, 1e-2, 1e-1, 1.0, 5.0, 10.0],
                  K_grid=[2, 3, 5], outer_folds=5, inner_folds=5, kmeans_ninit=20),
}
COMMON = dict(tol=1e-5, max_pass=500, T_max=30, eps_C=1e-6, cv_metric="roc_auc", seed=2024)


# =====================================================================================
# 1. DATA
# =====================================================================================
def load_dorothea(data_path, labels_path, p=N_FEATURES):
    """DOROTHEA .data = one line per compound, 1-based indices of the non-zero (=1) features."""
    rows, cols = [], []
    with open(data_path, "r") as fh:
        for i, line in enumerate(fh):
            idx = np.array(line.split(), dtype=np.int64) - 1
            rows.append(np.full(idx.size, i, dtype=np.int64)); cols.append(idx)
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    n = rows.max() + 1
    X = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n, p))
    y = (np.loadtxt(labels_path) > 0).astype(float)          # +1 -> 1 , -1 -> 0
    assert X.shape[0] == y.size
    return X, y


def screen_and_standardize(Xtr, ytr, Xtests, m, min_nnz):
    """Training-only: drop rare/constant columns, keep top-m by |corr(X_j,y)|, standardise with
    training mu_j, s_j (population sd, so (1/n)||X_j||^2 = 1, Eq. 3.1-3.2). Returns dense arrays."""
    Xc = Xtr.tocsc()
    n = Xc.shape[0]
    mu = np.asarray(Xc.mean(axis=0)).ravel()
    s = np.sqrt(np.maximum(np.asarray(Xc.multiply(Xc).mean(axis=0)).ravel() - mu ** 2, 0))
    nnz = np.diff(Xc.indptr)
    ok = (nnz >= min_nnz) & (s > 1e-12)
    ybar, sy = ytr.mean(), ytr.std()
    cov = np.asarray(Xc.T @ ytr).ravel() / n - mu * ybar
    score = np.zeros(Xc.shape[1])
    score[ok] = np.abs(cov[ok]) / (s[ok] * sy)
    idx = np.argsort(-score)[:m]
    idx = np.sort(idx[ok[idx]])
    Xs = (Xc[:, idx].toarray() - mu[idx]) / s[idx]
    tests = [(Xt.tocsc()[:, idx].toarray() - mu[idx]) / s[idx] for Xt in Xtests]
    return np.asfortranarray(Xs), tests, idx


# =====================================================================================
# 2. CORE SOLVER  (Eq. 4.4 - 4.8, Option 1: w_i = 1/4 global majorizer)
# =====================================================================================
@njit(cache=True)
def cd_run(X, y, b0, beta, labels, K, lam, delta, gamma2, tol, max_pass):
    """Repeated (surrogate rebuild + ONE full cyclic CD pass) at fixed partition `labels`.
    gamma2 is an optional ridge term (gamma2/2)||beta||^2 used only for Ridge / Elastic-Net baselines."""
    n, m = X.shape
    beta = beta.copy()
    w = 0.25
    eta = np.full(n, b0)
    counts = np.zeros(K)
    for j in range(m):
        counts[labels[j]] += 1.0
        if beta[j] != 0.0:
            for i in range(n):
                eta[i] += X[i, j] * beta[j]
    fbar = np.zeros((K, n))
    for j in range(m):
        if beta[j] != 0.0:
            k = labels[j]
            c = beta[j] / counts[k]
            for i in range(n):
                fbar[k, i] += c * X[i, j]
    npass = 0
    while npass < max_pass:
        npass += 1
        rvec = np.empty(n)
        s = 0.0
        for i in range(n):
            e = eta[i]
            if e >= 0:
                pi = 1.0 / (1.0 + np.exp(-e))
            else:
                ez = np.exp(e)
                pi = ez / (1.0 + ez)
            rvec[i] = y[i] - pi
            s += rvec[i]
        d0 = s / (n * w)                      # Eq. 4.4
        b0 += d0
        for i in range(n):
            eta[i] += d0
            rvec[i] -= w * d0
        num = d0 * d0
        den = b0 * b0
        for j in range(m):
            k = labels[j]
            cnt = counts[k]
            A = w + 2.0 * lam * (cnt - 1.0) / cnt + gamma2          # Eq. 4.5 (+ ridge)
            xr = 0.0
            xf = 0.0
            for i in range(n):
                xr += X[i, j] * rvec[i]
                xf += X[i, j] * fbar[k, i]
            bo = beta[j]
            U = xr / n + w * bo + 2.0 * lam * (xf / n - bo / cnt)   # Eq. 4.6 (centroid identity 3.13)
            if U > delta:
                bn = (U - delta) / A                                 # Eq. 4.7
            elif U < -delta:
                bn = (U + delta) / A
            else:
                bn = 0.0
            dj = bn - bo
            if dj != 0.0:
                beta[j] = bn
                c = dj / cnt
                for i in range(n):                                   # Eq. 4.8, O(n)
                    xij = X[i, j]
                    eta[i] += dj * xij
                    rvec[i] -= w * dj * xij
                    fbar[k, i] += c * xij
            num += dj * dj
            den += bn * bn
        if num / max(den, 1e-12) < tol:
            break
    return b0, beta, npass


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def cluster_penalty(X, beta, labels, lam):
    """P_cluster = lam/n * sum_k sum_{j in C_k} ||f_j - fbar_k||^2   (Eq. 3.10)"""
    n = X.shape[0]
    tot = 0.0
    for k in np.unique(labels):
        mem = np.where(labels == k)[0]
        F = X[:, mem] * beta[mem]
        tot += np.sum((F - F.mean(axis=1, keepdims=True)) ** 2)
    return lam * tot / n


def objective(X, y, b0, beta, labels, lam, delta, gamma2=0.0):
    eta = b0 + X @ beta
    loss = float(np.mean(np.logaddexp(0.0, eta) - y * eta))
    pc = float(cluster_penalty(X, beta, labels, lam)) if lam > 0 else 0.0
    pl1 = float(delta * np.abs(beta).sum() + 0.5 * gamma2 * beta @ beta)
    return dict(loss=loss, pen_cluster=pc, pen_l1=pl1, total=loss + pc + pl1)


def _compact(labels):
    return np.unique(labels, return_inverse=True)[1].astype(np.int64)


def update_partition(X, beta, labels, lam, K, cfg, seed):
    """Step 2b: Lloyd from current centroids + random restarts, strict-decrease acceptance (Eq. 4.9)."""
    if lam == 0 or K <= 1 or not np.any(beta):
        return labels, 0
    F = (X * beta).T                                                   # p x n
    pen = lambda lab: cluster_penalty(X, beta, lab, lam)
    p_cur = pen(labels)
    cands = []
    ks = np.unique(labels)
    if len(ks) == K:
        cent = np.vstack([F[labels == k].mean(axis=0) for k in ks])
        cands.append(_compact(KMeans(K, init=cent, n_init=1, max_iter=300).fit(F).labels_))
    cands.append(_compact(KMeans(K, n_init=cfg["kmeans_ninit"], random_state=seed).fit(F).labels_))
    best = min(cands, key=pen)
    if pen(best) <= p_cur - cfg["eps_C"]:
        return best, int(np.sum(best != labels))        # (#switched is label-permutation sensitive; indicative only)
    return labels, 0


def fit_cen(X, y, lam, delta, K, b0, beta, cfg, gamma2=0.0, labels=None, seed=0, log=False):
    """Alternating algorithm of Section 4.3. K=1 or lam=0 -> convex, single CD run."""
    m = X.shape[1]
    hist = []
    t0 = time.time()
    if K <= 1 or lam == 0:
        lab = np.zeros(m, dtype=np.int64)
        b0, beta, npass = cd_run(X, y, b0, beta, lab, 1, lam, delta, gamma2, cfg["tol"], cfg["max_pass"])
        return b0, beta, lab, hist
    if labels is None:
        if np.any(beta):
            labels = _compact(KMeans(K, n_init=cfg["kmeans_ninit"], random_state=seed).fit((X * beta).T).labels_)
        else:
            labels = np.zeros(m, dtype=np.int64)
    for t in range(1, cfg["T_max"] + 1):
        beta_prev = beta.copy()
        b0, beta, npass = cd_run(X, y, b0, beta, labels, int(labels.max()) + 1, lam, delta, gamma2,
                                 cfg["tol"], cfg["max_pass"])
        labels_new, nsw = update_partition(X, beta, labels, lam, K, cfg, seed + t)
        changed = not np.array_equal(labels_new, labels)
        rel = float(np.sum((beta - beta_prev) ** 2) / max(np.sum(beta_prev ** 2), 1e-12))
        labels = labels_new
        if log:
            ob = objective(X, y, b0, beta, labels, lam, delta, gamma2)
            hist.append(dict(iter=t, loss_logistic=ob["loss"], pen_cluster=ob["pen_cluster"], pen_l1=ob["pen_l1"],
                             obj_total=ob["total"], rel_beta_change=rel, n_active=int(np.sum(beta != 0)),
                             partition_changed=changed, n_switched=nsw, cd_passes=npass,
                             elapsed_sec=time.time() - t0, converged=(not changed) and rel < cfg["tol"]))
        if (not changed) and rel < cfg["tol"]:
            break
    return b0, beta, labels, hist


# =====================================================================================
# 3. ALL CANDIDATE MODELS ON ONE TRAINING SET  (paths with warm starts)
# =====================================================================================
def fit_all_configs(X, y, cfg, seed=0):
    n, m = X.shape
    ybar = y.mean()
    b00 = np.log(ybar / (1 - ybar))
    zero = np.zeros(m)
    lab0 = np.zeros(m, dtype=np.int64)
    dmax = np.max(np.abs(X.T @ (y - ybar))) / n
    deltas = dmax * np.geomspace(1.0, cfg["delta_min_ratio"], cfg["n_delta"])    # descending
    out, times = {}, {}
    tic = lambda: time.time()
    run = lambda b0, be, lam, d, g: cd_run(X, y, b0, be, lab0, 1, lam, d, g, cfg["tol"], cfg["max_pass"])[:2]

    t = tic(); b0, be = b00, zero                                    # Lasso (lambda = 0)
    for i, d in enumerate(deltas):
        b0, be = run(b0, be, 0.0, d, 0.0); out[("lasso", i)] = (b0, be, None, d)
    times["Lasso"] = tic() - t

    t = tic(); b0, be = b00, zero                                    # Ridge
    for g in sorted(cfg["lam_grid"], reverse=True):
        b0, be = run(b0, be, 0.0, 0.0, g); out[("ridge", g)] = (b0, be, None, 0.0)
    times["Ridge"] = tic() - t

    t = tic()                                                        # Elastic Net  (gamma2 = lam grid)
    for g in cfg["lam_grid"]:
        b0, be = b00, zero
        for i, d in enumerate(deltas):
            b0, be = run(b0, be, 0.0, d, g); out[("enet", g, i)] = (b0, be, None, d)
    times["ElasticNet"] = tic() - t

    for K in [1] + list(cfg["K_grid"]):                              # CEN: init from EN(gamma2 = lam)
        t = tic()
        for lam in cfg["lam_grid"]:
            for i, d in enumerate(deltas):
                b0e, bee, _, _ = out[("enet", lam, i)]
                b0c, bec, lab, _ = fit_cen(X, y, lam, d, K, b0e, bee, cfg, labels=None, seed=seed)
                out[("cen", K, lam, i)] = (b0c, bec, lab, d)
        times["CEN-K%d" % K] = tic() - t
    return out, times


def _hyper_str(key, d):
    if key[0] == "ridge": return "gamma2=%g" % key[1]
    if key[0] == "lasso": return "delta=%.4g" % d
    if key[0] == "enet":  return "gamma2=%g, delta=%.4g" % (key[1], d)
    return "K=%d, lambda=%g, delta=%.4g" % (key[1], key[2], d)


def families(key):
    f = key[0]
    if f == "lasso": return ["Lasso"]
    if f == "ridge": return ["Ridge"]
    if f == "enet":  return ["ElasticNet"]
    return ["CEN-K%d" % key[1]] + (["CEN-multi(K tuned)"] if key[1] >= 2 else [])


# =====================================================================================
# 4. METRICS / THRESHOLD
# =====================================================================================
def best_threshold(y, p):
    cand = np.unique(np.quantile(p, np.linspace(0, 1, 401)))
    ba = np.array([_ba(y, p >= c) for c in cand])
    return float(np.median(cand[ba >= ba.max() - 1e-12]))


def _ba(y, pred):
    tp = np.sum((pred == 1) & (y == 1)); fn = np.sum((pred == 0) & (y == 1))
    tn = np.sum((pred == 0) & (y == 0)); fp = np.sum((pred == 1) & (y == 0))
    return 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))


def evaluate(y, p, tau):
    pred = (p >= tau).astype(int)
    tp = np.sum((pred == 1) & (y == 1)); fn = np.sum((pred == 0) & (y == 1))
    tn = np.sum((pred == 0) & (y == 0)); fp = np.sum((pred == 1) & (y == 0))
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    return dict(ROC_AUC=roc_auc_score(y, p), PR_AUC=average_precision_score(y, p), BalAcc=0.5 * (sens + spec),
                BER=1 - 0.5 * (sens + spec), Sens=sens, Spec=spec,
                F1=2 * prec * sens / max(prec + sens, 1e-12), Brier=float(np.mean((y - p) ** 2)))


# =====================================================================================
# 5. ONE FULL LEAK-FREE RUN:  inner-CV tuning on (Xtr,ytr)  ->  refit  ->  evaluate on (Xte,yte)
# =====================================================================================
def run_pipeline(Xtr, ytr, Xte, yte, cfg, seed, tag=""):
    t0 = time.time()
    ntr = len(ytr)
    skf = StratifiedKFold(cfg["inner_folds"], shuffle=True, random_state=seed)
    fold_auc, oof = {}, {}
    for tr, va in skf.split(np.zeros(ntr), ytr):
        Xs, (Xv,), _ = screen_and_standardize(Xtr[tr], ytr[tr], [Xtr[va]], cfg["m"], cfg["min_nnz"])
        res, _ = fit_all_configs(Xs, ytr[tr], cfg, seed)
        for key, (b0, be, _, _) in res.items():
            p = sigmoid(b0 + Xv @ be)
            sc = roc_auc_score(ytr[va], p) if cfg["cv_metric"] == "roc_auc" else average_precision_score(ytr[va], p)
            fold_auc.setdefault(key, []).append(sc)
            oof.setdefault(key, np.full(ntr, np.nan))[va] = p
    score = {k: np.mean(v) for k, v in fold_auc.items()}
    best = {}                                                        # best config per family
    for k in score:                                                  # insertion order = strongest regularisation first
        for fam in families(k):
            if fam not in best or score[k] > score[best[fam]] + 1e-12:
                best[fam] = k

    Xs, (Xt,), idx = screen_and_standardize(Xtr, ytr, [Xte], cfg["m"], cfg["min_nnz"])
    res, times = fit_all_configs(Xs, ytr, cfg, seed)
    times["CEN-multi(K tuned)"] = sum(times["CEN-K%d" % K] for K in cfg["K_grid"])
    rows, hist = [], None
    for fam, key in best.items():
        b0, be, lab, dval = res[key]
        p = sigmoid(b0 + Xt @ be)
        tau = best_threshold(ytr, oof[key])
        r = dict(tag=tag, family=fam, cv_score=score[key], tau=tau, n_selected=int(np.sum(be != 0)),
                 train_sec=times[fam], hyper=_hyper_str(key, dval),
                 sel_idx=idx[be != 0], y_test=yte, p_test=p, beta=be)
        r.update(evaluate(yte, p, tau))
        r["labels"] = lab; r["idx"] = idx          # needed by simulation_cen.py (ARI) and R15
        if lab is not None and key[0] == "cen":                      # cluster inspection (Table 9.4)
            act = be != 0
            sizes, corr = [], []
            for k in np.unique(lab):
                mem = np.where((lab == k) & act)[0]
                sizes.append(int(len(mem)))
                if len(mem) > 1:
                    R = np.corrcoef(Xs[:, mem].T); corr.append(float(np.abs(R[np.triu_indices(len(mem), 1)]).mean()))
            r["cluster_sizes_active"] = str(sizes); r["mean_abs_corr_within"] = float(np.mean(corr)) if corr else np.nan
        rows.append(r)
        if fam == "CEN-multi(K tuned)":                              # convergence log for Fig. 9.1
            Kc, lam, i = key[1], key[2], key[3]
            b0e, bee, _, dv = res[("enet", lam, i)]
            *_, hist = fit_cen(Xs, ytr, lam, dv, Kc, b0e, bee, cfg, seed=seed, log=True)
    print("   [%s] done in %.0fs" % (tag, time.time() - t0), flush=True)
    return rows, hist


# =====================================================================================
# 6. SELF TESTS  (guideline Section 5.2 adapted)
# =====================================================================================
def selftest():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(1)
    cfg = dict(PRESETS["quick"]); cfg.update(COMMON); cfg.update(tol=1e-10, max_pass=20000)
    n, p = 100, 30
    Z = rng.standard_normal((n, p)); base = rng.standard_normal((n, 3))
    for b in range(3): Z[:, 5 * b:5 * b + 5] += 1.5 * base[:, [b]]
    eta = 1.0 * Z[:, :5].sum(1) - 1.0 * Z[:, 5:10].sum(1) + 0.5 * Z[:, 10:15].sum(1)
    y = (rng.random(n) < sigmoid(eta / 2)).astype(float)
    X = (Z - Z.mean(0)) / Z.std(0); X = np.asfortranarray(X)
    print("T1 standardisation: max|mean| = %.1e, max|(1/n)||Xj||^2-1| = %.1e" %
          (np.abs(X.mean(0)).max(), np.abs((X ** 2).mean(0) - 1).max()))
    # T2 gradient identity (finite differences)
    lab = np.repeat(np.arange(3), 10).astype(np.int64); lam = 0.7
    beta = rng.standard_normal(p); j = 4; k = lab[j]
    f = X * beta; fbar = f[:, lab == k].mean(1)
    g_mf = 2 * lam * 1.0 * beta[j] - 2 * lam * (X[:, j] @ fbar) / n
    e = 1e-6; bp, bm = beta.copy(), beta.copy(); bp[j] += e; bm[j] -= e
    g_fd = (cluster_penalty(X, bp, lab, lam) - cluster_penalty(X, bm, lab, lam)) / (2 * e)
    print("T2 centroid-gradient identity: |matrix-free - finite diff| = %.2e" % abs(g_mf - g_fd))
    # T3 lambda=0 vs sklearn L1 ; T3b EN vs sklearn
    d = 0.02; z = np.zeros(p); lab1 = np.zeros(p, dtype=np.int64)
    b0, be, _ = cd_run(X, y, 0.0, z, lab1, 1, 0.0, d, 0.0, 1e-14, 200000)
    sk = LogisticRegression(penalty="l1", C=1 / (n * d), solver="saga", tol=1e-12, max_iter=200000).fit(X, y)
    print("T3 Lasso (lam=0) vs sklearn: max|beta diff| = %.2e" % np.abs(be - sk.coef_.ravel()).max())
    l1r, C = 0.5, 1.0
    b0, be, _ = cd_run(X, y, 0.0, z, lab1, 1, 0.0, l1r / (C * n), (1 - l1r) / (C * n), 1e-14, 200000)
    sk = LogisticRegression(penalty="elasticnet", l1_ratio=l1r, C=C, solver="saga", tol=1e-12, max_iter=200000).fit(X, y)
    print("T3b ElasticNet vs sklearn: max|beta diff| = %.2e" % np.abs(be - sk.coef_.ravel()).max())
    # T4 monotonicity: every single CD pass (K=3 fixed partition) and every outer step
    b0, be = 0.0, np.zeros(p); prev = np.inf; ok = True
    for _ in range(60):
        b0, be, _ = cd_run(X, y, b0, be, lab, 3, 0.5, 0.01, 0.0, 0.0, 1)
        F = objective(X, y, b0, be, lab, 0.5, 0.01)["total"]; ok &= F <= prev + 1e-12; prev = F
    print("T4a every CD pass decreases F (fixed partition): %s" % ok)
    b0e, bee, _ = cd_run(X, y, 0.0, z, lab1, 1, 0.0, 0.01, 0.5, 1e-10, 100000)
    *_, hist = fit_cen(X, y, 0.5, 0.01, 3, b0e, bee, cfg, seed=0, log=True)
    F = [h["obj_total"] for h in hist]
    print("T4b outer iterations: %d, F non-increasing: %s, final F = %.6f" % (len(F), bool(np.all(np.diff(F) <= 1e-10)), F[-1]))
    # T5 convexity sanity: K=1 CEN from two different starts -> same minimiser
    a = cd_run(X, y, 0.0, z, lab1, 1, 0.5, 0.01, 0.0, 1e-14, 200000)[1]
    b = cd_run(X, y, 1.0, rng.standard_normal(p) * 0.3, lab1, 1, 0.5, 0.01, 0.0, 1e-14, 200000)[1]
    print("T5 K=1 CEN unique minimiser (two starts): max|diff| = %.2e" % np.abs(a - b).max())
    # T6 parser/screening on a small DOROTHEA-format file
    S = sp.random(120, 500, density=0.05, format="csr", random_state=3); S.data[:] = 1
    yy = (rng.random(120) < 0.3).astype(float)
    Xs2, (Xt2,), idx = screen_and_standardize(S[:80], yy[:80], [S[80:]], 50, 3)
    print("T6 screening: shape %s, train mean max %.1e, train sd range [%.3f, %.3f]" %
          (Xs2.shape, np.abs(Xs2.mean(0)).max(), Xs2.std(0).min(), Xs2.std(0).max()))


# =====================================================================================
# 7. ADDITIONAL DOROTHEA PLOTS  (R2 ★, R8 ★, R5 ★, R4, R11, R15)
# =====================================================================================
def make_additional_dorothea_plots(df_folds, ext_rows, X_train, y_train, cfg, out_dir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    order_all = ["Lasso", "Ridge", "ElasticNet", "CEN-K1", "CEN-K2", "CEN-K3", "CEN-K5", "CEN-multi(K tuned)"]
    fams_present = [f for f in order_all if f in df_folds["family"].unique()]

    # --- R5 ★ : Per-fold paired difference (CEN-multi - ElasticNet) ---
    if "CEN-multi(K tuned)" in fams_present and "ElasticNet" in fams_present:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for ax, (m, title) in zip(axes, [("ROC_AUC", "ROC-AUC (higher is better)"),
                                         ("PR_AUC", "PR-AUC (higher is better)"),
                                         ("Brier", "Brier Score (lower is better)")]):
            piv = df_folds.pivot(index="tag", columns="family", values=m)
            diff = piv["CEN-multi(K tuned)"] - piv["ElasticNet"]
            x = np.arange(len(diff))
            colors = ["#2ca02c" if (v > 0 and m != "Brier") or (v < 0 and m == "Brier") else "#d62728" for v in diff]
            ax.bar(x, diff.values, color=colors, alpha=0.8, edgecolor="k", width=0.55)
            ax.axhline(0, color="k", linestyle="--", linewidth=1)
            ax.axhline(diff.mean(), color="#1f77b4", linestyle=":", linewidth=1.5, label="Mean diff = %+.4f" % diff.mean())
            ax.set_xticks(x); ax.set_xticklabels(diff.index, rotation=20)
            ax.set_title(title, fontsize=10); ax.set_ylabel("CEN-multi − ElasticNet")
            ax.legend(fontsize=8)
        fig.suptitle("R5: Per-Fold Paired Difference (CEN-multi minus Elastic Net)", fontsize=11, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_R5_paired_diff.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    # --- R4 : Per-fold boxplot/dot plot (ROC-AUC, PR-AUC, BER) ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, m in zip(axes, ["ROC_AUC", "PR_AUC", "BER"]):
        data = [df_folds[df_folds["family"] == f][m].dropna().values for f in fams_present]
        bp = ax.boxplot(data, patch_artist=True, labels=fams_present, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor("#cfe2f3"); patch.set_alpha(0.8)
        for i, vals in enumerate(data):
            jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else [0]
            ax.plot(np.full(len(vals), i + 1) + jitter, vals, "ko", markersize=4, alpha=0.7)
        ax.set_xticklabels(fams_present, rotation=35, ha="right", fontsize=8)
        ax.set_title("Per-fold %s" % m, fontsize=10); ax.set_ylabel(m)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_R4_per_fold_boxplot.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- R11 : Sparsity vs Performance Scatter ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    g_mean = df_folds.groupby("family")[["n_selected", "ROC_AUC", "PR_AUC"]].mean().reindex(fams_present)
    g_sd = df_folds.groupby("family")[["n_selected", "ROC_AUC", "PR_AUC"]].std().reindex(fams_present)
    for ax, m in zip(axes, ["ROC_AUC", "PR_AUC"]):
        for f in fams_present:
            ax.errorbar(g_mean.loc[f, "n_selected"], g_mean.loc[f, m],
                        xerr=g_sd.loc[f, "n_selected"], yerr=g_sd.loc[f, m],
                        fmt="o", markersize=7, capsize=3, label=f)
        ax.set_xlabel("Number of Selected Features (mean ± sd)")
        ax.set_ylabel("%s (mean ± sd)" % m)
        ax.set_title("Sparsity vs %s (Outer CV)" % m, fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_R11_sparsity_vs_perf.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- External Validation Plots: R2 ★ (PR Curve), R8 ★ (Calibration), R15 (Cluster Corr Heatmap) ---
    if ext_rows:
        ext_map = {r["family"]: r for r in ext_rows}
        key_fams = [f for f in ["Lasso", "Ridge", "ElasticNet", "CEN-K1", "CEN-multi(K tuned)"] if f in ext_map]

        # R2 ★ : PR Curve on External Validation Set
        fig, ax = plt.subplots(figsize=(6, 4.5))
        y_ext = ext_rows[0]["y_test"]
        prev = float(np.mean(y_ext))
        for f in key_fams:
            r = ext_map[f]
            pr, rc, _ = precision_recall_curve(r["y_test"], r["p_test"])
            lw = 2.2 if f == "CEN-multi(K tuned)" else 1.4
            ax.plot(rc, pr, linewidth=lw, label="%s (AP = %.4f)" % (f, r["PR_AUC"]))
        ax.axhline(prev, color="k", linestyle=":", label="Baseline prevalence (%.2f)" % prev)
        ax.set_xlabel("Recall (Sensitivity)"); ax.set_ylabel("Precision (PPV)")
        ax.set_title("R2: Precision-Recall Curve (External Validation, n=%d)" % len(y_ext), fontsize=10)
        ax.legend(fontsize=8, loc="upper right"); ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_R2_pr_external.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

        # R8 ★ : Calibration (Reliability) Diagram on External Validation Set
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 6.2), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
        ax1.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        for f in [fm for fm in ["Lasso", "ElasticNet", "CEN-K1", "CEN-multi(K tuned)"] if fm in ext_map]:
            r = ext_map[f]
            prob_true, prob_pred = calibration_curve(r["y_test"], r["p_test"], n_bins=8, strategy="quantile")
            lw = 2.2 if f == "CEN-multi(K tuned)" else 1.4
            ax1.plot(prob_pred, prob_true, "o-", linewidth=lw, label="%s (Brier = %.4f)" % (f, r["Brier"]))
            ax2.hist(r["p_test"], bins=20, range=(0, 1), histtype="step", linewidth=1.2, label=f)
        ax1.set_ylabel("Observed Fraction of Positives")
        ax1.set_title("R8: Calibration Diagram (External Validation Set)", fontsize=10)
        ax1.legend(fontsize=8, loc="upper left"); ax1.grid(True, linestyle=":", alpha=0.6)
        ax2.set_xlabel("Mean Predicted Probability"); ax2.set_ylabel("Count")
        ax2.set_yscale("log"); ax2.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_R8_calibration_external.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

        # R15 : Cluster-Ordered Correlation Heatmap (active features of CEN-multi)
        if "CEN-multi(K tuned)" in ext_map:
            rcen = ext_map["CEN-multi(K tuned)"]
            lab = rcen.get("labels")
            be = rcen.get("beta")
            if lab is not None and be is not None and np.any(be != 0):
                Xs, _, _ = screen_and_standardize(X_train, y_train, [], cfg["m"], cfg["min_nnz"])
                act = np.where(be != 0)[0]
                # If many active features, keep top 150 by |beta_j| per cluster order for clean visual
                if len(act) > 150:
                    top_sub = np.argsort(-np.abs(be[act]))[:150]
                    act = act[top_sub]
                order_idx = np.argsort(lab[act])
                act_ord = act[order_idx]
                lab_ord = lab[act_ord]
                C = np.abs(np.corrcoef(Xs[:, act_ord].T))
                fig, ax = plt.subplots(figsize=(6.5, 5.5))
                im = ax.imshow(C, cmap="viridis", vmin=0, vmax=1, aspect="auto")
                # Draw cluster boundaries
                bounds = np.where(np.diff(lab_ord) != 0)[0] + 0.5
                for b in bounds:
                    ax.axhline(b, color="white", linestyle="--", linewidth=1.2)
                    ax.axvline(b, color="white", linestyle="--", linewidth=1.2)
                plt.colorbar(im, ax=ax, label="|Correlation|")
                ax.set_title("R15: Cluster-Ordered |Correlation| Heatmap (Active Features)", fontsize=10)
                ax.set_xlabel("Active features ordered by CEN cluster")
                ax.set_ylabel("Active features ordered by CEN cluster")
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, "fig_R15_cluster_corr_heatmap.png"), dpi=200, bbox_inches="tight")
                plt.close(fig)


# =====================================================================================
# 8. MAIN
# =====================================================================================
def jaccard_mean(sets):
    v = [len(set(a) & set(b)) / max(len(set(a) | set(b)), 1) for i, a in enumerate(sets) for b in sets[i + 1:]]
    return float(np.mean(v)) if v else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="quick", choices=list(PRESETS))
    ap.add_argument("--m", type=int, default=None, help="number of screened features (e.g. 1000 or 5000)")
    ap.add_argument("--n_jobs", type=int, default=1, help="parallel outer folds")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--skip_cv", action="store_true", help="reuse existing per_fold_results.csv and only run external valid")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    cfg = dict(PRESETS[a.preset]); cfg.update(COMMON)
    if a.m: cfg["m"] = a.m
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading data ..."); X, y = load_dorothea(TRAIN_DATA, TRAIN_LABELS)
    print("n=%d p=%d positives=%d (%.2f%%)  config=%s" % (X.shape[0], X.shape[1], y.sum(), 100 * y.mean(), cfg))

    metrics = ["ROC_AUC", "PR_AUC", "BalAcc", "BER", "Sens", "Spec", "F1", "Brier", "n_selected", "train_sec"]
    order = ["Lasso", "Ridge", "ElasticNet", "CEN-K1", "CEN-K2", "CEN-K3", "CEN-K5", "CEN-multi(K tuned)"]
    per_fold_csv = os.path.join(OUT_DIR, "per_fold_results.csv")

    rows, hist = [], None
    if a.skip_cv and os.path.exists(per_fold_csv):
        print("Reusing existing outer-fold CV results from:", per_fold_csv)
        df = pd.read_csv(per_fold_csv)
    else:
        outer = list(StratifiedKFold(cfg["outer_folds"], shuffle=True, random_state=cfg["seed"]).split(np.zeros(len(y)), y))
        jobs = [(X[tr], y[tr], X[te], y[te], cfg, cfg["seed"] + f, "outer%d" % (f + 1)) for f, (tr, te) in enumerate(outer)]
        if a.n_jobs > 1:
            from joblib import Parallel, delayed
            outs = Parallel(n_jobs=a.n_jobs)(delayed(run_pipeline)(*j) for j in jobs)
        else:
            outs = [run_pipeline(*j) for j in jobs]
        rows = [r for o in outs for r in o[0]]
        hist = outs[0][1]

        df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("sel_idx", "y_test", "p_test", "beta", "labels", "idx")} for r in rows])
        df.to_csv(per_fold_csv, index=False)
        g = df.groupby("family")[metrics]
        summ = (g.mean().round(4).astype(str) + " ± " + g.std().round(4).astype(str)).reindex(order)
        summ.to_csv(os.path.join(OUT_DIR, "table_9_3_nested_cv_summary.csv"))
        print("\n=== Table 9.3 : outer-fold CV, mean ± sd ===\n", summ.to_string())

        stab = []                                                        # Table 9.4
        for fam in order:
            sets = [r["sel_idx"] for r in rows if r["family"] == fam]
            extra = df[(df.family == fam) & df.get("cluster_sizes_active", pd.Series(dtype=str)).notna()] if "cluster_sizes_active" in df else pd.DataFrame()
            stab.append(dict(family=fam, mean_pairwise_jaccard=round(jaccard_mean(sets), 4),
                             mean_n_selected=round(np.mean([len(s) for s in sets]), 1) if sets else np.nan,
                             cluster_sizes_active_fold1=extra["cluster_sizes_active"].iloc[0] if len(extra) else "",
                             mean_abs_corr_within=round(float(extra["mean_abs_corr_within"].mean()), 4) if len(extra) else ""))
        st = pd.DataFrame(stab); st.to_csv(os.path.join(OUT_DIR, "table_9_4_stability_clusters.csv"), index=False)
        print("\n=== Table 9.4 : stability / clusters ===\n", st.to_string(index=False))

    ext_rows = []
    if os.path.exists(VALID_DATA) and os.path.exists(VALID_LABELS):
        print("External validation set found -> evaluated ONCE (tuning on the 800 training compounds only).")
        Xv, yv = load_dorothea(VALID_DATA, VALID_LABELS)
        ext_rows, hist_ext = run_pipeline(X, y, Xv, yv, cfg, cfg["seed"] + 99, "external")
        hist = hist_ext if hist_ext is not None else hist
    else:
        print("(no dorothea_valid.* next to the train files -> only nested CV on the 800 training compounds)")

    if ext_rows:
        de = pd.DataFrame([{k: v for k, v in r.items() if k not in ("sel_idx", "y_test", "p_test", "beta", "labels", "idx")} for r in ext_rows])
        de = de.set_index("family").reindex(order)
        de.to_csv(os.path.join(OUT_DIR, "table_9_3_external_valid.csv"))
        print("\n=== External test (dorothea_valid, n=350) ===\n", de[metrics + ["tau"]].round(4).to_string())

        # Save external predicted probabilities and cluster assignments to CSV for future reuse
        pred_df = pd.DataFrame({"y_true": ext_rows[0]["y_test"]})
        for r in ext_rows:
            pred_df[r["family"]] = r["p_test"]
        pred_df.to_csv(os.path.join(OUT_DIR, "external_predictions.csv"), index=False)

        for r in ext_rows:
            if r["family"] == "CEN-multi(K tuned)" and r.get("labels") is not None:
                cl_df = pd.DataFrame({
                    "feature_orig_idx": r["idx"],
                    "cluster_label": r["labels"],
                    "beta": r["beta"],
                    "active": (r["beta"] != 0).astype(int)
                })
                cl_df.to_csv(os.path.join(OUT_DIR, "external_clusters_cen_multi.csv"), index=False)

    if hist:
        pd.DataFrame(hist).to_csv(os.path.join(OUT_DIR, "cen_iteration_log.csv"), index=False)
    try:                                                             # figures
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        if hist:
            h = pd.DataFrame(hist); fig, ax = plt.subplots(1, 4, figsize=(15, 3.2))
            for a_, c, t in zip(ax, ["obj_total", "loss_logistic", "pen_cluster", "pen_l1"],
                                ["F (total)", "Logistic loss L", "P_cluster", "P_L1"]):
                a_.plot(h["iter"], h[c], "o-"); a_.set_title(t); a_.set_xlabel("outer iteration t")
            fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig_9_1_convergence.png"), dpi=200); plt.close(fig)
        if rows:
            fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
            for fam in ["Lasso", "ElasticNet", "CEN-K1", "CEN-multi(K tuned)"]:
                yy = np.concatenate([r["y_test"] for r in rows if r["family"] == fam])
                pp = np.concatenate([r["p_test"] for r in rows if r["family"] == fam])
                fpr, tpr, _ = roc_curve(yy, pp); pr, rc, _ = precision_recall_curve(yy, pp)
                ax[0].plot(fpr, tpr, label=fam); ax[1].plot(rc, pr, label=fam)
            ax[0].plot([0, 1], [0, 1], "k:"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].set_title("ROC (pooled outer folds)")
            ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("PR (pooled outer folds)"); ax[0].legend()
            fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig_roc_pr.png"), dpi=200); plt.close(fig)

        # Generate the 6 new DOROTHEA figures (R2, R8, R5, R4, R11, R15)
        make_additional_dorothea_plots(df, ext_rows, X, y, cfg, OUT_DIR)
    except Exception as e:
        print("plot skipped:", e)
    print("\nAll outputs saved in:", OUT_DIR)


if __name__ == "__main__":
    main()