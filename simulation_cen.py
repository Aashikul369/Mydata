# -*- coding: utf-8 -*-
"""
Simulation study for Logistic CEN  (guideline Chapter 6, Scenarios 1-4)
Put this file in the SAME folder as logistic_cen_dorothea.py (it imports the solver from there).

  python simulation_cen.py --preset tiny  --R 3                  # smoke test
  python simulation_cen.py --preset quick --R 30                 # real run (p=10000, uses ~90% of cores)
  python simulation_cen.py --preset quick --R 30 --plots_only    # instant plots + CSV data from existing CSVs (0s rerun)

Outputs (folder  sim_results_p<p>/ next to this file):
  S{k}_per_rep.csv, S{k}_summary.csv        : one pair per scenario (all its settings)
  sim_datasets/*.csv                        : saved simulated datasets in .csv format
  ALL_comparison_roc_auc.csv, ALL_comparison_pr_auc.csv
  verdicts.csv + scenario_notes.txt         : for every setting -> CEN is BETTER / SIMILAR / WORSE
  fig_sim_roc_auc.png + 6 new graphs (S1, S2, S3, S5, S8, S9)
"""
import os, sys, time, re, argparse, warnings, io, contextlib
# one thread per worker, so that N parallel workers do not fight for the same cores (big speed-up)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count())
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")


def _silence_loky():
    """removes the 'wmic / physical cores' traceback on Windows"""
    try:
        from joblib.externals.loky.backend import context as _c
        n = os.cpu_count() or 1
        if hasattr(_c, "physical_cores_cache"):
            _c.physical_cores_cache = n
        _c._count_physical_cores = lambda: (n, None)
    except Exception:
        pass


_silence_loky()
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy import stats
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logistic_cen_dorothea as cen

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_results")
ORDER = ["Lasso", "Ridge", "ElasticNet", "CEN-K1", "CEN-K2", "CEN-K3", "CEN-K5", "CEN-multi(K tuned)"]
PROPOSED = "CEN-multi(K tuned)"
N_TRAIN, N_TEST = 200, 1000

# (scenario, label, params)      -- fixed in advance, as in guideline Section 6.2
SETTINGS = (
    [(1, "rho=%.1f, b=%.1f" % (r, b), dict(rho=r, b=b)) for r in (0.5, 0.8) for b in (0.3, 0.6)] +
    [(2, "rho=0.8, b=%.1f" % b, dict(rho=0.8, b=b)) for b in (0.3, 0.6)] +
    [(3, "Sigma=I", dict(ar=0.0)), (3, "Sigma=AR(0.3)", dict(ar=0.3))] +
    [(4, "rho=0.8, b=%.1f, prev~10%%" % b, dict(rho=0.8, b=b)) for b in (0.3, 0.6)]
)


# ------------------------------------------------------------------ data generation
def _block(rng, n, size, rho):
    f = rng.standard_normal((n, 1))
    return np.sqrt(rho) * f + np.sqrt(1 - rho) * rng.standard_normal((n, size))


def true_model(scn, par, p, setting_seed):
    """true beta, true block id of each active feature (-1 = none)."""
    beta = np.zeros(p); blk = -np.ones(p, dtype=int)
    if scn in (1, 2, 4):
        b = par["b"]
        for g, c in enumerate([b, -b, 0.5 * b]):
            sl = slice(10 * g, 10 * g + 10); beta[sl] = c; blk[sl] = g
        if scn == 2:                      # second half of each block: negatively correlated, opposite sign
            for g in range(3):
                beta[10 * g + 5:10 * g + 10] *= -1
    else:                                 # scenario 3: 30 sparse heterogeneous signals, fixed positions per setting
        r = np.random.default_rng(setting_seed)
        act = r.choice(p, 30, replace=False)
        beta[act] = r.uniform(0.2, 0.8, 30)
    return beta, blk


def gen_data(scn, par, p, beta, n, rng):
    if scn == 3:
        phi = par["ar"]
        Z = np.empty((n, p)); Z[:, 0] = rng.standard_normal(n)
        for j in range(1, p):
            Z[:, j] = phi * Z[:, j - 1] + np.sqrt(1 - phi ** 2) * rng.standard_normal(n)
    else:
        Z = rng.standard_normal((n, p))
        for g in range(3):
            Z[:, 10 * g:10 * g + 10] = _block(rng, n, 10, par["rho"])
        if scn == 2:
            for g in range(3):
                Z[:, 10 * g + 5:10 * g + 10] *= -1
    eta = Z @ beta
    b0 = 0.0
    if scn == 4:                          # calibrate intercept so that prevalence ~ 10%
        lo, hi = -15.0, 5.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if np.mean(cen.sigmoid(mid + eta)) > 0.10: hi = mid
            else: lo = mid
        b0 = 0.5 * (lo + hi)
    y = (rng.random(n) < cen.sigmoid(b0 + eta)).astype(float)
    return Z, y


def save_simulated_dataset_csv(Z, y, beta, blk, scn, label, rep, cfg_m, out_dir):
    """Saves generated dataset to CSV in sim_datasets/ (all active features + top screened features)."""
    ds_dir = os.path.join(out_dir, "sim_datasets")
    os.makedirs(ds_dir, exist_ok=True)
    safe_lbl = re.sub(r"[^a-zA-Z0-9_.-]+", "_", label)
    act_idx = np.where(beta != 0)[0]
    # Keep all 30 true active features + top screened features up to cfg_m columns so CSV is compact & complete
    ytr = y[:N_TRAIN]
    cov = np.abs((Z[:N_TRAIN] - Z[:N_TRAIN].mean(0)).T @ (ytr - ytr.mean()))
    top_m = np.argsort(-cov)[:cfg_m]
    keep_cols = np.unique(np.concatenate([act_idx, top_m]))
    df_data = pd.DataFrame(np.round(Z[:, keep_cols], 5), columns=["X_%d" % j for j in keep_cols])
    df_data.insert(0, "y", y.astype(int))
    df_data.insert(0, "split", ["train"] * N_TRAIN + ["test"] * N_TEST)
    df_data.to_csv(os.path.join(ds_dir, "S%d_%s_rep%d_data.csv" % (scn, safe_lbl, rep)), index=False)

    if rep == 0:
        df_meta = pd.DataFrame({
            "feature": ["X_%d" % j for j in keep_cols],
            "orig_idx": keep_cols,
            "true_beta": beta[keep_cols],
            "true_block": blk[keep_cols],
            "is_active": (beta[keep_cols] != 0).astype(int)
        })
        df_meta.to_csv(os.path.join(ds_dir, "S%d_%s_true_model.csv" % (scn, safe_lbl)), index=False)


# ------------------------------------------------------------------ one Monte-Carlo replicate
def one_rep(scn, label, par, p, rep, cfg, base_seed, setting_seed, out_dir="", save_data_reps=1):
    warnings.filterwarnings("ignore"); _silence_loky()
    rng = np.random.default_rng(base_seed + 1000 * rep + scn)
    beta, blk = true_model(scn, par, p, setting_seed)
    while True:
        Z, y = gen_data(scn, par, p, beta, N_TRAIN + N_TEST, rng)
        if y[:N_TRAIN].sum() >= 8 and (1 - y[:N_TRAIN]).sum() >= 8 and 0 < y[N_TRAIN:].sum() < N_TEST: break
    if out_dir and rep < save_data_reps:
        save_simulated_dataset_csv(Z, y, beta, blk, scn, label, rep, cfg["m"], out_dir)
    Xtr, Xte = sp.csr_matrix(Z[:N_TRAIN]), sp.csc_matrix(Z[N_TRAIN:])
    del Z
    with contextlib.redirect_stdout(io.StringIO()):
        rows, _ = cen.run_pipeline(Xtr, y[:N_TRAIN], Xte, y[N_TRAIN:], cfg, base_seed + rep, tag="")
    A = set(np.where(beta != 0)[0].tolist())
    out = []
    for r in rows:
        sel = set(r["sel_idx"].tolist())
        ari = np.nan
        lab, idx = r.get("labels"), r.get("idx")
        if lab is not None and blk.max() >= 0 and r["family"] not in ("CEN-K1",) and idx is not None:
            act = np.where(blk >= 0)[0]
            pos = np.searchsorted(idx, act)
            okk = (pos < len(idx)) & (idx[np.minimum(pos, len(idx) - 1)] == act)
            if okk.sum() > 1:
                ari = adjusted_rand_score(blk[act][okk], lab[pos[okk]])
        out.append(dict(scenario=scn, setting=label, rep=rep, family=r["family"],
                        ROC_AUC=r["ROC_AUC"], PR_AUC=r["PR_AUC"], BalAcc=r["BalAcc"], Brier=r["Brier"],
                        n_selected=r["n_selected"], TPR=len(sel & A) / len(A),
                        FDR=len(sel - A) / max(len(sel), 1), ARI=ari, hyper=r["hyper"],
                        sel=",".join(map(str, sorted(sel)))))
    return out


# ------------------------------------------------------------------ summaries
def jacc(sets):
    v = [len(a & b) / max(len(a | b), 1) for i, a in enumerate(sets) for b in sets[i + 1:]]
    return float(np.mean(v)) if v else np.nan


def summarize(df):
    rows = []
    for (st, fam), g in df.groupby(["setting", "family"]):
        sets = [set(map(int, str(s).split(","))) if pd.notna(s) and str(s) else set() for s in g["sel"]] if "sel" in g else []
        d = dict(setting=st, family=fam)
        for m in ["ROC_AUC", "PR_AUC", "BalAcc", "Brier", "n_selected", "TPR", "FDR", "ARI"]:
            v = g[m].dropna()
            d[m] = "%.4f ± %.4f" % (v.mean(), v.std()) if len(v) else "-"
        d["Jaccard"] = round(jacc(sets), 4) if sets else np.nan
        rows.append(d)
    s = pd.DataFrame(rows)
    s["family"] = pd.Categorical(s["family"], ORDER, ordered=True)
    return s.sort_values(["setting", "family"]).reset_index(drop=True)


METRICS = {  # metric: (higher is better, practical margin)
    "ROC_AUC": (True, 0.005), "PR_AUC": (True, 0.01), "Brier": (False, 0.002), "TPR": (True, 0.02), "FDR": (False, 0.02)}


def verdict(a, b, hib, margin):
    """a = proposed, b = competitor (paired by replicate)."""
    d = (a - b).values
    mean = float(np.mean(d))
    p = 1.0 if np.allclose(d, 0) else float(stats.ttest_rel(a, b).pvalue)
    s = mean if hib else -mean
    v = "BETTER" if (p < 0.05 and s >= margin) else "WORSE" if (p < 0.05 and s <= -margin) else "SIMILAR"
    return mean, p, v


def make_verdicts(df):
    out = []
    for (scn, st), g in df.groupby(["scenario", "setting"]):
        W = {m: g.pivot(index="rep", columns="family", values=m) for m in METRICS}
        for comp in ["ElasticNet", "Lasso", "Ridge", "CEN-K1"]:
            for m, (hib, mar) in METRICS.items():
                if PROPOSED not in W[m] or comp not in W[m]: continue
                mean, p, v = verdict(W[m][PROPOSED], W[m][comp], hib, mar)
                out.append(dict(scenario=scn, setting=st, competitor=comp, metric=m,
                                proposed=W[m][PROPOSED].mean(), competitor_value=W[m][comp].mean(),
                                diff=mean, p_value=p, verdict=v))
    return pd.DataFrame(out)


def write_notes(vd, path):
    L = ["NOTES: proposed = CEN-multi (K tuned).  BETTER/WORSE = paired t-test p<0.05 AND |diff| >= practical margin",
         "(ROC-AUC 0.005, PR-AUC 0.01, Brier 0.002, TPR 0.02, FDR 0.02); otherwise SIMILAR.  Lower is better for Brier, FDR.", ""]
    for scn in sorted(vd.scenario.unique()):
        L.append("=" * 90); L.append("SCENARIO %d" % scn)
        for st, g in vd[vd.scenario == scn].groupby("setting", sort=False):
            L.append("  [%s]" % st)
            for comp in ["ElasticNet", "Lasso", "Ridge", "CEN-K1"]:
                h = g[g.competitor == comp]
                parts = ["%s %s (%+.4f, p=%.3f)" % (r.metric, r.verdict, r["diff"], r.p_value) for _, r in h.iterrows()]
                L.append("     vs %-10s : %s" % (comp, " | ".join(parts)))
            h = g[(g.competitor == "ElasticNet") & g.metric.isin(["ROC_AUC", "PR_AUC"])]
            vs = list(h.verdict)
            tag = ("CEN BETTER than Elastic Net" if vs.count("BETTER") == len(vs) else
                   "CEN WORSE than Elastic Net" if vs.count("WORSE") == len(vs) else
                   "CEN mostly BETTER (mixed)" if vs.count("BETTER") > vs.count("WORSE") else
                   "CEN mostly WORSE (mixed)" if vs.count("WORSE") > vs.count("BETTER") else
                   "CEN SIMILAR to Elastic Net")
            L.append("     ==> prediction (ROC-AUC & PR-AUC): %s" % tag)
        sub = vd[(vd.scenario == scn) & (vd.competitor == "ElasticNet") & (vd.metric == "ROC_AUC")]
        L.append("  Scenario %d summary vs Elastic Net (ROC-AUC) over %d setting(s): BETTER=%d, SIMILAR=%d, WORSE=%d" %
                 (scn, len(sub), (sub.verdict == "BETTER").sum(), (sub.verdict == "SIMILAR").sum(), (sub.verdict == "WORSE").sum()))
        L.append("")
    open(path, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


# ------------------------------------------------------------------ 6 NEW SIMULATION PLOTS (S1, S2, S3, S5, S8, S9)
def make_additional_sim_plots(df, vd, out_dir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    df = df.copy()
    df["scn_setting"] = "S" + df["scenario"].astype(str) + ": " + df["setting"].astype(str)
    settings_ordered = list(dict.fromkeys(df["scn_setting"]))
    fams_box = [f for f in ["Lasso", "Ridge", "ElasticNet", "CEN-K1", PROPOSED] if f in df["family"].unique()]

    # --- S1 : Boxplot of ROC-AUC and PR-AUC across all settings ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True)
    colors = ["#f4cccc", "#d9d2e9", "#fff2cc", "#d0e0e3", "#b6d7a8"]
    for ax, m in zip(axes, ["ROC_AUC", "PR_AUC"]):
        n_f = len(fams_box)
        width = 0.15
        for i, f in enumerate(fams_box):
            vals = [df[(df["scn_setting"] == st) & (df["family"] == f)][m].dropna().values for st in settings_ordered]
            pos = np.arange(len(settings_ordered)) + (i - (n_f - 1) / 2) * width
            bp = ax.boxplot(vals, positions=pos, widths=width * 0.85, patch_artist=True, showfliers=False)
            for patch in bp["boxes"]:
                patch.set_facecolor(colors[i % len(colors)])
            ax.plot([], [], color=colors[i % len(colors)], linewidth=6, label=f)
        ax.set_ylabel(m); ax.set_title("S1: Distribution of Test %s across Simulation Settings" % m, fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        ax.legend(fontsize=8, loc="lower left", ncol=n_f)
    axes[1].set_xticks(np.arange(len(settings_ordered)))
    axes[1].set_xticklabels(settings_ordered, rotation=28, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_S1_boxplots.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- S2 ★ : Forest plot of Paired Difference (CEN-multi - ElasticNet, 95% CI) ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    y_pos = np.arange(len(settings_ordered))
    for ax, (m, title) in zip(axes, [("ROC_AUC", "ROC-AUC Diff (CEN − EN)"),
                                     ("PR_AUC", "PR-AUC Diff (CEN − EN)"),
                                     ("Brier", "Brier Diff (CEN − EN, <0 is better)")]):
        means, cis = [], []
        for st in settings_ordered:
            sub = df[df["scn_setting"] == st].pivot(index="rep", columns="family", values=m)
            d = (sub[PROPOSED] - sub["ElasticNet"]).dropna().values
            mu = float(np.mean(d))
            se = float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
            tval = float(stats.t.ppf(0.975, max(len(d) - 1, 1)))
            means.append(mu); cis.append(tval * se)
        ax.errorbar(means, y_pos, xerr=cis, fmt="o", color="#1f77b4", ecolor="#333333", elinewidth=1.5, capsize=3)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(title, fontsize=9.5); ax.set_xlabel("Mean Difference (95% CI)")
        ax.grid(axis="x", linestyle=":", alpha=0.6)
    axes[0].set_yticks(y_pos); axes[0].set_yticklabels(settings_ordered, fontsize=8.5)
    axes[0].invert_yaxis()
    fig.suptitle("S2: Forest Plot of Paired Differences (CEN-multi minus Elastic Net)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_S2_forest_plot.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- S3 ★ : Verdict Heatmap (Setting x Metric vs ElasticNet) ---
    vd_en = vd[vd["competitor"] == "ElasticNet"].copy()
    vd_en["scn_setting"] = "S" + vd_en["scenario"].astype(str) + ": " + vd_en["setting"].astype(str)
    m_list = ["ROC_AUC", "PR_AUC", "Brier", "TPR", "FDR"]
    v_map = {"WORSE": -1, "SIMILAR": 0, "BETTER": 1}
    mat = np.zeros((len(settings_ordered), len(m_list)))
    txt = [["" for _ in m_list] for _ in settings_ordered]
    for i, st in enumerate(settings_ordered):
        for j, m in enumerate(m_list):
            row = vd_en[(vd_en["scn_setting"] == st) & (vd_en["metric"] == m)]
            if len(row):
                r0 = row.iloc[0]
                mat[i, j] = v_map.get(r0["verdict"], 0)
                txt[i][j] = "%s\n(%+.3f)" % (r0["verdict"], r0["diff"])
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    cmap = ListedColormap(["#f4cccc", "#efefef", "#b6d7a8"])
    ax.imshow(mat, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    for i in range(len(settings_ordered)):
        for j in range(len(m_list)):
            ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=7.5)
    ax.set_xticks(np.arange(len(m_list))); ax.set_xticklabels(m_list, fontsize=9)
    ax.set_yticks(np.arange(len(settings_ordered))); ax.set_yticklabels(settings_ordered, fontsize=8.5)
    ax.set_title("S3: Verdict Heatmap — CEN-multi(K tuned) vs Elastic Net", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_S3_verdict_heatmap.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- S5 ★ : TPR vs FDR Scatter across Scenarios ---
    scns = sorted(df["scenario"].unique())
    fig, axes = plt.subplots(1, len(scns), figsize=(4.2 * len(scns), 3.8), squeeze=False)
    fams_sc = [f for f in ["Lasso", "Ridge", "ElasticNet", "CEN-K1", "CEN-K3", PROPOSED] if f in df["family"].unique()]
    markers = ["o", "s", "^", "D", "v", "P"]
    for k, s_ in enumerate(scns):
        ax = axes[0, k]
        sub = df[df["scenario"] == s_]
        for idx_f, f in enumerate(fams_sc):
            gf = sub[sub["family"] == f].groupby("setting")[["FDR", "TPR"]].mean()
            ax.scatter(gf["FDR"], gf["TPR"], marker=markers[idx_f % len(markers)], s=55, alpha=0.85, label=f)
        ax.set_title("Scenario %d" % s_, fontsize=10)
        ax.set_xlabel("False Discovery Rate (FDR)"); ax.set_ylabel("True Positive Rate (TPR)")
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("S5: Support Recovery Trade-off (TPR vs FDR)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_S5_tpr_vs_fdr.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- S8 : ARI by K Bar Chart (Cluster Recovery on Scenarios 1, 2, 4) ---
    df_ari = df[df["scenario"].isin([1, 2, 4])].copy()
    ari_fams = [f for f in ["CEN-K2", "CEN-K3", "CEN-K5", PROPOSED] if f in df_ari["family"].unique()]
    if len(df_ari) and ari_fams:
        sts_ari = list(dict.fromkeys(df_ari["scn_setting"]))
        fig, ax = plt.subplots(figsize=(11, 4.2))
        x = np.arange(len(sts_ari))
        w = 0.18
        for i, f in enumerate(ari_fams):
            means = [df_ari[(df_ari["scn_setting"] == st) & (df_ari["family"] == f)]["ARI"].mean() for st in sts_ari]
            ses = [df_ari[(df_ari["scn_setting"] == st) & (df_ari["family"] == f)]["ARI"].sem() for st in sts_ari]
            ax.bar(x + (i - (len(ari_fams) - 1) / 2) * w, means, w, yerr=ses, capsize=2, label=f)
        ax.set_xticks(x); ax.set_xticklabels(sts_ari, rotation=25, ha="right", fontsize=8.5)
        ax.set_ylabel("Adjusted Rand Index (ARI)"); ax.set_ylim(0, 1.08)
        ax.set_title("S8: Cluster Recovery (ARI on True Active Blocks, True K = 3)", fontsize=10.5)
        ax.legend(fontsize=8.5); ax.grid(axis="y", linestyle=":", alpha=0.6)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_S8_ari_by_K.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    # --- S9 : K-Selection Frequency Bar Chart for CEN-multi(K tuned) ---
    df_cm = df[df["family"] == PROPOSED].copy()
    if len(df_cm) and "hyper" in df_cm:
        df_cm["K_chosen"] = df_cm["hyper"].astype(str).str.extract(r"K=(\d+)").astype(float)
        ks = sorted(df_cm["K_chosen"].dropna().unique())
        if ks:
            fig, ax = plt.subplots(figsize=(10.5, 4.2))
            x = np.arange(len(settings_ordered))
            bottom = np.zeros(len(settings_ordered))
            for k_val in ks:
                props = [np.mean(df_cm[df_cm["scn_setting"] == st]["K_chosen"] == k_val) * 100 for st in settings_ordered]
                ax.bar(x, props, width=0.55, bottom=bottom, label="K = %d" % int(k_val))
                bottom += np.array(props)
            ax.set_xticks(x); ax.set_xticklabels(settings_ordered, rotation=25, ha="right", fontsize=8.5)
            ax.set_ylabel("Selection Frequency (%)"); ax.set_ylim(0, 105)
            ax.set_title("S9: Inner-CV Selected Number of Clusters (K) in CEN-multi", fontsize=10.5)
            ax.legend(fontsize=8.5, loc="upper right"); ax.grid(axis="y", linestyle=":", alpha=0.6)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "fig_S9_K_selection_freq.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="quick", choices=list(cen.PRESETS))
    ap.add_argument("--R", type=int, default=30)
    ap.add_argument("--p", type=int, default=10000, help="number of features generated (high-dimensional)")
    ap.add_argument("--m", type=int, default=1000, help="features kept by training-fold screening (as in the DOROTHEA pipeline)")
    ap.add_argument("--n_jobs", type=int, default=max(1, int(0.9 * (os.cpu_count() or 1))), help="default = 90%% of cores")
    ap.add_argument("--scenarios", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--save_data_reps", type=int, default=1, help="number of replicates per setting to save as CSV dataset")
    ap.add_argument("--plots_only", action="store_true", help="skip simulation run, build all plots & sample CSV data from existing S*_per_rep.csv")
    a = ap.parse_args()
    global OUT_DIR
    OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_results_p%d" % a.p)
    cfg = dict(cen.PRESETS[a.preset]); cfg.update(cen.COMMON); cfg["m"] = min(a.m, a.p)
    os.makedirs(OUT_DIR, exist_ok=True)
    base = cfg["seed"]

    parts = {}
    if a.plots_only:
        print("Loading existing S*_per_rep.csv files from:", OUT_DIR)
        for scn in a.scenarios:
            f_csv = os.path.join(OUT_DIR, "S%d_per_rep.csv" % scn)
            if os.path.exists(f_csv):
                parts[scn] = [pd.read_csv(f_csv)]
        # Also dump simulated datasets to CSV (rep 0..save_data_reps-1) so CSV datasets exist without rerunning models
        if a.save_data_reps > 0:
            for si, (scn, label, par) in enumerate(SETTINGS):
                if scn not in a.scenarios: continue
                beta, blk = true_model(scn, par, a.p, base + si)
                for rep in range(a.save_data_reps):
                    rng = np.random.default_rng(base + 1000 * rep + scn)
                    while True:
                        Z, y = gen_data(scn, par, a.p, beta, N_TRAIN + N_TEST, rng)
                        if y[:N_TRAIN].sum() >= 8 and (1 - y[:N_TRAIN]).sum() >= 8 and 0 < y[N_TRAIN:].sum() < N_TEST: break
                    save_simulated_dataset_csv(Z, y, beta, blk, scn, label, rep, cfg["m"], OUT_DIR)
            print("Saved simulated datasets (.csv) in:", os.path.join(OUT_DIR, "sim_datasets"))
    else:
        print("Simulation: n_train=%d n_test=%d p=%d (p/n=%.0f) screened m=%d R=%d preset=%s n_jobs=%d" %
              (N_TRAIN, N_TEST, a.p, a.p / N_TRAIN, cfg["m"], a.R, a.preset, a.n_jobs), flush=True)
        for si, (scn, label, par) in enumerate(SETTINGS):
            if scn not in a.scenarios: continue
            t0 = time.time()
            args = [(scn, label, par, a.p, r, cfg, base, base + si, OUT_DIR, a.save_data_reps) for r in range(a.R)]
            if a.n_jobs > 1:
                from joblib import Parallel, delayed
                res = Parallel(n_jobs=a.n_jobs)(delayed(one_rep)(*x) for x in args)
            else:
                res = [one_rep(*x) for x in args]
            parts.setdefault(scn, []).append(pd.DataFrame([r for rr in res for r in rr]))
            print("  S%d [%s] done: %d reps in %.0fs" % (scn, label, a.R, time.time() - t0), flush=True)
            dfs = pd.concat(parts[scn])
            dfs.drop(columns=["sel"], errors="ignore").to_csv(os.path.join(OUT_DIR, "S%d_per_rep.csv" % scn), index=False)
            summarize(dfs).to_csv(os.path.join(OUT_DIR, "S%d_summary.csv" % scn), index=False)

    df = pd.concat([d for v in parts.values() for d in v], ignore_index=True)
    if not a.plots_only:
        for scn in sorted(parts):
            print("\n=== Scenario %d : mean ± sd over %d replicates ===" % (scn, a.R))
            print(summarize(pd.concat(parts[scn])).to_string(index=False))
        for m in ["ROC_AUC", "PR_AUC"]:                                   # cross-scenario comparison
            w = df.groupby(["scenario", "setting", "family"])[m].mean().unstack("family").reindex(columns=ORDER).round(4)
            w.to_csv(os.path.join(OUT_DIR, "ALL_comparison_%s.csv" % m.lower()))
            print("\n=== Cross-scenario comparison: mean %s ===\n%s" % (m, w.to_string()))

    vd = make_verdicts(df); vd.to_csv(os.path.join(OUT_DIR, "verdicts.csv"), index=False)
    if not a.plots_only:
        print(); write_notes(vd, os.path.join(OUT_DIR, "scenario_notes.txt"))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fams = ["Lasso", "Ridge", "ElasticNet", "CEN-K1", PROPOSED]
        sc = sorted(parts); fig, ax = plt.subplots(1, len(sc), figsize=(4.2 * len(sc), 4), squeeze=False)
        for k, s_ in enumerate(sc):
            g = df[df.scenario == s_]; sts = list(dict.fromkeys(g.setting))
            for i, f in enumerate(fams):
                mu = [g[(g.setting == t) & (g.family == f)].ROC_AUC.mean() for t in sts]
                se = [g[(g.setting == t) & (g.family == f)].ROC_AUC.std() / np.sqrt(a.R) for t in sts]
                ax[0, k].bar(np.arange(len(sts)) + i * 0.16, mu, 0.16, yerr=se, label=f)
            ax[0, k].set_xticks(np.arange(len(sts)) + 0.32); ax[0, k].set_xticklabels(sts, rotation=30, ha="right", fontsize=7)
            ax[0, k].set_title("Scenario %d" % s_); ax[0, k].set_ylim(0.5, 1.0); ax[0, k].set_ylabel("test ROC-AUC")
        ax[0, 0].legend(fontsize=6); fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig_sim_roc_auc.png"), dpi=200)
        plt.close(fig)

        # Generate the 6 new Simulation figures (S1, S2, S3, S5, S8, S9)
        make_additional_sim_plots(df, vd, OUT_DIR)
    except Exception as e:
        print("plot skipped:", e)
    print("\nAll outputs saved in:", OUT_DIR)


if __name__ == "__main__":
    main()