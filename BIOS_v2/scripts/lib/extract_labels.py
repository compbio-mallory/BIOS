"""
extract_labels.py — clinical parameter enrichment, following
Rappoport & Shamir (2018) benchmark.R: check.clinical.enrichment (line 1029)
and get.empirical.clinical (line 1096).

Six parameters tested:
  gender                              DISCRETE  -> chi-square
  age_at_initial_pathologic_diagnosis NUMERIC   -> Kruskal-Wallis
  pathologic_M, pathologic_N,
  pathologic_T, pathologic_stage      DISCRETE  -> chi-square

Skip rules (benchmark.R):
  - numeric: skip if fewer than half the values parse as numeric
  - discrete: skip if more than half missing, or only one distinct value

Two p-values per parameter (same idea as survival):
  - analytic:    chi-square / Kruskal-Wallis p-value from scipy
  - permutation: port of get.empirical.clinical — shuffle cluster labels,
                 count permutations with p-value <= observed p-value,
                 batches of 1000, stop when the Clopper-Pearson 95% CI
                 excludes 0.05 or after 100,000 permutations, seed 42.
                 Like the R code, the estimate can be 0 when no permutation
                 is as extreme.
Significance for both: Bonferroni, p * (number of parameters tested) < 0.05
(benchmark.R line 440).

Output: label_results/<METHOD>/<version>/labels.csv with a provenance header.

Usage:
  python extract_labels.py --cancer BRCA --method BIOS --version v2_... \
      [--clusters <path>] [--clinical <path>] [--no-perm]
"""
import argparse, os, sys
from datetime import datetime
import numpy as np, pandas as pd
from scipy.stats import chi2_contingency, kruskal, beta
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT, subtyping_dir, label_dir

PARAMS = {
    'gender': 'DISCRETE',
    'age_at_initial_pathologic_diagnosis': 'NUMERIC',
    'pathologic_M': 'DISCRETE',
    'pathologic_N': 'DISCRETE',
    'pathologic_T': 'DISCRETE',
    'pathologic_stage': 'DISCRETE',
}


# ---------------------------------------------------------------------------
# test statistics
# ---------------------------------------------------------------------------
def _chisq_p(cl_codes, val_codes, n_cl, n_val):
    tbl = np.bincount(cl_codes * n_val + val_codes,
                      minlength=n_cl * n_val).reshape(n_cl, n_val)
    tbl = tbl[tbl.sum(1) > 0][:, tbl.sum(0) > 0]
    if tbl.shape[0] < 2 or tbl.shape[1] < 2:
        return 1.0, np.nan
    stat, p, _, _ = chi2_contingency(tbl)
    return p, stat


def _kruskal_p(cl_codes, values):
    groups = [values[cl_codes == g] for g in np.unique(cl_codes)]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return 1.0, np.nan
    r = kruskal(*groups)
    return r.pvalue, r.statistic


def _clopper_pearson(k, n):
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return lo, hi


def empirical_clinical(clusters, values, is_chisq, seed=42,
                       batch=1000, max_iters=100_000, sig=0.05):
    """Port of get.empirical.clinical (benchmark.R lines 1096-1143).
    Returns (analytic_p, statistic, permutation_p, n_permutations)."""
    rng = np.random.default_rng(seed)
    cl_codes = np.unique(clusters, return_inverse=True)[1]
    n_cl = cl_codes.max() + 1
    if is_chisq:
        val_codes = np.unique(np.asarray(values).astype(str), return_inverse=True)[1]
        n_val = val_codes.max() + 1
        test = lambda c: _chisq_p(c, val_codes, n_cl, n_val)
    else:
        v = np.asarray(values, dtype=float)
        test = lambda c: _kruskal_p(c, v)

    orig_p, stat = test(cl_codes)
    total = extreme = 0
    while True:
        for _ in range(batch):
            if test(rng.permutation(cl_codes))[0] <= orig_p:
                extreme += 1
        total += batch
        lo, hi = _clopper_pearson(extreme, total)
        if (not (lo < sig < hi)) or total > max_iters:
            break
    return orig_p, stat, extreme / total, total


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--cancer", required=True)
ap.add_argument("--method", default="BIOS")
ap.add_argument("--version", required=True)
ap.add_argument("--clusters", default=None)
ap.add_argument("--clinical", default=None)
ap.add_argument("--no-perm", action="store_true",
                help="skip permutation p-values (analytic only)")
a = ap.parse_args()


def norm(s):
    return str(s).strip().upper().replace('_', '-').replace('.', '-')[:12]


pred_fp = a.clusters or f"{subtyping_dir(a.cancer, a.method, a.version)}/clusters.csv"
clin_fp = a.clinical or f"{ROOT}/{a.cancer}/raw/{a.cancer}_clinicalMatrix.txt"

pred = pd.read_csv(pred_fp, comment='#')
pred.columns = [c.lower() for c in pred.columns]
pred['patient'] = pred[pred.columns[0]].map(norm)
pred = pred.drop_duplicates('patient')[['patient', 'cluster']]

clin = pd.read_csv(clin_fp, sep='\t', low_memory=False)
clin['patient'] = clin[clin.columns[0]].map(norm)
clin = clin.drop_duplicates('patient').set_index('patient')

df = pred.set_index('patient')
common = df.index.intersection(clin.index)
df = df.loc[common]
clin = clin.loc[common]
print(f"{a.cancer}: {len(pred)} clustered, {len(common)} matched to clinical")

rows = []
for param, kind in PARAMS.items():
    base = {'parameter': param, 'type': kind}
    if param not in clin.columns:
        rows.append({**base, 'status': 'absent'})
        continue
    vals = clin[param]

    if kind == 'NUMERIC':
        num = pd.to_numeric(vals, errors='coerce')
        ok = num.notna()
        if 2 * ok.sum() < len(vals):
            rows.append({**base, 'status': 'skipped_missing'})
            continue
        cl, v, is_chisq = df.loc[ok, 'cluster'].values, num[ok].values, False
    else:
        ok = vals.notna() & (vals.astype(str).str.strip() != '')
        if 2 * ok.sum() < len(vals):
            rows.append({**base, 'status': 'skipped_missing'})
            continue
        if vals[ok].nunique() == 1:
            rows.append({**base, 'status': 'skipped_constant'})
            continue
        cl, v, is_chisq = df.loc[ok, 'cluster'].values, vals[ok].values, True

    if a.no_perm:
        codes = np.unique(cl, return_inverse=True)[1]
        if is_chisq:
            vc = np.unique(np.asarray(v).astype(str), return_inverse=True)[1]
            p, stat = _chisq_p(codes, vc, codes.max() + 1, vc.max() + 1)
        else:
            p, stat = _kruskal_p(codes, np.asarray(v, dtype=float))
        p_perm, n_perm = np.nan, 0
    else:
        p, stat, p_perm, n_perm = empirical_clinical(cl, v, is_chisq)

    rows.append({**base, 'status': 'tested', 'statistic': stat,
                 'p_value': p, 'p_perm': p_perm, 'n_perms': n_perm})
    print(f"  {param:37s} analytic p={p:.3e}   perm p={p_perm:.3e} ({n_perm} perms)")

out = pd.DataFrame(rows)
n_tested = int((out['status'] == 'tested').sum())
tested = out['status'] == 'tested'
out['p_bonferroni'] = np.where(tested, np.minimum(out.get('p_value', np.nan) * n_tested, 1.0), np.nan)
out['significant'] = tested & (out.get('p_value', np.nan) * n_tested < 0.05)
out['p_perm_bonferroni'] = np.where(tested, np.minimum(out.get('p_perm', np.nan) * n_tested, 1.0), np.nan)
out['significant_perm'] = tested & (out.get('p_perm', np.nan) * n_tested < 0.05)
n_sig = int(out['significant'].sum())
n_sig_perm = int(out['significant_perm'].sum())

out.insert(0, 'version', a.version)
out.insert(0, 'method', a.method)
out.insert(0, 'cancer', a.cancer)
cols = ['cancer', 'method', 'version', 'parameter', 'type', 'status', 'statistic',
        'p_value', 'p_bonferroni', 'significant',
        'p_perm', 'p_perm_bonferroni', 'significant_perm', 'n_perms']
out = out.reindex(columns=cols)

outdir = label_dir(a.cancer, a.method, a.version)
os.makedirs(outdir, exist_ok=True)
rel_src = os.path.relpath(os.path.abspath(pred_fp), ROOT)
hdr = (f"# source: {rel_src}\n"
       f"# clinical: {os.path.relpath(os.path.abspath(clin_fp), ROOT)}\n"
       f"# tested: {n_tested} of {len(PARAMS)} parameters\n"
       f"# significant after Bonferroni: analytic {n_sig}, permutation "
       f"{'n/a' if a.no_perm else n_sig_perm}\n"
       f"# permutation: port of benchmark.R get.empirical.clinical "
       f"(batches of 1000, stop when CI excludes 0.05 or >1e5, seed 42)\n"
       f"# generated: {datetime.now():%Y-%m-%d %H:%M} by extract_labels.py\n")

fp = f"{outdir}/labels.csv"
with open(fp, 'w') as fh:
    fh.write(hdr)
    out.to_csv(fh, index=False)

print(f"\n{a.cancer}: significant of {n_tested} tested — analytic {n_sig}, "
      f"permutation {'n/a' if a.no_perm else n_sig_perm} -> {fp}")
