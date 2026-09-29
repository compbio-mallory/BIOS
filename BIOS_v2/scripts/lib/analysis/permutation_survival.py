"""
permutation_survival.py — analytic vs permutation log-rank p-values.

Follows Rappoport & Shamir (2018) benchmark.R adaptive scheme, parallelised
across cores with multiprocessing.
"""
import argparse, os
import numpy as np, pandas as pd
from multiprocessing import Pool
from lifelines.statistics import multivariate_logrank_test
from scipy.stats import beta

ap = argparse.ArgumentParser()
ap.add_argument("--clusters", required=True)
ap.add_argument("--survival", required=True)
ap.add_argument("--format", choices=["cdr", "shamir"], default="cdr")
ap.add_argument("--endpoint", default="OS")
ap.add_argument("--max-perms", type=int, default=10_000_000)
ap.add_argument("--cores", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

def norm(s):
    return str(s).strip().upper().replace('_', '-').replace('.', '-')[:12]

pred = pd.read_csv(a.clusters, comment='#')
pred.columns = [c.lower() for c in pred.columns]
pred['patient'] = pred[pred.columns[0]].map(norm)
pred = pred.drop_duplicates('patient')[['patient', 'cluster']]

if a.format == "cdr":
    clin = pd.read_csv(a.survival, comment='#')
    bc = 'bcr_patient_barcode' if 'bcr_patient_barcode' in clin.columns else clin.columns[0]
    clin = clin[[bc, a.endpoint, f'{a.endpoint}.time']].copy()
else:
    clin = pd.read_table(a.survival)
    clin = clin[[clin.columns[0], 'Death', 'Survival']].copy()
clin.columns = ['patient', 'event', 'time']
clin['patient'] = clin['patient'].map(norm)
clin['event'] = pd.to_numeric(clin['event'], errors='coerce').fillna(0)
clin['time'] = pd.to_numeric(clin['time'], errors='coerce').fillna(0)

df = pred.merge(clin, on='patient', how='inner')
cnt = df['cluster'].value_counts()
df = df[df['cluster'].isin(cnt[cnt >= 2].index)]
print(f"n={len(df)}  events={int(df['event'].sum())}  "
      f"clusters={df['cluster'].nunique()}  cores={a.cores}", flush=True)

TIME = df['time'].values
EVENT = df['event'].values
LABELS = df['cluster'].values.copy()

obs = multivariate_logrank_test(TIME, LABELS, EVENT)
OBS_CHI2, analytic_p = obs.test_statistic, obs.p_value
print(f"analytic:    chi2={OBS_CHI2:.4f}  p={analytic_p:.4e}  "
      f"-log10p={-np.log10(analytic_p):.4f}", flush=True)

def _one(seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(LABELS)
    return 1 if multivariate_logrank_test(TIME, perm, EVENT).test_statistic >= OBS_CHI2 else 0

n_perms = int(min(max(10 / max(analytic_p, 1e-12), 1000), 1e6))
total_perms = total_extreme = 0
seed_base = a.seed

while True:
    n_perms = min(n_perms, a.max_perms - total_perms)
    if n_perms <= 0:
        break
    with Pool(processes=a.cores) as pool:
        chunk = sum(pool.map(_one, range(seed_base, seed_base + n_perms), chunksize=500))
    seed_base += n_perms
    total_perms += n_perms
    total_extreme += chunk
    lo = 0.0 if total_extreme == 0 else beta.ppf(0.025, total_extreme, total_perms - total_extreme + 1)
    hi = 1.0 if total_extreme == total_perms else beta.ppf(0.975, total_extreme + 1, total_perms - total_extreme)
    est = total_extreme / total_perms
    print(f"  perms={total_perms}  extreme={total_extreme}  p={est:.3e}  "
          f"CI=[{lo:.3e},{hi:.3e}]", flush=True)
    tight = (hi - est) < min(max(est, 1e-9) / 10, 0.01) and (est - lo) < min(max(est, 1e-9) / 10, 0.01)
    straddles = lo < 0.05 < hi
    if (tight and not straddles) or total_perms >= a.max_perms:
        break
    n_perms = total_perms

emp_p = max(total_extreme / total_perms, 1 / total_perms)
floored = total_extreme == 0
print(f"permutation: p={'<' if floored else ''}{emp_p:.4e}  "
      f"-log10p={'>' if floored else ''}{-np.log10(emp_p):.4f}  "
      f"({total_perms} permutations)", flush=True)
print(f"difference in -log10p: {-np.log10(emp_p) + np.log10(analytic_p):+.4f}")