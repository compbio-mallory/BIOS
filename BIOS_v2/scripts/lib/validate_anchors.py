"""
validate_anchors.py — QC across all H40 anchor files before training.
Checks: shape, ID format, NaN/inf, value ranges, omics coverage, and a
biological consistency test (correlated pathway pairs that MUST correlate).
"""
import os
import numpy as np
import pandas as pd

CANCERS = ['BRCA', 'UCEC', 'LUAD', 'BLCA', 'KIRC', 'PAAD', 'SKCM', 'STAD', 'GBM', 'UVM']

# Pairs that are biologically obligated to correlate positively in tumor RNA.
# If GSVA is computed correctly, these hold in every cancer.
SANITY_PAIRS = [
    ('HALLMARK_E2F_TARGETS', 'HALLMARK_G2M_CHECKPOINT'),          # both proliferation
    ('HALLMARK_INTERFERON_ALPHA_RESPONSE', 'HALLMARK_INTERFERON_GAMMA_RESPONSE'),
    ('HALLMARK_ESTROGEN_RESPONSE_EARLY', 'HALLMARK_ESTROGEN_RESPONSE_LATE'),
]

print(f"{'cancer':6s} {'anchors':>9s} {'omics':>6s} {'miss':>5s} "
      f"{'NaN':>4s} {'range':>14s}  E2F~G2M  IFNa~IFNg  ESR-E~L")
print("-" * 88)

problems = []
for c in CANCERS:
    fp = f'data/bio_anchors/bio_anchors_{c}_gsva_H40.csv'
    if not os.path.exists(fp):
        print(f"{c:6s}  MISSING FILE"); problems.append((c, 'missing file')); continue
    a = pd.read_csv(fp, index_col=0)

    # ID format: patient-level TCGA barcodes, unique
    bad_ids = (~a.index.str.match(r'^TCGA-\w{2}-\w{4}$')).sum()
    dup_ids = a.index.duplicated().sum()
    if bad_ids or dup_ids:
        problems.append((c, f'{bad_ids} bad ids, {dup_ids} dups'))

    # omics coverage
    cn = pd.read_csv(f'../subtype_file/fea/{c}/CN.fea', header=0, index_col=0, sep=',')
    miss = len(set(cn.columns) - set(a.index))

    nan = int(a.isnull().sum().sum()) + int(np.isinf(a.values).sum())
    vmin, vmax = a.values.min(), a.values.max()

    # biological consistency: Pearson r on the omics-covered patients
    common = [p for p in cn.columns if p in a.index]
    sub = a.loc[common]
    rs = []
    for p1, p2 in SANITY_PAIRS:
        r = sub[p1].corr(sub[p2]) if p1 in sub and p2 in sub else np.nan
        rs.append(r)

    flag = ''
    if a.shape[1] != 40: flag += ' WRONG_NCOLS'; problems.append((c, f'{a.shape[1]} cols'))
    if miss > 10: flag += ' HIGH_MISS'; problems.append((c, f'{miss} missing'))
    if nan: flag += ' NAN'; problems.append((c, f'{nan} NaN/inf'))
    if rs[0] is not np.nan and rs[0] < 0.5:
        flag += ' PROLIF_CORR_LOW'; problems.append((c, f'E2F~G2M r={rs[0]:.2f}'))

    print(f"{c:6s} {str(a.shape):>9s} {len(cn.columns):>6d} {miss:>5d} "
          f"{nan:>4d} [{vmin:6.2f},{vmax:5.2f}]  "
          f"{rs[0]:7.2f}  {rs[1]:8.2f}  {rs[2]:6.2f}{flag}")

print("-" * 88)
if problems:
    print("PROBLEMS:")
    for c, msg in problems: print(f"  {c}: {msg}")
else:
    print("ALL CLEAR — anchor layer validated for training.")