import os, shutil, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT, omics_dir, anchors

C = 'BLCA'
src = omics_dir(C)
bak = f"{ROOT}/{C}/raw_original"
os.makedirs(bak, exist_ok=True)

a = pd.read_csv(anchors(C, 'gsva_H40'), index_col=0)
for f in ['CN.fea', 'meth.fea', 'miRNA.fea', 'rna.fea']:
    fp = f"{src}/{f}"
    if not os.path.exists(f"{bak}/{f}"):
        shutil.copy2(fp, f"{bak}/{f}")          # backup first, never overwrite
    df = pd.read_csv(fp, header=0, index_col=0, sep=',')
    keep = [c for c in df.columns if c in a.index]
    dropped = [c for c in df.columns if c not in a.index]
    df[keep].to_csv(fp)
    print(f"{f}: {len(df.columns)} -> {len(keep)}  dropped={dropped}")
print(f"\noriginals backed up in {bak}")