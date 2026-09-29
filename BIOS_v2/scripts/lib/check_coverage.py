import pandas as pd, sys
sys.path.insert(0, '.')
from paths import omics, anchors
for C in ['BRCA','UCEC','LUAD','BLCA','KIRC','PAAD','SKCM','STAD','UVM']:
    cn = pd.read_csv(omics(C, 'CN'), header=0, index_col=0, sep=',')
    a = pd.read_csv(anchors(C, 'gsva_H40'), index_col=0)
    miss = set(cn.columns) - set(a.index)
    print(f'{C}: {len(cn.columns)} patients, {len(a)} anchor rows, {len(miss)} missing')
