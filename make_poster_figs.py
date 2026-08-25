#!/usr/bin/env python3
"""Poster: one grouped bar chart per cancer, ARI + V-measure for each method."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size':15,'axes.titlesize':18,'axes.labelsize':16,
    'xtick.labelsize':12,'ytick.labelsize':13,'legend.fontsize':13,
    'axes.spines.top':False,'axes.spines.right':False,
    'font.family':'DejaVu Sans','figure.dpi':200})
ARI_C='#4575b4'      # blue = ARI
VM_C ='#1b7837'      # green = V-measure
BIOS_HL='#c2185b'    # accent to mark BIOS on x-axis

def grouped(data, title, fname):
    # data: list of (method, ARI, Vmeasure), sorted by ARI ascending
    methods=[d[0] for d in data]; ari=[d[1] for d in data]; vm=[d[2] for d in data]
    x=np.arange(len(methods)); w=0.4
    fig,ax=plt.subplots(figsize=(11,5.8))
    b1=ax.bar(x-w/2, ari, w, label='ARI', color=ARI_C, edgecolor='white')
    b2=ax.bar(x+w/2, vm,  w, label='V-measure', color=VM_C, edgecolor='white')
    for bars in (b1,b2):
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.006,
                    f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=10.5)
    ax.set_xticks(x); ax.set_xticklabels(methods)
    # bold + color the BIOS tick label
    for lbl in ax.get_xticklabels():
        if lbl.get_text()=='BIOS':
            lbl.set_fontweight('bold'); lbl.set_color(BIOS_HL)
    ax.set_ylabel('Score'); ax.set_title(title, fontweight='bold')
    ax.set_ylim(0, max(max(ari),max(vm))*1.16)
    ax.legend(frameon=False, loc='upper left'); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig(fname, bbox_inches='tight'); plt.close()
    print('wrote', fname)

# UCEC (molecular, k=4) — sorted by ARI
grouped([
    ('Agglom.',      0.185, 0.204),
    ('Spectral',     0.258, 0.302),
    ('PCA+KMeans',   0.294, 0.286),
    ('Subtype-DCC',  0.329, 0.339),
    ('SNF',          0.334, 0.330),
    ('NEMO',         0.337, 0.354),
    ('KMeans',       0.344, 0.326),
    ('BIOS',         0.363, 0.360),
], 'UCEC — method comparison (molecular subtypes, k=4)',
   'results/poster_ucec_combined.png')

# BRCA (PAM50, k=5) — sorted by ARI (honest: Spectral leads)
grouped([
    ('NEMO',         0.291, 0.454),
    ('KMeans',       0.294, 0.457),
    ('PCA+KMeans',   0.309, 0.468),
    ('Subtype-DCC',  0.341, 0.475),
    ('SNF',          0.342, 0.427),
    ('Agglom.',      0.345, 0.441),
    ('BIOS',         0.356, 0.493),
    ('Spectral',     0.412, 0.510),
], 'BRCA — method comparison (PAM50, k=5)',
   'results/poster_brca_combined.png')
