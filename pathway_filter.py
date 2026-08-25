"""
pathway_filter.py — a-priori filtering of MSigDB Hallmark v2025.1 (50) down to a
cancer-relevant anchor panel for BIOS.

Rationale (Xian, 7/21): tissue-lineage Hallmarks that have no a-priori link to
tumour biology pollute the anchor signal and produce spurious cluster enrichment
(spermatogenesis, myogenesis showed up at significant q in UCEC/BRCA).

IMPORTANT: this filter is BIOLOGY-BASED and chosen BEFORE looking at results.
Do not tune it against ARI or against which pathways the attention picks — that
turns an a-priori panel into a data-driven one and destroys the interpretability
claim. If you change it, change it for a stated biological reason and record the
reason in the docstring below.

Usage:
    python pathway_filter.py \
        --in  data/bio_anchors/bio_anchors_UCEC_gsva_H50.csv \
        --out data/bio_anchors/bio_anchors_UCEC_gsva_H40.csv

    python pathway_filter.py --list          # print keep/drop panel and exit
"""

import argparse
import sys

# ---------------------------------------------------------------------------
# DROP LIST
# ---------------------------------------------------------------------------
# Each entry: pathway -> reason for exclusion. Reasons are the record of intent.
DROP_REASONS = {
    "HALLMARK_SPERMATOGENESIS":
        "Male germ-cell lineage programme. No mechanistic role in endometrial or "
        "breast tumour biology; cancer-testis antigen leakage makes it a known "
        "false-positive in GSVA on bulk tumour.",
    "HALLMARK_PANCREAS_BETA_CELLS":
        "Endocrine pancreas lineage identity. Organ-specific, not a tumour programme.",
    "HALLMARK_MYOGENESIS":
        "Skeletal/smooth muscle differentiation. In bulk tumour this tracks stromal "
        "and myometrial content, i.e. tissue composition, not tumour state. Major "
        "confounder in UCEC specifically (myometrial invasion / sampling).",
    "HALLMARK_ADIPOGENESIS":
        "Adipocyte differentiation. Breast is adipose-rich, so in BRCA this is a "
        "direct readout of sample composition rather than tumour phenotype.",
    "HALLMARK_BILE_ACID_METABOLISM":
        "Hepatic/enterohepatic metabolic programme. Not relevant to UCEC/BRCA. "
        "REVISIT if the panel is ever applied to LIHC/CHOL.",
    "HALLMARK_HEME_METABOLISM":
        "Erythroid lineage / haem biosynthesis. Tracks blood content of the sample.",
    "HALLMARK_COAGULATION":
        "Largely plasma/stromal and liver-derived factors. Cancer-associated "
        "thrombosis is real but is a systemic phenomenon, not a subtype axis, and "
        "the gene set overlaps heavily with complement/stroma. LOW CONFIDENCE drop "
        "— flag to Xian if she wants it kept.",
    "HALLMARK_PROTEIN_SECRETION":
        "Generic housekeeping secretory/Golgi machinery. Ubiquitous, low specificity.",
    "HALLMARK_APICAL_SURFACE":
        "Small (44 genes), narrow epithelial-polarity membrane set, poor GSVA "
        "stability. APICAL_JUNCTION is retained as the better-powered polarity/EMT "
        "proxy of the pair.",
    "HALLMARK_PEROXISOME":
        "Organelle housekeeping metabolism. No established subtype association in "
        "either cancer.",
}

# ---------------------------------------------------------------------------
# DELIBERATELY KEPT — these were on the draft drop list; here is why they stay.
# ---------------------------------------------------------------------------
KEEP_OVERRIDE_REASONS = {
    "HALLMARK_NOTCH_SIGNALING":
        "Canonical oncogenic/tumour-suppressive signalling hallmark. Context-"
        "dependent NOTCH is well documented in breast cancer and in endometrial "
        "carcinoma. Dropping a core signalling pathway would be hard to defend to "
        "a reviewer.",
    "HALLMARK_ALLOGRAFT_REJECTION":
        "This is the standard MSigDB proxy for immune infiltration in bulk tumour. "
        "Immune-hot vs immune-cold is a genuine subtype axis — MSI/POLE UCEC are "
        "hypermutated and immune-infiltrated, basal-like BRCA is immune-enriched. "
        "Expect and WANT this to load on the MSI cluster.",
}

DROP = frozenset(DROP_REASONS)


def _norm(name: str) -> str:
    """Normalise a column name to HALLMARK_UPPER_SNAKE for matching."""
    n = name.strip().upper().replace("-", "_").replace(" ", "_")
    if not n.startswith("HALLMARK_"):
        n = "HALLMARK_" + n
    return n


def filter_columns(columns):
    """Return (kept, dropped, unmatched) given an iterable of anchor column names."""
    kept, dropped = [], []
    for c in columns:
        (dropped if _norm(c) in DROP else kept).append(c)
    seen = {_norm(c) for c in columns}
    unmatched = sorted(DROP - seen)
    return kept, dropped, unmatched


def filter_anchor_file(in_path, out_path, id_col=None):
    import pandas as pd

    df = pd.read_csv(in_path, index_col=0)
    kept, dropped, unmatched = filter_columns(df.columns)

    if unmatched:
        print(f"[warn] drop-list entries not found in {in_path}:", file=sys.stderr)
        for u in unmatched:
            print(f"       {u}", file=sys.stderr)

    out = df[kept]
    out.to_csv(out_path)
    print(f"{in_path}  ->  {out_path}")
    print(f"  patients: {out.shape[0]}   anchors: {df.shape[1]} -> {out.shape[1]}")
    print(f"  dropped ({len(dropped)}): {', '.join(sorted(dropped))}")
    nan = out.isna().sum().sum()
    print(f"  NaNs in output: {nan}" + ("  <-- INVESTIGATE" if nan else ""))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", help="input anchor CSV (patients x 50)")
    p.add_argument("--out", dest="out", help="output anchor CSV (patients x 40)")
    p.add_argument("--list", action="store_true", help="print the panel and exit")
    a = p.parse_args()

    if a.list or not (a.inp and a.out):
        print(f"DROP ({len(DROP_REASONS)}):")
        for k, v in sorted(DROP_REASONS.items()):
            print(f"  {k}\n      {v}")
        print(f"\nEXPLICITLY KEPT ({len(KEEP_OVERRIDE_REASONS)}):")
        for k, v in sorted(KEEP_OVERRIDE_REASONS.items()):
            print(f"  {k}\n      {v}")
        print("\n50 - 10 = 40 anchors retained.")
        return

    filter_anchor_file(a.inp, a.out)


if __name__ == "__main__":
    main()
