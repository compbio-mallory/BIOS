#!/bin/bash
# Download TCGA HiSeqV2 RNA (gene symbols) from UCSC Xena for GSVA anchors
cd /gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC/data/raw
for C in BLCA GBM KIRC LUAD PAAD SKCM STAD UVM OV; do
  if [ -f ${C}_HiSeqV2_symbols.tsv.gz ]; then
    echo "$C: already present, skipping"; continue
  fi
  echo "=== downloading $C ==="
  wget -q "https://tcga.xenahubs.net/download/TCGA.${C}.sampleMap/HiSeqV2.gz" \
       -O ${C}_HiSeqV2_symbols.tsv.gz \
    && echo "$C ok: $(du -h ${C}_HiSeqV2_symbols.tsv.gz | cut -f1)" \
    || { echo "$C FAILED"; rm -f ${C}_HiSeqV2_symbols.tsv.gz; }
done
echo "=== verify: each should be tens of MB, gene symbols in col 1 ==="
for f in *_HiSeqV2_symbols.tsv.gz; do
  echo "$f: $(zcat $f | head -1 | tr '\t' '\n' | wc -l) samples, $(zcat $f | wc -l) genes"
done