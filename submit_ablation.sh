#!/bin/bash
cd /gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC
for HEAD in linear mlp attention; do
  for BD in 4 16; do
    for PW in 4 40; do
      if [ "$PW" = "4" ]; then AM=gsva_decorr_M4; else AM=gsva_H40; fi
      if [ "$HEAD" = "attention" ]; then HT=attention_query_v3; else HT=$HEAD; fi
      NAME="head_${HEAD}_biodim${BD}_pathways${PW}"
      sbatch --job-name=$NAME \
        --export=CANCER=BRCA,K=5,HEAD=$HT,ANCHOR_MODE=$AM,NANCH=$PW,BIODIM=$BD,EPOCHS=600 \
        slurm/run_bios_generic.sh
    done
  done
done
