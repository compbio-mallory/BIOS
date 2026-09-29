"""
paths.py — single source of truth for BIOS_v2 file locations.
Every script imports from here instead of hardcoding paths.
Override the root with the BIOS_ROOT environment variable if needed.
"""
import os

ROOT = os.environ.get("BIOS_ROOT", "/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2")

def omics(cancer, omic="CN"):
    """omic: CN, meth, miRNA, rna"""
    return f"{ROOT}/{cancer}/raw/{omic}.fea"

def omics_dir(cancer):
    return f"{ROOT}/{cancer}/raw"

def anchors(cancer, mode="gsva_H40"):
    return f"{ROOT}/{cancer}/anchors/bio_anchors_{cancer}_{mode}.csv"

def ground_truth(cancer):
    return f"{ROOT}/{cancer}/ground_truth/ground_truth_{cancer}.csv"

def survival(cancer):
    return f"{ROOT}/{cancer}/ground_truth/survival_endpoints_{cancer}_CDR.csv"

def results_dir(cancer, version):
    return f"{ROOT}/{cancer}/results/{version}"

def configs_dir(cancer):
    return f"{ROOT}/{cancer}/configs"

def config_template():
    return f"{ROOT}/scripts/lib/config_template.yaml"

def reference(name):
    return f"{ROOT}/scripts/lib/reference/{name}"

def subtyping_dir(cancer, method, version):
    return f"{ROOT}/{cancer}/results/subtyping_results/{method}/{version}"

def survival_dir(cancer, method, version):
    return f"{ROOT}/{cancer}/results/survival_results/{method}/{version}"

def label_dir(cancer, method, version):
    return f"{ROOT}/{cancer}/results/label_results/{method}/{version}"