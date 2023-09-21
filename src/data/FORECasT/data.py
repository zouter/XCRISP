import os
import platform
import pandas as pd

CELL_TYPES = {
    "HAP1": "HAP1",
    "TREX_B": "TREX2_12NB_DPI7",
    "TREX_A": "TREX2_12NA_DPI7",
    "2A_TREX_A": "2A_TREX2_12NA_DPI7",
    "2A_TREX_B": "2A_TREX2_12NB_DPI7",
    "K562": "K562",
    "mESC": "E14TG2A",
    "NULL": "NULL_New",
}

METHODS = ["forecast", "lindel", "lumc"]

DATASETS = ["forecast", "WT", "POLQ"]

def get_home():
    if platform.system() == "Darwin":
        return "/Users/colm"
    else:
        return "/tudelft.net/staff-umbrella/protonddr/"

def get_assembled_reads_folder(d):
    if d == "lumc":
        d = "LUMC" 
    return "%s/assembled_reads/%s" % (get_output_folder(), d)

def get_repos_folder():
    return "{}/repos".format(get_home()) 

def get_exp_oligo_file():
    return "{}/FORECasT/exp_target_pam.fasta".format(get_output_folder())

def get_expanded_targets_for_oligo(oligo_id):
    return "{}/mapped_reads/NULL/{}/{}_exptargets.txt".format(get_output_folder(), oligo_id, oligo_id)

def get_forecast_pam_locations_file():
    return "{}/repair-outcome-prediction/data/PRJEB29746/targets.txt".format(get_repos_folder())

def get_accessions_file():
    return "{}/repos/repair-outcome-prediction/local/data/FORECasT/sample_accessions.txt".format(get_home())

def get_aspera_file():
    return "{}/repos/repair-outcome-prediction/local/data/FORECasT/PRJEB29746.txt".format(get_home())

def get_data_folder(d):
    data_dir = "data/%s"
    if d == "forecast":
        data_dir = data_dir % "PRJEB29746"
    elif d == "lumc":
        data_dir = data_dir % "LUMC"    
    if platform.system() == "Darwin":
        return "{}/{}".format(get_repos_folder(), data_dir)
    else:
        return "{}/{}".format(get_home(), data_dir)

def get_output_folder():
    if platform.system() == "Darwin":
        return "{}/local".format(get_repos_folder())
    else:
        return "{}/local".format(get_home())

def get_generated_indels_folder(c):
    if c not in DATASETS: raise Exception("c not in %s" % DATASETS)
    if c in ["WT", "POLQ"]: c = "lumc/%s" % c
    return "{}/generated_indels/{}".format(get_output_folder(), c)

def get_features_folder(c):
    if c not in DATASETS: raise Exception("c not in %s" % DATASETS)
    if c in ["WT", "POLQ"]: c = "lumc/%s" % c
    return "{}/features_for_gen_indels/{}".format(get_output_folder(), c)

def get_reads_folder(c):
    if c not in DATASETS: raise Exception("c not in %s" % DATASETS)
    if c in ["WT", "POLQ"]: c = "lumc/%s" % c
    return "{}/FORECasT/reads_for_gen_indels/{}".format(get_output_folder(), c)

def get_logs_folder():
    d = get_repos_folder() if platform.system() == "Darwin" else get_home()
    return "{}/logs".format(d)

def get_jobs_folder():
    d = get_repos_folder() if platform.system() == "Darwin" else get_home()
    return "{}/jobs".format(d)

def get_model_output_folder():
    return "{}/model_output".format(get_output_folder())

def get_mapped_reads_folder(study, cell_type):
    return "{}/mapped_reads/{}/{}".format(get_output_folder(), study, cell_type)

def get_cell_types():
    return CELL_TYPES

def is_old_lib(full_dirname):
    dirname = full_dirname.split('/')[-1]
    if ('O' in dirname and 'BOB' not in dirname and 'CHO' not in dirname) or 'Old' in dirname or 'old' in dirname:
        return True
    return False

def get_WT_dir(full_dirname):
    return get_data_folder() + "/WT"

def load_sample_accession_ERS(cell_line):
    sa = pd.read_csv(get_accessions_file(), sep="\t")
    return list(sa[sa["Sample Label"].str.contains(CELL_TYPES[cell_line])]["ERS"])

def find_ERR_for_ERS(ERSs):
    d = pd.read_csv(get_aspera_file(), sep="\t")
    return list(d[d["secondary_sample_accession"].isin(ERSs)]["run_accession"])

def get_forecast_target_details():
    return get_repos_folder() + "/repair-outcome-prediction/FORECasT/data/PRJEB29746/targets.txt"

def get_lumc_oligo_file(genotype):
    return "{}/LUMC/lumc_targets_pam.{}.fasta".format(get_output_folder(), genotype)
