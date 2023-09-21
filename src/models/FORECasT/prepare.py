import os, sys
import pandas as pd
from features import read_features_data
from tqdm import tqdm
sys.path.append("../")
from data_loader import load_Tijsterman_data, read_target_sequence_from_file, get_guides_from_fasta
from test_setup import MIN_NUMBER_OF_READS

MIN_READS_PER_TARGET = MIN_NUMBER_OF_READS
DATA_DIR = "../../"
PREPROCESSING_OPTIONS = ["Tijsterman_Analyser", "FORECasT"]
FEATURES_DIR = os.environ["OUTPUT_DIR"] + "model_training/data/FORECasT/features/{}/"
PROFILES_DIR = os.environ["OUTPUT_DIR"] + "model_training/data/FORECasT/profiles/{}/{}/"
COMBINED_DIR = os.environ["OUTPUT_DIR"] + "model_training/data_{}x".format(MIN_READS_PER_TARGET) + "/{}/{}/"
VERBOSE = True

def prepare_dataset(dataset, test_file):
    if dataset != "inDelphi":
        parts = test_file.split("_")
        samples_f = "../../data/{}/{}.fasta".format(dataset, parts[-1])
    else:
        parts = test_file.split("_")
        samples_f = "../../data/inDelphi/{}.fasta".format(parts[1])

    samples = get_guides_from_fasta(samples_f)
    output_dir = COMBINED_DIR.format(test_file, PREPROCESSING_OPTIONS[0])
    os.makedirs(output_dir, exist_ok=True)
    completed = 0
    for s in tqdm(samples):
        if dataset != "inDelphi":
            features_f = FEATURES_DIR.format(test_file) + s
        else:
            features_f = FEATURES_DIR.format(parts[0]) + s
        if not os.path.exists(features_f): 
            print(features_f)
            continue
        feature_data, feature_cols = read_features_data(features_f)

        if dataset != "inDelphi":
            profile_f = PROFILES_DIR.format(PREPROCESSING_OPTIONS[0], test_file) + s
        else:
            profile_f = PROFILES_DIR.format(PREPROCESSING_OPTIONS[0], parts[0]) + s
        
        if not os.path.exists(profile_f): 
            if VERBOSE: print(profile_f)
            continue
        profile_data = pd.read_csv(profile_f, sep='\t')
        profile_data["Counts"] = profile_data["Counts"] + 0.5
        total_reads = profile_data["Counts"].sum()
        if total_reads < MIN_READS_PER_TARGET: 
            if VERBOSE: print('Less than %s mutated Reads in %s' % (MIN_READS_PER_TARGET, profile_f))
            continue
        profile_data["Frac Sample Reads"] = profile_data["Counts"]/total_reads
        profile_data = profile_data[["Indel", "Counts", "Frac Sample Reads"]]
        merged_data = profile_data
        merged_data = feature_data\
            .set_index("Indel", drop=False)\
            .join(profile_data.set_index("Indel"), how='left')\
            .reset_index(drop=True)
        os.makedirs(output_dir, exist_ok=True)
        merged_data.to_pickle(output_dir + s)
        completed += 1
    print("completed {}, {}".format(completed, output_dir))
if __name__ == "__main__":
    datasets = [
        ("FORECasT", "train"),
        ("FORECasT", "test"),
        ("FORECasT", "TREX_A_test"),
        ("FORECasT", "HAP1_test"),
        ("LUMC", "WT"),
        ("LUMC", "POLQ"),
        ("LUMC", "KU80"),
        ("LUMC", "LIG4"),
        ("inDelphi", "0226-PRLmESC-Lib1-Cas9_transfertest"),
        ("inDelphi", "0105-mESC-Lib1-Cas9-Tol2-BioRep2-techrep1"),
        ("inDelphi", "052218-U2OS-+-LibA-postCas9-rep1_transfertest")
    ]
    
    d = sys.argv[1]
    t = sys.argv[2]

    if (d, t) not in datasets:
        print("dataset does not exist", d, t)
        exit()

    prepare_dataset(d, t)
    # prepare_dataset("FORECasT", "train")
    # prepare_dataset("FORECasT", "test")
    # prepare_dataset("FORECasT", "TREX_A_test")
    # prepare_dataset("FORECasT", "HAP1_test")
    # prepare_dataset("LUMC", "WT")
    # prepare_dataset("LUMC", "POLQ")
    # prepare_dataset("LUMC", "KU80")
    # prepare_dataset("LUMC", "LIG4")
    prepare_dataset("inDelphi", "052218-U2OS-+-LibA-postCas9-rep1_transfertest")
    prepare_dataset("inDelphi", "0226-PRLmESC-Lib1-Cas9_transfertest")
    # prepare_dataset("inDelphi", "0105-mESC-Lib1-Cas9-Tol2-BioRep2-techrep1")
