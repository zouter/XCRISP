import os, sys
from numpy import correlate
import pandas as pd

transfer_fasta_test_file = "transfertest.fasta"
transfer_fasta_train_file = "transfertrain.fasta"
OURMODEL_DIR = os.environ["OUTPUT_DIR"] + "model_training/data_100x/X-CRISP/"
correctable = ["0226-PRLmESC-Lib1-Cas9", "052218-U2OS-+-LibA-postCas9-rep1", "TREX_A", "HAP1"]

def get_guides_from_fasta(oligo_f):
    guides = []
    with open(oligo_f, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            guides.append(lines[i].split()[0][1:])
    return guides    

for d in [OURMODEL_DIR]:
    files = os.listdir(d)

    for f in files:
        prefix = f.replace(".pkl", "").replace("_train", "")
        if prefix not in correctable:
            continue

        if "train" not in f:
            continue

        train_data = pd.read_pickle(d + f)
        test_data = pd.read_pickle(d + f.replace("train", "test"))

        all_data = pd.concat([train_data, test_data])

        new_train_guides = get_guides_from_fasta(transfer_fasta_train_file)
        new_test_guides = get_guides_from_fasta(transfer_fasta_test_file)

        new_train_data = all_data.loc[new_train_guides]
        new_test_data = all_data.loc[new_test_guides]

        print("New train data shape:", new_train_data.shape)
        print("New test data shape:", new_test_data.shape)

        if sys.argv[1] != "dry":
            new_train_data.to_pickle(d + f.replace("train", "transfertrain"))
            new_test_data.to_pickle(d + f.replace("train", "transfertest"))    





