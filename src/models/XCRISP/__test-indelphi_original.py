import sys, os
import torch
import warnings
import pickle as pkl
import numpy as np
import pandas as pd
from Bio.Seq import Seq
from tqdm import tqdm
from src.models.XCRISP.__indelphi_original import load_model, load_data, DualNeuralNetwork
from src.models.XCRISP.__indelphi_kNN import predict, load_insertion_data, NUCLEOTIDES
from src.config.test_setup import read_test_file, TEST_FILES, MIN_NUMBER_OF_READS
from src.data.data_loader import get_common_samples

OUTPUT_DIR = os.environ['OUTPUT_DIR']
PREDICTIONS_DIR = OUTPUT_DIR + "model_predictions/X-CRISP/"
MIN_NUM_READS = MIN_NUMBER_OF_READS

def run():
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    model = load_model(min_num_reads=MIN_NUM_READS)
    for dataset, oligo_file, genotype in TEST_FILES:
        print("Running indelphi tests on", genotype)
        profiles = {}
        oligos = read_test_file(oligo_file)
        X_mh, y_mh, y_mh_less, samples = load_data(dataset = genotype, fractions=False, num_samples=None)
        X_ins, y_ins, _ = load_insertion_data(dataset = genotype, num_samples=None, fractions=False)
        # use common samples for experiment consistency
        common_samples = get_common_samples(genotype=genotype,min_reads=MIN_NUMBER_OF_READS)
        oligos = [o for o in oligos if o["ID"] in common_samples]
        print("Testing indelphi on {}".format(genotype))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for o in tqdm(oligos):
                if o["ID"] ==  "overbeek_spacer_9_GGGAGGGCTGTGCTGCTAGT":
                    print(o["ID"])
                if o["ID"] not in samples: continue

                X_mh_s = X_mh.loc[o["ID"]]
                X_mh_less_s = y_mh_less.loc[o["ID"]].index.to_numpy()

                a, b, c = model._predict(X_mh_s)

                (y_mh_phi, mh_indels) = a
                (y_mh_less_phi, del_lens) = b 
                (y_mh_full_phi, mh_full_indels) = c 

                y_pred = torch.cat((y_mh_phi, y_mh_less_phi))
                y_pred = y_pred/sum(y_pred)
                y_pred = y_pred.detach().numpy().flatten()

                x_ins = X_ins.loc[[o["ID"]]]
                y_pred_ins = predict(x_ins)
                y_pred = np.concatenate((y_pred*(1-y_pred_ins.sum()), y_pred_ins))

                y_obs = np.concatenate((y_mh[o["ID"]].to_numpy(), y_mh_less.loc[pd.MultiIndex.from_product([[o["ID"]], del_lens]), "countEvents"].to_numpy(), y_ins.loc[o["ID"], NUCLEOTIDES].to_numpy()))
                y_obs = y_obs/y_obs.sum()
                profiles[o["ID"]] = {
                    "predicted": y_pred,
                    "actual": y_obs,
                    "indels": y_mh[o["ID"]].index.to_list() + [str(dl) for dl in del_lens] + ["1+A", "1+C", "1+G", "1+T"],
                    "mh": [True] * len(y_mh[o["ID"]]) + [False] * (len(del_lens) + 4)
                } 
        predictions_output = PREDICTIONS_DIR + "predictions_{}x_{}_indelphi.pkl".format(MIN_NUM_READS, genotype)
        print(f"Outputing {len(profiles)} predictions to {predictions_output}")
        pkl.dump(profiles , open(predictions_output, "wb"))  

if __name__ == "__main__":
    run()
    print("Done.")
