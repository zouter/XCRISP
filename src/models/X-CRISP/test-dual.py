import sys, os
import torch
import pickle as pkl
import numpy as np
from Bio.Seq import Seq
from tqdm import tqdm
from model_dual_series import load_model, load_data, MHNeuralNetwork, MHLessNeuralNetwork, predict
from features import DELETION_FEATURES

sys.path.append("../")
from test_setup import read_test_file, TEST_FILES, MIN_NUMBER_OF_READS
from data_loader import get_common_samples

OUTPUT_DIR = os.environ['OUTPUT_DIR']
PREDICTIONS_DIR = OUTPUT_DIR + "model_predictions/X-CRISP/"
MIN_NUM_READS = MIN_NUMBER_OF_READS
VERBOSE = True

def run():
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    model = load_model()
    for dataset, oligo_file, genotype in TEST_FILES:
        profiles = {}
        oligos = read_test_file(oligo_file)
        X, y, samples = load_data(dataset = genotype)
        # use common samples for experiment consistency
        common_samples = get_common_samples(genotype=genotype,min_reads=MIN_NUM_READS)
        oligos = [o for o in oligos if o["ID"] in common_samples]
        print("Testing dual NN on {}".format(genotype))
        for o in tqdm(oligos):
            if o["ID"] not in samples: 
                if VERBOSE: print(o["ID"])
                continue
            with torch.no_grad():
                y_pred = predict(X.loc[o["ID"],:], model)
                y_pred = y_pred/sum(y_pred)
                y_obs = y[o["ID"]]
                profiles[o["ID"]] = {
                    "predicted": y_pred,
                    "actual": y_obs,
                    "indels": y_obs.index.to_list(),
                    "mh": (X.loc[o["ID"], "homologyLength"] > 0).to_list()
                } 
        print(len(profiles))
        predictions_f = PREDICTIONS_DIR + "predictions_{}x_{}_dual_series.pkl".format(MIN_NUM_READS, genotype)
        if os.path.exists(predictions_f):
            os.remove(predictions_f)
        else:
            print("{} does not exist, creating new".format(predictions_f))
        pkl.dump(profiles , open(predictions_f, "wb"))  

if __name__ == "__main__":
    run()
    print("Done.")
