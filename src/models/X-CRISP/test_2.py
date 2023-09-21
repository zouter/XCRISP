import sys, os
import torch
import pickle as pkl
import numpy as np
import pandas as pd
from Bio.Seq import Seq
from tqdm import tqdm
if sys.argv[1] == "model_2":
    from model_2 import load_model, load_data, NeuralNetwork, MH_FEATURES, MH_LESS_FEATURES
if sys.argv[1] == "model_2_homologygap":
    from model_2_homologygap import load_model, load_data, NeuralNetwork, MH_FEATURES, MH_LESS_FEATURES
if sys.argv[1] == "model_2_ndcg":
    from model_2_ndcg import load_model, load_data, NeuralNetwork, MH_FEATURES, MH_LESS_FEATURES
if sys.argv[1] == "model_2_sparsemax":
    from model_2_sparsemax import load_model, load_data, NeuralNetwork, MH_FEATURES, MH_LESS_FEATURES

from activation.sparsemax import Sparsemax
sparsemax = Sparsemax()

sys.path.append("../")
from test_setup import read_test_file, TEST_FILES, MIN_NUMBER_OF_READS
from data_loader import get_common_samples

from Lindel.features import onehotencoder
from tensorflow import keras

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

OUTPUT_DIR = os.environ['OUTPUT_DIR']
INPUT_F = OUTPUT_DIR + "/model_training/data_100x/X-CRISP/{}.pkl" 
INSERTION_MODEL_F = "../Lindel/models/100x_insertion.h5"
INDEL_MODEL_F = "../Lindel/models/100x_indel.h5"
PREDICTIONS_DIR = OUTPUT_DIR + "model_predictions/X-CRISP/"
MIN_NUM_READS = MIN_NUMBER_OF_READS
VERBOSE = True
RANDOM_STATE = 1

def load_data(dataset="test", num_samples = None):
    data = pd.read_pickle(INPUT_F.format(dataset))
    counts = data["counts"]
    del_features = data["del_features"]
    ins_features = data["ins_features"]
    samples = counts.index.levels[0]
    if num_samples is not None:
        samples = samples[:num_samples]
    del_features[["homologyLeft", "homologyRight", "homologyDist"]] = del_features[["leftEdge", "rightEdge", "Size"]]
    idx_mh = del_features["homologyLength"] > 0
    del_features.loc[(del_features["homologyLength"] > 0) & (del_features["homologyLength"] != del_features["homologyDist"]), MH_LESS_FEATURES] = 0
    del_features.loc[~idx_mh, MH_FEATURES] = 0
    X_del = del_features
    X_ins = ins_features
    y = counts

    return X_del, X_ins, y, samples

def run():
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    model = load_model(random_state=RANDOM_STATE)
    insertion_model = keras.models.load_model(INSERTION_MODEL_F)
    indel_model = keras.models.load_model(INDEL_MODEL_F)
    for dataset, oligo_file, genotype in TEST_FILES:
        X_del, X_ins, y, samples = load_data(dataset=genotype, num_samples=None)

        profiles = {}
        oligos = read_test_file(oligo_file)

        # use common samples for experiment consistency
        common_samples = get_common_samples(genotype=genotype,min_reads=MIN_NUM_READS)
        oligos = [o for o in oligos if o["ID"] in common_samples]
        print("Testing {}".format(genotype))

        for o in tqdm(oligos):
            if o["ID"] not in samples: 
                if VERBOSE: print(o["ID"])
                continue
            with torch.no_grad():
                x = torch.tensor(X_del.loc[o["ID"], MH_FEATURES + MH_LESS_FEATURES].to_numpy()).float()
                ds = model(x)
                if sys.argv[1] == "model_2_sparsemax":
                    sparsemax(ds[:,0])
                else:
                    ds = (ds/sum(ds))[:, 0]
                ds = ds.detach().numpy()

                # insertion predictions from Lindel
                seq = o["TargetSequence"][o["PAM Index"]-33:o["PAM Index"] + 27]
                pam = {'AGG':0,'TGG':0,'CGG':0,'GGG':0}
                guide = seq[13:33]
                if seq[33:36] not in pam:
                    return ('Error: No PAM sequence is identified.')
                input_indel = onehotencoder(guide)
                input_ins   = onehotencoder(guide[-6:])
                dratio, insratio = indel_model.predict(np.matrix(input_indel))[0,:]
                ins = insertion_model.predict(np.matrix(input_ins))[0,:]

                # combine predictions from both models
                y_hat = np.concatenate((ds*dratio,ins*insratio),axis=None)

                # get labels
                ins_labels = ['1+A', '1+T', '1+C', '1+G', '2+AA', '2+AT', '2+AC', '2+AG', '2+TA', '2+TT', '2+TC', '2+TG', '2+CA', '2+CT', '2+CC', '2+CG', '2+GA', '2+GT', '2+GC', '2+GG', '3+X']
                y_obs = y.loc[o["ID"]]
                indels = list(y_obs.index[y_obs.Type == "DELETION"]) + ins_labels
                y_obs = y_obs.loc[indels]

                profiles[o["ID"]] = {
                    "predicted": y_hat,
                    "actual": y_obs["countEvents"]/y_obs["countEvents"].sum(),
                    "indels": indels,
                    "mh": (X_del.loc[o["ID"], "homologyLength"] > 0).to_list()
                } 
            print(len(profiles))

        predictions_f = PREDICTIONS_DIR + "{}_RS_{}_{}.pkl".format(sys.argv[1], RANDOM_STATE, genotype)
        if os.path.exists(predictions_f):
            os.remove(predictions_f)
        else:
            print("{} does not exist, creating new".format(predictions_f))
        pkl.dump(profiles , open(predictions_f, "wb"))  

if __name__ == "__main__":
    run()
    print("Done.")
