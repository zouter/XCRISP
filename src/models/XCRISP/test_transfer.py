import sys, os
import torch
import pickle as pkl
import numpy as np
import pandas as pd
from Bio.Seq import Seq
from tqdm import tqdm
from src.models.XCRISP.transfer import load_model, NeuralNetwork, TransferNeuralNetwork, TransferNeuralNetworkWithOneExtraHiddenLayer, TransferNeuralNetworkOneFrozenLayer
from src.models.XCRISP.model import FEATURE_SETS
from src.config.test_setup  import read_test_file, TRANSFER_TEST_FILES
from src.data.data_loader import get_common_samples
from src.models.Lindel.features import onehotencoder
from tensorflow import keras

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

OUTPUT_DIR = os.environ['OUTPUT_DIR'] if 'OUTPUT_DIR' in os.environ else "./data/Transfer"
INPUT_F = OUTPUT_DIR + "/model_training/data_100x/X-CRISP/{}.pkl"
INSERTION_MODEL_F = "./src/models/Lindel/models/100x_{}_{}_{}_insertion.h5"
INDEL_MODEL_F = "./src/models/Lindel/models/100x_{}_{}_{}_indel.h5"
PREDICTIONS_DIR = OUTPUT_DIR + "model_predictions/X-CRISP/"
VERBOSE = False

def load_data(dataset="test", num_samples = None):
    data = pd.read_pickle(INPUT_F.format(dataset))
    counts = data["counts"]
    del_features = data["del_features"]
    ins_features = data["ins_features"]
    samples = counts.index.levels[0]
    if num_samples is not None:
        samples = samples[:num_samples]
    del_features["Gap"] = del_features["Size"] - del_features["homologyLength"]
    X_del = del_features
    X_ins = ins_features
    y = counts
    return X_del, X_ins, y, samples

def run():
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    exp = "v4"
    loss = "Base"
    state = 1
    mode = sys.argv[1]
    num_samples = int(sys.argv[2])

    if mode not in ["pretrained", "baseline", "pretrainedsamearch", "pretrainedplusonelayer", "pretrainedonefrozenlayer", "weightinit"]:
        print("mode is incorrect:", mode)
        exit()

    if num_samples not in [2, 5, 10, 20, 50, 100, 200, 500, 1000, 5600]:
        print("num samples is incorrect:", mode)
        exit()

    for dataset, oligo_file, genotype, genotype_short_name in TRANSFER_TEST_FILES:
        deletion_model = load_model(mode, genotype_short_name, num_samples)
        
        if mode == "baseline":
            insertion_model = keras.models.load_model(INSERTION_MODEL_F.format(mode, genotype_short_name, num_samples))
            indel_model = keras.models.load_model(INDEL_MODEL_F.format(mode, genotype_short_name, num_samples))
        else:
            insertion_model = keras.models.load_model(INSERTION_MODEL_F.format("transfer", genotype_short_name, num_samples))
            indel_model = keras.models.load_model(INDEL_MODEL_F.format("transfer", genotype_short_name, num_samples))

        profiles = {}
        oligos = read_test_file(oligo_file)
        X_del, X_ins, y, samples = load_data(dataset=genotype, num_samples=None)

        # use common samples for experiment consistency
        print("Testing {} on {}".format(mode, num_samples))

        # not sure if I should remove this
        common_samples = get_common_samples(genotype=genotype,min_reads=100)

        for o in tqdm(oligos):
            if o["ID"] not in common_samples: 
                if VERBOSE: print(o["ID"])
                continue
            with torch.no_grad():
                # deletion predictions from our Model
                x = torch.tensor(X_del.loc[o["ID"], FEATURE_SETS[exp]].to_numpy()).float()
                ds = deletion_model(x)
                ds = ds/sum(ds)
                ds = ds.detach().numpy()[:, 0]

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

        predictions_f = PREDICTIONS_DIR + "transfer_kld_{}_{}_RS_1_{}.pkl".format(mode, num_samples, genotype)
        if os.path.exists(predictions_f):
            os.remove(predictions_f)
        else:
            print("{} does not exist, creating new".format(predictions_f))
        pkl.dump(profiles , open(predictions_f, "wb"))  

def run_gold_standard_comparison():
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    for dataset, oligo_file, genotype, genotype_short_name in TRANSFER_TEST_FILES:
        profiles = {}
        oligos = read_test_file(oligo_file)
        X_del, X_ins, y, samples = load_data(dataset=genotype, num_samples=None)
        X2_del, X2_ins, y2, samples2 = load_data(dataset=genotype.replace("rep1", "rep2"), num_samples=None)

        for o in tqdm(oligos):
            if o["ID"] not in samples: 
                if VERBOSE: print(1, o["ID"])
                continue

            if o["ID"] not in samples2: 
                if VERBOSE: print(2, o["ID"])
                continue

            ins_labels = ['1+A', '1+T', '1+C', '1+G', '2+AA', '2+AT', '2+AC', '2+AG', '2+TA', '2+TT', '2+TC', '2+TG', '2+CA', '2+CT', '2+CC', '2+CG', '2+GA', '2+GT', '2+GC', '2+GG', '3+X']
            y_obs = y.loc[o["ID"]]
            indels = list(y_obs.index[y_obs.Type == "DELETION"]) + ins_labels
            y_obs = y_obs.loc[indels]
            y_hat = y2.loc[o["ID"]]
            y_hat = y_hat.loc[indels]

            profiles[o["ID"]] = {
                "predicted": y_hat["countEvents"]/y_hat["countEvents"].sum(),
                "actual": y_obs["countEvents"]/y_obs["countEvents"].sum(),
                "indels": indels,
                "mh": (X_del.loc[o["ID"], "homologyLength"] > 0).to_list()
            } 

        predictions_f = PREDICTIONS_DIR + "replicate_{}.pkl".format(genotype)
        if os.path.exists(predictions_f):
            os.remove(predictions_f)
        else:
            print("{} does not exist, creating new".format(predictions_f))
        pkl.dump(profiles , open(predictions_f, "wb"))  

def run_wild_type_mESC_comparison():
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    for dataset, oligo_file, genotype, genotype_short_name in TRANSFER_TEST_FILES:
        profiles = {}
        oligos = read_test_file(oligo_file)
        X_del, X_ins, y, samples = load_data(dataset=genotype, num_samples=None)

        mESC_samples = "test" if dataset == "FORECasT" else "0105-mESC-Lib1-Cas9-Tol2-BioRep2-techrep1" 
        X2_del, X2_ins, y2, samples2 = load_data(dataset=mESC_samples, num_samples=None)

        for o in tqdm(oligos):
            if o["ID"] not in samples: 
                if VERBOSE: print(1, o["ID"])
                continue

            if o["ID"] not in samples2: 
                if VERBOSE: print(2, o["ID"])
                continue

            ins_labels = ['1+A', '1+T', '1+C', '1+G', '2+AA', '2+AT', '2+AC', '2+AG', '2+TA', '2+TT', '2+TC', '2+TG', '2+CA', '2+CT', '2+CC', '2+CG', '2+GA', '2+GT', '2+GC', '2+GG', '3+X']
            y_obs = y.loc[o["ID"]]
            indels = list(y_obs.index[y_obs.Type == "DELETION"]) + ins_labels
            y_obs = y_obs.loc[indels]
            y_hat = y2.loc[o["ID"]]
            y_hat = y_hat.loc[indels]

            profiles[o["ID"]] = {
                "predicted": y_hat["countEvents"]/y_hat["countEvents"].sum(),
                "actual": y_obs["countEvents"]/y_obs["countEvents"].sum(),
                "indels": indels,
                "mh": (X_del.loc[o["ID"], "homologyLength"] > 0).to_list()
            } 

        predictions_f = PREDICTIONS_DIR + "mESC_WT_{}.pkl".format(genotype)
        if os.path.exists(predictions_f):
            os.remove(predictions_f)
        else:
            print("{} does not exist, creating new".format(predictions_f))
        pkl.dump(profiles , open(predictions_f, "wb"))  

if __name__ == "__main__":
    run()
    # run_gold_standard_comparison()
    # run_wild_type_mESC_comparison()
    print("Done.")
