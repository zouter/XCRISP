from cmath import nan
import sys, os
import pandas as pd
import numpy as np
import pickle as pkl
from tqdm import tqdm
from pandas.api.types import is_number


sys.path.append("../")
from test_setup import read_test_file, TEST_FILES
from data_loader import get_common_samples

import predict


TEST_FILE = "../../data/{}/{}{}.fasta"
OUTPUT_DIR = os.environ['OUTPUT_DIR']
PREDICTIONS_DIR = OUTPUT_DIR + "model_predictions/inDelphi/"
FORECasT = ["train", "test"]


def init_model(run_iter = '', 
               param_iter = 'ado'):
    print('Initializing model %s/%s...' % (run_iter, param_iter))
    
    model_out_dir = './models'

    param_fold = model_out_dir + '%s/parameters/' % (run_iter)
    nn_params = pkl.load(open(param_fold + '%s_nn.pkl' % (param_iter), 'rb'))
    nn2_params = pkl.load(open(param_fold + '%s_nn2.pkl' % (param_iter), 'rb'))

    bp_model = pkl.load(open('%s/bp_model_v2.pkl' % (model_out_dir), 'rb'))
    rate_model = pkl.load(open('%s/rate_model_v2.pkl' % (model_out_dir), 'rb'))
    normalizer = pkl.load(open('%s/Normalizer_v2.pkl' % (model_out_dir), 'rb'))

    return nn_params, nn2_params, bp_model, rate_model, normalizer

def load_observed_data(dataset, genotype):
    data_dir = os.environ["OUTPUT_DIR"] + "model_training/data_100x/inDelphi/"
    ins_1bp_obs = pd.read_csv(data_dir + "1bpins_{}.csv".format(genotype))
    ins_ratio_obs = pd.read_csv(data_dir + "ins_ratio_{}.csv".format(genotype))
    del_obs = pkl.load(open(data_dir + "{}.pkl".format(genotype), "rb"))
    return del_obs, ins_1bp_obs, ins_ratio_obs


def convert_obs_to_dataframe(del_obs, ins_1bp_obs, ins_ratio_obs, target_id):
    idx = del_obs[0].index(target_id)
    del_lens = del_obs[3][idx]
    gt_pos = del_obs[6][idx]
    freqs = del_obs[4][idx]

    mh_obs = pd.DataFrame({
        "Length": del_lens,
        "Genotype Position": gt_pos,
        "Observed_Frequency": freqs,
        "Category": "del",
        "Inserted Bases": np.nan
    })

    dl_freqs = del_obs[5][idx]
    mhless_obs = pd.DataFrame({
        "Length": list(range(1, len(dl_freqs) + 1)),
        "Genotype Position": "e",
        "Observed_Frequency": dl_freqs,
        "Category": "del",
        "Inserted Bases": np.nan
    })

    idx = ins_1bp_obs[ins_1bp_obs["_Experiment"] == target_id].index[0]
    ins_obs = ins_1bp_obs.loc[idx, ["A frac", "C frac", "G frac", "T frac"]]
    ins_ratio = ins_1bp_obs.loc[idx, "Frequency"]

    obs_freq = pd.concat([mh_obs, mhless_obs])
    obs_freq["Observed_Frequency"] = (obs_freq["Observed_Frequency"]/sum(obs_freq["Observed_Frequency"])) * (1 - ins_ratio)

    ins_freq = pd.DataFrame({
        "Length": 1,
        "Genotype Position": np.nan,
        "Observed_Frequency": ins_obs * ins_ratio,
        "Category": "ins",
        "Inserted Bases": ["A", "C", "G", "T"]
    })

    obs_freq = pd.concat([obs_freq, ins_freq])

    return obs_freq


def convert_to_indel(row):
    if row["Genotype Position"] == "e":
        return "DL" + str(row["Length"])
    
    if row["Category"] == "ins":
        return "1+" + row["Inserted Bases"]

    return "{}+{}".format(row["Genotype Position"] - row["Length"], row["Length"])

def run():
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    
    nn_params, nn2_params, bp_model, rate_model, normalizer = init_model()
    
    profiles = {}
    for dataset, oligo_file, genotype in TEST_FILES:
        profiles = {}

        del_obs, ins_1bp_obs, ins_ratio_obs = load_observed_data(dataset, genotype)
        oligos = read_test_file(oligo_file)

        common_samples = get_common_samples(genotype=genotype,min_reads=100)
        oligos = [o for o in oligos if o["ID"] in common_samples]
        print("Testing inDelphi on {}".format(genotype))
        for o in tqdm(oligos):
            cutsite = o["PAM Index"] - 3
            seq = o["TargetSequence"]

            try:
                idx = del_obs[0].index(o["ID"])
                ins_1bp_obs[ins_1bp_obs["_Experiment"] == o["ID"]].index[0]
            except (ValueError, IndexError):
                continue
            # mh_len = del_obs[1][idx]
            # gc_frac = del_obs[2][idx]
            # del_lens = del_obs[3][idx]
            # gt_pos = del_obs[6][idx]
            predictions = predict.predict_all(seq, cutsite, nn_params, nn2_params, rate_model, bp_model, normalizer)[1].set_index(["Length", "Genotype Position", "Category", "Inserted Bases"])
            observed = convert_obs_to_dataframe(del_obs, ins_1bp_obs, ins_ratio_obs, o["ID"]).set_index(["Length", "Genotype Position", "Category", "Inserted Bases"])
            everything = predictions.join(observed).reset_index()
            everything["Observed_Frequency"] = everything["Observed_Frequency"].fillna(0)


            profiles[o["ID"]] = {
                "predicted": everything["Predicted_Frequency"],
                "actual": everything["Observed_Frequency"]/sum(everything["Observed_Frequency"]),
                "indels": everything.apply(convert_to_indel, axis=1).to_list(),
                "mh": (everything["Genotype Position"].notnull() & everything["Genotype Position"].apply(is_number)).to_list(),
            } 

        print(len(profiles))

        predictions_f = PREDICTIONS_DIR + "{}_predictions.pkl".format( genotype)
        if os.path.exists(predictions_f):
            os.remove(predictions_f)
        else:
            print("{} does not exist, creating new".format(predictions_f))
        pkl.dump(profiles , open(predictions_f, "wb"))
             
if __name__ == "__main__":
    run()
    print("Done.")