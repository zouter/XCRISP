import os, csv, io, sys
import pandas as pd
import numpy as np
import pickle as pkl
from tqdm import tqdm
from model import load, read_theta, compute_predicted_profile
from profile import get_profile_counts

sys.path.append("../")
from test_setup import read_test_file, TEST_FILES, MIN_NUMBER_OF_READS
from data_loader import get_common_samples

MIN_NUM_READS = MIN_NUMBER_OF_READS
OUTPUT_DIR = os.environ['OUTPUT_DIR']
TRAINING_DATA_DIR = OUTPUT_DIR + "model_training/data_{}x/{}/{}/"
PREPROCESSING_OPTIONS = ["FORECasT", "Tijsterman_Analyser"]
THETA = "{}x_model_thetas_l20.001_l10.001".format(MIN_NUM_READS)
MODEL_THETA_FILE = "./models/{}.txt".format(THETA)
PREDICTIONS_DIR = OUTPUT_DIR + "model_predictions/FORECasT/"

# _, BLACKLIST, _ = read_theta(MODEL_THETA_FILE)

def write_predicted_profile_to_summary(p1, fout):
    counts = get_profile_counts(p1)
    for cnt,indel,_,_ in counts:
        if cnt < 0.5: break
        fout.write(u'%s\t-\t%d\n' % (indel, np.round(cnt)))

def write_profiles_to_file(out_prefix, profiles):
    fout = io.open(out_prefix + '_predictedindelsummary.txt', 'w')
    for guide_id in profiles:
        prof = profiles[guide_id]
        if len(profiles) > 1: 
            id_str = u'@@@%s\n' % guide_id
            fout.write(id_str)
        write_predicted_profile_to_summary(prof, fout)
    fout.close()

def run(dataset, test_file, genotype, preprocessing):
    theta, _, feature_columns = read_theta(MODEL_THETA_FILE)
    samples = read_test_file(test_file)
    common_samples = get_common_samples(genotype=genotype,min_reads=MIN_NUMBER_OF_READS)
    samples = [o for o in samples if o["ID"] in common_samples]
    data = {}
    profiles = {}
    for o in tqdm(samples):
        oligo_id = o["ID"]
        try:
            data[oligo_id] = pd.read_pickle(TRAINING_DATA_DIR.format(MIN_NUM_READS, genotype, preprocessing) + oligo_id)
        except FileNotFoundError:
            print("FNFError " + oligo_id)
            continue
        
        
        obs = data[oligo_id].set_index('Indel')
        # if (dataset=="FORECasT") and ("Counts" not in obs.columns): continue
        feature_data = data[oligo_id]
        p, _ = compute_predicted_profile(feature_data, theta, feature_columns)
        indels = list(p.keys())
        # indels = [i for i in indels if i[0] == "D"]
        # counts = [obs.loc[i]['Counts'] for i in indels]
        # if (dataset=="FORECasT") and (sum(counts)< 1000):
        #     continue
        y_pred = [p[i] for i in indels]
        y_obs = [obs.loc[i]['Counts'] for i in indels]
        profiles[oligo_id] = {
            "predicted": y_pred/sum(y_pred),
            "actual": y_obs,
            "indels": indels,
            "mh": ["C" in a for a in indels]
        }
    print(len(profiles))
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    pkl.dump(profiles , open(PREDICTIONS_DIR + "predictions_{}x_{}.pkl".format(MIN_NUM_READS, genotype), "wb"))  

if __name__ == "__main__":
    for dataset, oligo_file, genotype in TEST_FILES: 
        preprocessing = "Tijsterman_Analyser"

        if preprocessing not in PREPROCESSING_OPTIONS:
            print("Select a valid preprocessing option")
            sys.exit()

        run(dataset, oligo_file, genotype, preprocessing)
