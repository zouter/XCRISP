import sys, os
import pickle as pkl
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.models import load_model
from model import load_data
sys.path.append("../")
from test_setup import read_test_file, TEST_FILES, MIN_NUMBER_OF_READS
from data_loader import get_common_samples

OUTPUT_DIR = os.environ['OUTPUT_DIR']
PREDICTIONS_DIR = OUTPUT_DIR + "model_predictions/CROTON/"
STATS = ["del_freq", "prob_1bpins", "prob_1bpdel", "one_bp_frameshift", "two_bp_frameshift", "frameshift"]

if __name__ == "__main__":
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    model = load_model("models/CROTON_new.h5")
    for dataset, oligo_file, genotype in TEST_FILES:
        X, y, samples = load_data(dataset=genotype, num_samples=None)

        profiles = {}
        oligos = read_test_file(oligo_file)

        # use common samples for experiment consistency
        common_samples = get_common_samples(genotype=genotype,min_reads=100)
        oligos = np.isin(samples, common_samples)
        print("Testing {}".format(genotype))

        y_pred = model.predict(X[oligos, :, :])
        y = y[oligos, :]
        samples = samples[oligos]

        profiles = {
            "predicted": pd.DataFrame(y_pred, index=samples, columns=STATS),
            "actual": pd.DataFrame(y, index=samples, columns=STATS),
            "samples": samples,
        }
        
        predictions_f = PREDICTIONS_DIR + "{}_new.pkl".format(genotype)
        
        if os.path.exists(predictions_f):
            os.remove(predictions_f)
        print("Outputting predictions to: {}".format(predictions_f))
        pkl.dump(profiles , open(predictions_f, "wb")) 
    print("Done.")
