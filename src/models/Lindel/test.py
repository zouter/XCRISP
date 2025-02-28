import os, sys
import numpy as np
import pickle as pkl
from tqdm import tqdm
from Bio.Seq import Seq
from tensorflow import keras

from src.models.Lindel.features import onehotencoder, create_feature_array
from src.models.Lindel.indels import gen_indel, gen_cmatrix

from src.config.test_setup import read_test_file, TEST_FILES, MIN_NUMBER_OF_READS
from src.data.data_loader import get_common_samples

MIN_NUM_READS = MIN_NUMBER_OF_READS
PREPROCESSING = "Tijsterman_Analyser"
OUTPUT_DIR = os.environ['OUTPUT_DIR']
DATA_DIR = OUTPUT_DIR + "model_training/data_{}x/Lindel/Tijsterman_Analyser/".format(MIN_NUM_READS)
PREDICTIONS_DIR = OUTPUT_DIR + "model_predictions/Lindel/"
INSERTION_MODEL_F = "./models/Lindel/{}x_insertion.h5".format(MIN_NUM_READS)
DELETION_MODEL_F = "./models/Lindel/{}x_deletion.h5".format(MIN_NUM_READS)
INDEL_MODEL_F = "./models/Lindel/{}x_indel.h5".format(MIN_NUM_READS)


def read_prereq():
    return pkl.load(open("./src/models/Lindel/model_prereq.pkl", 'rb'))

def gen_prediction(seq, models, prereq):
    '''generate the prediction for all classes, redundant classes will be combined'''
    pam = {'AGG':0,'TGG':0,'CGG':0,'GGG':0}
    guide = seq[13:33]
    if seq[33:36] not in pam:
        return ('Error: No PAM sequence is identified.')
    indel_model, deletion_model, insertion_model = models
    label,rev_index,features,frame_shift = prereq
    indels = gen_indel(seq,30) 
    input_indel = onehotencoder(guide)
    input_ins   = onehotencoder(guide[-6:])
    input_del   = np.concatenate((create_feature_array(features,indels),input_indel),axis=None)
    cmax = gen_cmatrix(indels,label) # combine redundant classes
    dratio, insratio = indel_model.predict(np.matrix(input_indel))[0,:]
    ds  = deletion_model.predict(np.matrix(input_del))[0,:]
    ins = insertion_model.predict(np.matrix(input_ins))[0,:]
    y_hat = np.concatenate((ds*dratio,ins*insratio),axis=None) * cmax
    return y_hat, indels

def tf_pearson(a, b):
    return 0

def run():
    prereq = read_prereq()
    _, rev_index, _, _ = prereq
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    indel_model = keras.models.load_model(INDEL_MODEL_F)
    deletion_model = keras.models.load_model(DELETION_MODEL_F, custom_objects={"tf_pearson": tf_pearson})
    insertion_model = keras.models.load_model(INSERTION_MODEL_F, custom_objects={"tf_pearson": tf_pearson})
    models = (indel_model, deletion_model, insertion_model)

    for dataset, oligo_f, genotype in TEST_FILES:
        print("Running predictions on " + genotype)
        samples = read_test_file(oligo_f)
        # use common samples for experiment consistency
        common_samples = get_common_samples(genotype=genotype,min_reads=MIN_NUM_READS)
        samples = [o for o in samples if o["ID"] in common_samples]

        data_f = DATA_DIR + genotype
        data = pkl.load(open(data_f, 'rb'))
        
        profiles = {}
        for s in tqdm(samples):
            try:
                seq = s["TargetSequence"]
                cutsite = s["PAM Index"] - 3
                seq = seq[cutsite-30: cutsite+35]
                y_pred, indels = gen_prediction(seq, models, prereq)
                y_obs = data[s["ID"]][1]
                mhs = ["{}+{}".format(i[4], i[5]) for i in indels if i[-2] == "mh"]
                all_indels = list(rev_index.values())

                profiles[s["ID"]] = {
                    "predicted": y_pred/sum(y_pred),
                    "actual": y_obs,
                    "indels": all_indels,
                    "mh": [a in mhs for a in all_indels]
                }
            except KeyError:
                print("KeyError ", s["ID"])
                continue
        
        print(len(profiles))
        pkl.dump(profiles , open(PREDICTIONS_DIR + "Lindel_{}.pkl".format(genotype), "wb"))  

if __name__ == "__main__":
    run()
    print("Done.")
