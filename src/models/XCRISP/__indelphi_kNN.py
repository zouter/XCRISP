import sys, os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import dump, load

from src.models.XCRISP.__indelphi_original import load_model, DualNeuralNetwork
from src.data.data_loader import get_common_samples, get_details_from_fasta
from src.config.test_setup import MIN_NUMBER_OF_READS

TRAIN_GENOTYPE = "train"
# TRAIN_GENOTYPE = "0105-mESC-Lib1-Cas9-Tol2-BioRep2-techrep1"
OUTPUT_DIR = os.environ['OUTPUT_DIR']
INPUT_F = OUTPUT_DIR + "/model_training/data_{}x".format(MIN_NUMBER_OF_READS) + "/X-CRISP/{}.pkl" 
DELETION_MODEL = load_model(experiment_name="inDelphi", min_num_reads=MIN_NUMBER_OF_READS)
NUCLEOTIDES = ["A", "C", "G", "T"]
KNN_FEATURES = ["total_del_phi", "precision"] + ["-3" + n for n in NUCLEOTIDES] + ["-4" + n for n in NUCLEOTIDES]
SCALER_F = "./src/models/XCRISP/models/preprocessing/knn_scaler.bin"
MODEL_F = "./src/models/XCRISP/models/kNN_indelphi.bin"
NUCLEOTIDE_LOOKUP_F = "./src/models/XCRISP/models/preprocessing/indelphi_1bp_lookup.bin"
RANDOM_STATE = 1

def precision_score(y):
    y = y/y.sum()
    s = 1 - (-np.sum(y * np.log(y))/np.log(len(y)))
    return s

def get_fasta_file_for_dataset(d):
    if d in ["train", "test"]:
        oligo_f = "FORECasT/{}.fasta".format(d)
    elif "_test" in d:
        oligo_f = "FORECasT/test.fasta"
    elif "_transfertest" in d:
        oligo_f =  "inDelphi/LibA.forward.fasta"
    elif d == "0105-mESC-Lib1-Cas9-Tol2-BioRep2-techrep1":
        oligo_f = "inDelphi/LibA.forward.fasta"
    else:
        oligo_f = "LUMC/{}.forward.fasta".format(d)
    
    return get_details_from_fasta("./src/data/{}".format(oligo_f))

def one_hot_encode(nucleotide):
    lookup = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
    }
    return lookup[nucleotide]

def get_deletion_features(X, samples):
    print("Prepping deletion features")
    x_mh = X.loc[X.homologyLength != 0, ["homologyLength", "homologyGCContent", "Size"]]
    total_del_phi_scores = []
    del_len_precisions = []
    for s in tqdm(samples):
        mh, del_len, mh_full = DELETION_MODEL._predict(x_mh.loc[s])
        phi_mh = mh[0]
        phi_del_len = del_len[0]
        phi_mh_full = mh_full[0]
        
        total_del_phi_scores.append(phi_mh.sum() + phi_del_len.sum())
        del_len_precisions.append(precision_score(np.concatenate((phi_mh_full, phi_del_len))))

    del_features = pd.DataFrame({
        "total_del_phi": total_del_phi_scores,
        "precision": del_len_precisions
    }, index=samples)
    return del_features


def get_sequence_features(dataset, samples):
    print("Prepping Sequence features")
    oligos = get_fasta_file_for_dataset(dataset)
    minus3bases = []
    minus4bases = []
    minus5bases = []
    references = []
    for s in tqdm(samples):
        o = oligos[s]
        guide = o["TargetSequence"][o["PAM Index"]-20:o["PAM Index"]]
        minus3bases.append(one_hot_encode(guide[-3]))
        minus4bases.append(one_hot_encode(guide[-4]))
        minus5bases.append(one_hot_encode(guide[-5]))
        references.append(guide[-5:-2])


    minus3bases = np.array(minus3bases)
    minus4bases = np.array(minus4bases)
    minus5bases = np.array(minus5bases)
    references = np.expand_dims(np.array(references), axis=-1)

    bases = np.hstack((minus3bases, minus4bases, minus5bases, references))
    base_labels = []
    for pos in ["-3", "-4", "-5"]:
        for n in NUCLEOTIDES:
            base_labels.append(pos+n)
    base_labels.append("reference")
    seq_features = pd.DataFrame(bases, index=samples, columns=base_labels)
    return seq_features

def get_1bp_insertions(counts, samples, fractions=True):
    insertions = ["1+T", "1+A", "1+C", "1+G"]
    del_counts = counts.loc[counts.Type == "DELETION"].loc[samples].reset_index()
    ins_counts = counts.loc[counts.Type == "INSERTION"].loc[samples].reset_index()[["Sample_Name", "Indel", "fraction" if fractions else "countEvents"]]
    ins_counts = ins_counts[ins_counts.Indel.isin(insertions)]
    ins_counts.Indel = ins_counts.Indel.str[2]
    ins_counts = ins_counts.set_index(["Sample_Name", "Indel"]).unstack(level=-1)
    ins_counts.columns = ins_counts.columns.droplevel()
    total_ins = ins_counts.sum(axis=1)
    total_dels = del_counts[["Sample_Name",  "fraction" if fractions else "countEvents"]].groupby(["Sample_Name"]).sum().iloc[:,0]
    ins_counts["fraction"] = total_ins / (total_ins + total_dels)
    return ins_counts

def load_insertion_data(dataset = TRAIN_GENOTYPE, num_samples = None, fractions=True):
    data = pd.read_pickle(INPUT_F.format(dataset))
    counts = data["counts"]
    del_features = data["del_features"]
    samples = counts.index.levels[0]
    if num_samples is not None:
        samples = samples[:num_samples]
    del_features = get_deletion_features(del_features, samples)
    sequence_features = get_sequence_features(dataset, samples)
    X = pd.concat((del_features, sequence_features), axis=1)
    y = get_1bp_insertions(counts, samples, fractions)
    return X, y, samples

def predict(X):
    assert(isinstance(X, pd.DataFrame))
    assert(pd.Series(KNN_FEATURES).isin(X.columns).all())

    scaler = load(SCALER_F)
    model = load(MODEL_F)
    nucleotide_lookup = load(NUCLEOTIDE_LOOKUP_F)
    X_scaled = scaler.transform(X[KNN_FEATURES])
    ratio_pred = model.predict(X_scaled)[0]
    nucleotide_frac = nucleotide_lookup.loc[X["reference"]]
    nucleotide_frac = nucleotide_frac * ratio_pred
    return nucleotide_frac[NUCLEOTIDES].to_numpy()[0,:]

if __name__ == "__main__":
    X, y, samples = load_insertion_data(num_samples=None)
    # use common samples for experiment consistency
    common_samples = get_common_samples(genotype=TRAIN_GENOTYPE, min_reads=MIN_NUMBER_OF_READS)
    samples = np.intersect1d(samples, common_samples)
    # split samples into train and validation
    samples, val_samples = train_test_split(samples, test_size = 100, random_state=RANDOM_STATE)
    X_train = X.loc[samples]
    X_val = X.loc[val_samples]
    # standardise columns
    scaler = StandardScaler()
    scaler.fit(X_train[KNN_FEATURES])
    X_train_scaled = scaler.transform(X_train[KNN_FEATURES])
    print(scaler.mean_)
    print(scaler.var_)
    # train kNN model
    model = KNeighborsRegressor()
    model.fit(X_train_scaled, y.loc[samples, "fraction"])
    # create lookup table
    nucleotide_lookup = pd.concat((X_train[["reference"]], y[["A", "C", "T", "G"]]), axis=1).groupby(["reference"]).mean()
    nucleotide_lookup = nucleotide_lookup.div(nucleotide_lookup.sum(axis=1), axis=0)
    # save scaler, kNN model and lookup table
    dump(scaler, SCALER_F ,compress=True)
    dump(model, MODEL_F, compress=True)
    dump(nucleotide_lookup, NUCLEOTIDE_LOOKUP_F, compress=True)
    # predict 
    y_pred = predict(X_val) # verify model was built by predicting 5 rows
    mse = mean_squared_error(y.loc[val_samples, "fraction"], y_pred)
    print("MSE on training set: {:.5f}".format(mse))
    