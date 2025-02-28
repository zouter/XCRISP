# mpiexec -n 3 /Users/colm/anaconda3/envs/X-CRISP/bin/python interpretability_insertion.py test

import sys, os, random
import torch
print(torch.__version__)
from src.data.data_loader import get_common_samples
from src.config.test_setup import MIN_NUMBER_OF_READS
from tensorflow import keras
import pickle as pkl
import pandas as pd
import numpy as np
import shap
from sklearn.linear_model import LogisticRegression
from shap.utils._legacy import DenseData

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# model and dataset loading code
seed = 1
test_genotype = "test"
model_to_explain = sys.argv[1]
# model_to_explain = "indel"
train_genotype = "train"
output_dir = os.environ['OUTPUT_DIR']
data_dir = output_dir + "model_training/data_100x/Lindel/Tijsterman_Analyser/"
model_f = "./models/Lindel/100x_{}.h5".format(model_to_explain)
background_samples_num = 5000
samples_to_explain_num = 400

def get_insertion_labels():
    '''convert to single and di-nucleotide hotencode'''
    nt= ['A','T','C','G']
    head = []
    l = 20
    for k in range(l):
        for i in range(4):
            head.append(nt[i]+str(k))

    for k in range(l-1):
        for i in range(4):
            for j in range(4):
                head.append(nt[i]+nt[j]+str(k))
    return head

def sample_data(dataset="train", samples=None):
    data_f = data_dir + dataset
    data = pkl.load(open(data_f, 'rb'))
    if isinstance(samples, int):
        background_samples = random.sample(list(data.keys()), samples)
    if isinstance(samples, list) or isinstance(samples, np.ndarray):
        background_samples = samples
    background = []
    idx_features = 3 if model_to_explain == "insertion" else 2 # indel model
    insertion_labels = get_insertion_labels()
    if model_to_explain == "insertion":
        insertion_labels = insertion_labels[56:80] + insertion_labels[-80:]

    for b in background_samples:
        background.append(data[b][idx_features])
    
    background = pd.DataFrame(background, columns=insertion_labels, index=background_samples)
    return background

class LogisticPredictionModel(LogisticRegression):
    """
    This model is for prediction only.  It has no fit method.
    You can initialize it with fixed values for coefficients 
    and intercepts.  

    Parameters
    ----------
    coef, intercept : arrays
        See attribute descriptions below.

    Attributes
    ----------
    coef_ : array of shape (n_features, ) or (n_targets, n_features)
        Coefficients of the linear model.  If there are multiple targets
        (y 2D), this is a 2D array of shape (n_targets, n_features), 
        whereas if there is only one target, this is a 1D array of 
        length n_features.
    intercept_ : float or array of shape of (n_targets,)
        Independent term in the linear model.
    """

    def __init__(self, coef=None, intercept=None, classes=None):
        if coef is not None:
            coef = np.array(coef)
            if intercept is None:
                intercept = np.zeros(coef.shape[0])
            else:
                intercept = np.array(intercept)
            assert coef.shape[0] == intercept.shape[0]
        else:
            if intercept is not None:
                raise ValueError("Provide coef only or both coef and intercept")
        self.intercept_ = intercept
        self.coef_ = coef
        if classes is not None:
            classes = np.array(classes)
            assert classes.shape[0] == intercept.shape[0]
            self.classes_ = np.array(classes)
            self.multi_class = "multinomial"

    def fit(self, X, y):
        """This model does not have a fit method."""
        raise NotImplementedError("model is only for prediction")

if __name__ == "__main__":
    model = keras.models.load_model(model_f)
    # Create Explainer on root, share to all processes
    if rank == 0:
        output_dir = os.environ["OUTPUT_DIR"] + "/model_shap_values/"
        os.makedirs(output_dir, exist_ok=True)
        random.seed(seed)
        background = sample_data(dataset=train_genotype, samples=background_samples_num)
    else:
        background = None
    background = comm.bcast(background, root=0)

    # get model weights and intercepts, and convert to sklearn linear model
    prereq = pkl.load(open("./src/models/Lindel/model_prereq.pkl", 'rb'))
    label,rev_index,features,frame_shift = prereq
    if model_to_explain == "insertion":
        classes = list(rev_index.values())[-21:]
    if model_to_explain == "indel":
        classes = ["Deletion %", "Insertion %"]
    weights = model.layers[0].get_weights()[0]
    biases = model.layers[0].get_weights()[1]
    new_model = LogisticPredictionModel(coef=weights.T, intercept=biases, classes=classes)

    e = shap.LinearExplainer(new_model, np.array(background))
    print('Rank: ', rank, ', explainer: ', e)

    # load test set and scatter amoung processes
    samples_to_explain = []
    if rank == 0:
        common_samples = get_common_samples(genotype=test_genotype, min_reads=100)
        samples_to_explain = common_samples[:samples_to_explain_num]
        samples_to_explain = [samples_to_explain[i:len(samples_to_explain):size] for i in range(size)]
    else: 
        samples_to_explain = None
    
    samples_to_explain = comm.scatter(samples_to_explain, root=0)
    print('Rank: ', rank, ', Samples To Explain: ', samples_to_explain)

    # collect these shap values and save them 
    results = []
    X_test = sample_data(dataset = test_genotype, samples=samples_to_explain)
    shap_values = np.array(e.shap_values(np.array(X_test.loc[samples_to_explain])))
    
    results = comm.gather(shap_values, root=0)
    if len(results) == 1:
        results = results[0]
    else:
        results = np.concatenate(results, axis=1)

    samples_to_explain = comm.gather(samples_to_explain, root=0)
    if len(samples_to_explain) == 1:
        samples_to_explain = samples_to_explain[0]
    else:
        samples_to_explain = np.concatenate(samples_to_explain, axis=0)

    if rank == 0:
        print(results.shape)
        print(samples_to_explain.shape)
        output_f = output_dir + "{}_{}.shap.tsv".format(model_to_explain, test_genotype)
        pkl.dump((results, samples_to_explain, X_test.columns, \
            list(rev_index.values())[-21:]) if model_to_explain == "insertion" else ["Deletion %", "Insertion %"], open(output_f, "wb"))
        print("Shap values saved to " + output_f)
