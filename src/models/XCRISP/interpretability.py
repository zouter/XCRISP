# mpiexec -n 3 /Users/colm/anaconda3/envs/X-CRISP/bin/python interpretability.py test mh

import sys, os
import torch
print(torch.__version__)
from src.data.data_loader import get_common_samples
from src.config.test_setup import MIN_NUMBER_OF_READS
import shap
import random
from sklearn.model_selection import train_test_split
from src.models.XCRISP.deletion import load_model, load_data, NeuralNetwork, FEATURE_SETS, _to_tensor
from tqdm import tqdm
import pandas as pd
import numpy as np

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if __name__ == "__main__":
    # model and dataset loading code
    seed = 1
    test_genotype = sys.argv[1]
    mh_or_mhless = sys.argv[2]
    train_genotype = "train"
    background_samples_num = 10000
    samples_to_explain_num = 400
    features = FEATURE_SETS["v4"]

    # load training examples to get distribution of data
    # code taken from https://github.com/slundberg/shap
    # TODO: model needs to be update to accept a list of examples, and return a list of distributions
    # perhaps this is not necessary. At this point, the model is trained, and the model makes decisions independently on each indel, so I just need to get the scores of each indel seperately
    # and then normalise together. Should make sure the model sees enough samples though
    
    # Create Explainer on root, share to all processes
    if rank == 0:
        model = load_model(model=os.environ["OUTPUT_DIR"] + "/model_training/models/XCRISP/")
        model.eval()
        output_dir = os.environ["OUTPUT_DIR"] + "/model_shap_values/"
        os.makedirs(output_dir, exist_ok=True)
        X_train, _, samples_train = load_data(dataset = train_genotype)
        random.seed(seed)
        # Sample 1000 indels at random from the entire training set
        if mh_or_mhless == "mh":
            idx = X_train[X_train["homologyLength"] > 0].index
        elif mh_or_mhless == "mhless":
            idx = X_train[X_train["homologyLength"] == 0].index
        elif mh_or_mhless == "all":
            idx = X_train.index
        background = X_train.loc[idx, features].sample(background_samples_num)

        # sample certain oligos and include ALL indels for that oligo
        # background = X_train.loc[random.sample(samples_train.to_list(), background_samples_num), features]
        deep_ex_args = (model, _to_tensor(background))
    else:
        deep_ex_args = None
    model, background = comm.bcast(deep_ex_args, root=0)
    e = shap.DeepExplainer(model, background)
    print('Rank: ', rank, ', explainer: ', e)

    # load test set and scatter amoung processes
    X_test, _, samples_test = load_data(dataset = test_genotype)
    if mh_or_mhless == "mh":
        idx = X_test[X_test["homologyLength"] > 0].index
    elif mh_or_mhless == "mhless":
        idx = X_test[X_test["homologyLength"] == 0].index
    elif mh_or_mhless == "all":
        idx = X_test.index
    X_test = X_test.loc[idx]

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
    for s in tqdm(samples_to_explain):
        X_s = X_test.loc[[s], features]
        ex = _to_tensor(X_s)
        shap_values = e.shap_values(ex)
        shap_df = X_s.copy()
        shap_df.loc[[s], features] = shap_values
        results.append(shap_df)
    data = pd.concat(results)
  
    data = comm.gather(data, root=0)
    if rank == 0:
        data = pd.concat(data)
        data.to_csv(output_dir + "deletion_mse_0.05___model__{}.{}.shap.tsv".format(test_genotype, mh_or_mhless), sep="\t")
    else:
        assert data is None
