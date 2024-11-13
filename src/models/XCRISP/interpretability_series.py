# mpiexec -n 3 /Users/colm/anaconda3/envs/X-CRISP/bin/python interpretability_series.py

# TODO: Extract MH and MHLess Net seperately 

import sys, os, time
import torch
print(torch.__version__)
sys.path.append("../")
from data_loader import get_common_samples
from test_setup import MIN_NUMBER_OF_READS
import shap
import random
from sklearn.model_selection import train_test_split
from model_dual_series import load_model, load_data, MHNeuralNetwork, MHLessNeuralNetwork, Predictor, _to_tensor, MH_FEATURES, MH_LESS_FEATURES
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
    test_genotype = "test"
    train_genotype = "train"
    loss = "Base"
    background_samples_num = 10
    samples_to_explain_num = 2

    # load training examples to get distribution of data
    # code taken from https://github.com/slundberg/shap
    # TODO: model needs to be update to accept a list of examples, and return a list of distributions
    # perhaps this is not necessary. At this point, the model is trained, and the model makes decisions independently on each indel, so I just need to get the scores of each indel seperately
    # and then normalise together. Should make sure the model sees enough samples though
    
    # Create Explainer on root, share to all processes
    if rank == 0:
        # setup output folder
        output_dir = os.environ["OUTPUT_DIR"] + "/model_shap_values/"
        os.makedirs(output_dir, exist_ok=True)

        # load models
        predictor = Predictor(load_model())

        # load background data
        X_train, _, samples_train = load_data(dataset = train_genotype)

        # Sample 1000 indels at random from the entire training set
        random.seed(seed)
        background = X_train.loc[:,MH_FEATURES].sample(background_samples_num)

        start_time = time.time()
        explainer = shap.KernelExplainer(predictor.predict_single, background, link="identity")
        print("Explainer created in {} seconds".format(time.time() - start_time))

    else:
        explainer = None
    explainer = comm.bcast(explainer, root=0)
    print('Rank: ', rank, ', Explainer: ', explainer)

    # load test set and scatter amoung processes
    X_test, _, samples_test = load_data(dataset = test_genotype)

    samples_to_explain = []
    if rank == 0:
        common_samples = get_common_samples(genotype=test_genotype, min_reads=MIN_NUMBER_OF_READS)
        samples_to_explain = common_samples[:samples_to_explain_num]
        samples_to_explain = [samples_to_explain[i:len(samples_to_explain):size] for i in range(size)]
    else: 
        samples_to_explain = None
    
    samples_to_explain = comm.scatter(samples_to_explain, root=0)
    print('Rank: ', rank, ', Samples To Explain: ', len(samples_to_explain))

    # collect these shap values and save them 
    results = []
    for s in tqdm(samples_to_explain):
        X_s = X_test.loc[[s], MH_FEATURES]
        shap_values = explainer.shap_values(X_s)
        shap_df = X_s.copy()
        shap_df.loc[[s], MH_FEATURES] = shap_values
        results.append(shap_df)
    data = pd.concat(results)
  
    data = comm.gather(data, root=0)
    if rank == 0:
        data = pd.concat(data)
        data.to_csv(output_dir + "dual_series_{}x_{}.all.shap.tsv".format(MIN_NUMBER_OF_READS, test_genotype), sep="\t")
    else:
        assert data is None
