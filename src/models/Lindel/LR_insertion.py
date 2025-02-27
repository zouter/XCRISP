#!/usr/bin/env python

#System tools 
import pickle as pkl
import os,sys,csv,re,random

from tqdm import tqdm
import numpy as np
np.seterr(all='raise')

import multiprocessing as mp

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import normalize

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input, Flatten
from keras.models import Sequential, load_model
from keras.regularizers import l2, l1
from keras import backend as K

import matplotlib.pyplot as plt

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from src.data.data_loader import get_guides_from_fasta, get_common_samples
from src.config.test_setup import MIN_NUMBER_OF_READS

NUM_FOLDS = 5
RANDOM_STATE = 1
INPUT_SIZE = 104
OUTPUT_SIZE = 21
NUM_TRANSFER_VAL_SAMPLES = 100
MIN_READS_PER_TARGET = MIN_NUMBER_OF_READS
TRANSFER_LEARNING_RATE = 0.001
NUM_EPOCHS = 200

# Define useful functions
def mse(x, y):
    return ((x-y)**2).mean()

def corr(x, y):
    return np.corrcoef(x, y)[0, 1] ** 2

def plot_training_curves(history, prefix):
    output_dir = os.environ["LOGS_DIR"] + "LR_insertion_transfer/"
    os.makedirs(output_dir, exist_ok=True)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    
    plt.savefig(output_dir + prefix + "_training_loss_curve.png")
    plt.clf()

    plt.plot(history.history['mse'])
    plt.plot(history.history['val_mse'])
    plt.title('model mse')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(output_dir + prefix + "_training_mse.png")
    plt.clf()

def train_model(x_train, y_train, reg, x_valid=None, y_valid=None, pretrained_model=None):
    np.random.seed(0)
    model = Sequential()
    model.add(Dense(OUTPUT_SIZE,  activation='softmax', input_shape=(INPUT_SIZE,), kernel_regularizer=reg))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse'])
    if pretrained_model is not None:
        model = pretrained_model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse'])
        K.set_value(model.optimizer.learning_rate, TRANSFER_LEARNING_RATE)
        history = model.fit(x_train, y_train, epochs=NUM_EPOCHS, verbose=1, validation_data=(x_valid, y_valid))
        return model, None, history
    elif x_valid is not None and y_valid is not None:
        history = model.fit(x_train, y_train, epochs=NUM_EPOCHS, validation_data=(x_valid, y_valid), 
                callbacks=[EarlyStopping(patience=1)], verbose=1)
        y_hat = model.predict(x_valid)
        return model, mse(y_hat, y_valid), history
    else:
        history = model.fit(x_train, y_train, epochs=NUM_EPOCHS, verbose=1)
        return model, None, history

def benchmark_models(X, Y, lambdas, split):
    errors_l1 = []
    errors_l2 = []

    x_train = X[split[0], :]
    y_train = Y[split[0], :]
    x_valid = X[split[1], :]
    y_valid = Y[split[1], :]

    for l in tqdm(lambdas):
        _, err, _ = train_model(x_train, y_train, l1(l), x_valid=x_valid, y_valid=y_valid)
        errors_l1.append(err)
        _, err, _ = train_model(x_train, y_train, l2(l), x_valid=x_valid, y_valid=y_valid)
        errors_l2.append(err)    
    return errors_l1, errors_l2

def average_fold_errors(errors):
    means = np.mean(np.array(errors), axis=0)
    min_mean = np.min(means)
    min_index = np.argmin(means)
    return min_mean, min_index

def get_X_and_Y(data):
    Y = np.array([y[1][-21:]/sum(y[1][-21:]) for y in data])
    Y = np.nan_to_num(Y)
    X = np.array([x[3] for x in data])
    return X, Y

def run(data, workdir, prefix="", reg=None, pretrained_model=None, valid_data=None):
    os.makedirs(workdir, exist_ok=True)
    kf = KFold(n_splits=NUM_FOLDS, random_state=RANDOM_STATE, shuffle=True)
    X, Y = get_X_and_Y(data)

    if reg is None:
        folds = list(kf.split(X, Y))
        lambdas = 10 ** np.arange(-10, -1, 0.1) # for activation function test
        errors_l1 = []
        errors_l2 = []
        def log_results(results):
            e1, e2 = results
            errors_l1.append(e1)
            errors_l2.append(e2)
        pool = mp.Pool(5)
        for fold in folds:
            pool.apply_async(benchmark_models, args=(X, Y, lambdas, fold), callback=log_results)    
        pool.close()
        pool.join()

        np.save(workdir+'{}mse_l1_ins.npy'.format(prefix),errors_l1)
        np.save(workdir+'{}mse_l2_ins.npy'.format(prefix),errors_l2)

        best_lambda, reg = get_best_params(errors_l1, errors_l2)
        print(best_lambda, reg)    

    
    if valid_data is None:
        m, err, history = train_model(X, Y, reg, pretrained_model=pretrained_model)
    else:
        X_valid, Y_valid = get_X_and_Y(valid_data)
        m, err, history = train_model(X, Y, reg, pretrained_model=pretrained_model, x_valid=X_valid, y_valid=Y_valid)
    
    m.save(workdir + '{}x_{}insertion.h5'.format(MIN_READS_PER_TARGET, prefix))
    # plot_training_curves(history, prefix)

    print("Done.")

def get_best_params(errors_l1, errors_l2):
    lambdas = 10 ** np.arange(-10, -1, 0.1) # for activation function test
    # find best params and train final model
    min_mean_l1, min_idx_l1 = average_fold_errors(errors_l1)
    min_mean_l2, min_idx_l2 = average_fold_errors(errors_l2)

    best_lambda, reg = None, None
    if min_mean_l1 < min_mean_l2:
        best_lambda = lambdas[min_idx_l1]
        reg = l1(best_lambda)
    else:
        best_lambda = lambdas[min_idx_l2]
        reg = l2(best_lambda)

    return best_lambda, reg

def load_transfer_learning_samples(dataset, n):
    with open("../X-CRISP/transfer_{}.pkl".format(dataset), "rb") as openfile:
        tr_samples = pkl.load(openfile)
    return np.array(tr_samples[n]), np.array(tr_samples["validation"])

def train_baseline_model():
    workdir = os.environ["OUTPUT_DIR"] + "model_training/model/Lindel/"
    data = load_data()
    run(data, workdir)

def transfer_model(genotype, num_samples, pretrained=True):
    mappings = {
        "0226-PRLmESC-Lib1-Cas9_transfertrain": "mESC-NHEJ-deficient",
        "0226-PRLmESC-Lib1-Cas9_transfertest": "mESC-NHEJ-deficient",
        "052218-U2OS-+-LibA-postCas9-rep1_transfertrain": "U2OS",
        "052218-U2OS-+-LibA-postCas9-rep1_transfertest": "U2OS",
        "HAP1_train": "HAP1",
        "HAP1_test": "HAP1",
        "TREX_A_train": "TREX_A",
        "TREX_A_test": "TREX_A",
        "train": "WT",
    }

    num_validation_samples = 100
    test_or_train = genotype.split("_")[-1]
    workdir = os.environ["OUTPUT_DIR"] + "model_training/model/Lindel/"
    data_f = os.environ["OUTPUT_DIR"] + "model_training/data_100x/Lindel/Tijsterman_Analyser/{}".format(genotype)
    data = list(pkl.load(open(data_f, 'rb')).values())
    guides, val_guides = load_transfer_learning_samples(genotype, num_samples)
    train_data = [d for d in data if d[5]["ID"] in guides]
    valid_data = [d for d in data if d[5]["ID"] in val_guides]
    pretrained_model = load_model('./models/{}x_{}insertion.h5'.format(MIN_READS_PER_TARGET, ""))
    reg = pretrained_model.layers[0].kernel_regularizer
    if pretrained:
        run(train_data, workdir, reg=reg, prefix="transfer_" + str(TRANSFER_LEARNING_RATE) + "_" + mappings[genotype] + "_" + str(num_samples) + "_", pretrained_model=pretrained_model, valid_data=valid_data)
    else:
        pretrained_model = None # we just want the same regulariser as in the pretrained model
        run(train_data, workdir, reg=reg, prefix="baseline_" + mappings[genotype] + "_" + str(num_samples) + "_", pretrained_model=None, valid_data=valid_data)

def load_data():
    data_f = os.environ["OUTPUT_DIR"] + "model_training/data_{}x/Lindel/Tijsterman_Analyser/train".format(MIN_READS_PER_TARGET)
    all_data = pkl.load(open(data_f, 'rb'))
    samples = get_common_samples(min_reads=MIN_READS_PER_TARGET)
    data = [all_data[s] for s in samples]
    return data

if __name__ == "__main__":
    mp.set_start_method("spawn")
    
    experiment = sys.argv[2]

    if experiment not in ["baseline", "pretrained"]:
        print("experiment is not valid: ", experiment)
        exit()    

    if experiment == "baseline":
        train_baseline_model()
    elif experiment == "pretrained":
        for num_samples in [2, 5, 10, 20, 50, 100, 200, 500]:
            TRANSFER_LEARNING_RATE = float(sys.argv[3])
            transfer_model(sys.argv[1], num_samples, pretrained=True)
    else:
        print("Invalid experiment.")
