import pickle as pkl
import os,sys,csv,re,random
import numpy as np
import multiprocessing as mp

from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input, Flatten
from keras.models import Sequential, load_model
from keras.regularizers import l2, l1
from keras import backend as K

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tommassi_regularizer import MMKT
from tradaboost import TrAdaBoostClassifier

sys.path.append("../")
from data_loader import get_guides_from_fasta, get_common_samples
from test_setup import MIN_NUMBER_OF_READS

NUM_FOLDS = 5
RANDOM_STATE = 42
INPUT_SIZE = 3033
OUTPUT_SIZE = 536
MIN_READS_PER_TARGET = MIN_NUMBER_OF_READS

# Define useful functions
def mse(x, y):
    return ((x-y)**2).mean()

def corr(x, y):
    return np.corrcoef(x, y)[0, 1]

def train_model(x_train, y_train, reg, x_valid=None, y_valid=None, pretrained_model=None):
    np.random.seed(0)
    if pretrained_model == None:
        model = Sequential()
        model.add(Dense(OUTPUT_SIZE,  activation='softmax', input_shape=(INPUT_SIZE,), kernel_regularizer=reg))
    else:
        model = pretrained_model
        model.layers[0].kernel_regularizer = reg
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse'])
    if x_valid is not None and y_valid is not None:
        model.fit(x_train, y_train, epochs=200, validation_data=(x_valid, y_valid), 
                callbacks=[EarlyStopping(patience=1)], verbose=1)
        y_hat = model.predict(x_valid)
        return model, mse(y_hat, y_valid)
    else:
        model.fit(x_train, y_train, epochs=200, verbose=1)
        return model, None

def benchmark_models(X, Y, lambdas, split, pretrained_model):
    errors_l1 = []
    errors_l2 = []

    x_train = X[split[0], :]
    y_train = Y[split[0], :]
    x_valid = X[split[1], :]
    y_valid = Y[split[1], :]

    if pretrained_model != None:
        pretrained_model = load_model(pretrained_model)

    for l in lambdas:
        if pretrained_model == None:
            _, err = train_model(x_train, y_train, l1(l), x_valid=x_valid, y_valid=y_valid, pretrained_model=None)
            errors_l1.append(err)
            _, err = train_model(x_train, y_train, l2(l), x_valid=x_valid, y_valid=y_valid, pretrained_model=None)
            errors_l2.append(err)
        else:
            pretrained_weights = pretrained_model.layers[0].get_weights()[0]
            _, err = train_model(x_train, y_train, MMKT(pretrained_weights, l2=l), x_valid=x_valid, y_valid=y_valid, pretrained_model=pretrained_model)
            errors_l2.append(err)
    
    return errors_l1, errors_l2

def average_fold_errors(errors):
    means = np.mean(np.array(errors), axis=0)
    min_mean = np.min(means)
    min_index = np.argmin(means)
    return min_mean, min_index

def run(data, workdir, prefix="", pretrained_model=None):
    os.makedirs(workdir, exist_ok=True)

    Y = np.array([y[1][:OUTPUT_SIZE]/sum(y[1][:OUTPUT_SIZE]) for y in data])
    Y = np.nan_to_num(Y)
    X = np.array([x[4] for x in data])
    lambdas = 10 ** np.arange(-10, -1, 0.1) # for activation function test
    kf = KFold(n_splits=NUM_FOLDS, random_state=RANDOM_STATE, shuffle=True)
    folds = list(kf.split(X, Y))

    errors_l1 = []
    errors_l2 = []
    def log_results(results):
        e1, e2 = results
        errors_l1.append(e1)
        errors_l2.append(e2)
    pool = mp.Pool(5)
    for fold in folds:
        pool.apply_async(benchmark_models, args=(X, Y, lambdas, fold, pretrained_model), callback = log_results)
    pool.close()
    pool.join()
    np.save(workdir+'{}mse_l1_del.npy'.format(prefix),errors_l1)
    np.save(workdir+'{}mse_l2_del.npy'.format(prefix),errors_l2)

    # find best params and train final model
    min_mean_l1, min_idx_l1 = average_fold_errors(errors_l1)
    min_mean_l2, min_idx_l2 = average_fold_errors(errors_l2)

    if pretrained_model == None:
        best_lambda, reg = None, None
        if min_mean_l1 < min_mean_l2:
            best_lambda = lambdas[min_idx_l1]
            reg = l1(best_lambda)
        else:
            best_lambda = lambdas[min_idx_l2]
            reg = l2(best_lambda)
    else:
        pretrained_model = load_model(pretrained_model)
        pretrained_weights = pretrained_model.layers[0].get_weights()[0]
        best_lambda = lambdas[min_idx_l2]
        reg = MMKT(pretrained_weights ,best_lambda)
        print("Refining full model using MMKT", best_lambda, reg)
    
    m, err = train_model(X, Y, reg, pretrained_model=pretrained_model)
    m.save(workdir + '{}x_{}deletion.h5'.format(MIN_READS_PER_TARGET, prefix))

    print("Done.")

def run_tradaboost(source_data, target_data, workdir, pretrained_model=None, prefix=""):
    os.makedirs(workdir, exist_ok=True)

    y_s = np.array([y[1][:OUTPUT_SIZE]/sum(y[1][:OUTPUT_SIZE]) for y in source_data])
    exclude_s = np.isnan(y_s).any(axis=1)
    y_s = y_s if len(exclude_s) == 0 else y_s[~exclude_s, :]
    y_s = normalize(y_s, axis=1, norm='l1')
    X_s = np.array([x[4] for x in source_data])
    X_s = X_s if len(exclude_s) == 0 else X_s[~exclude_s, :]

    y_t = np.array([y[1][:OUTPUT_SIZE]/sum(y[1][:OUTPUT_SIZE]) for y in target_data])
    exclude_t = np.isnan(y_t).any(axis=1)
    y_t = y_t if len(exclude_t) == 0 else y_t[~exclude_t, :]
    y_t = normalize(y_t, axis=1, norm='l1')
    X_t = np.array([x[4] for x in target_data])
    X_t = X_t if len(exclude_t) == 0 else X_t[~exclude_t, :]

    clf = TrAdaBoostClassifier(input_shape=INPUT_SIZE, output_size=OUTPUT_SIZE)
    clf.fit(X_s, y_s, X_t, y_t)

    clf.save(workdir, '{}deletion'.format(prefix))
    print("Done.")

def train_simple_transfer_model():
    workdir = os.environ["OUTPUT_DIR"] + "model_training/model/Lindel/transfer/"
    os.makedirs(workdir, exist_ok=True)
    data_f = os.environ["OUTPUT_DIR"] + "model_training/data/Lindel/Tijsterman_Analyser/0226-PRLmESC-Lib1-Cas9"
    data = list(pkl.load(open(data_f, 'rb')).values())
    guides = get_guides_from_fasta("../../data/inDelphi/train.fasta")

    data = [d for d in data if d[5]["ID"] in guides]
    sizes = [5, 10, 15, 20, 30, 50, 100, 200, 500, 1000, 1500]
    datasets = []
    for s in sizes:
        random.seed(42)
        datasets.append(random.sample(data, s))

    for i, data in enumerate(datasets):
        print("Transfer training with size {}".format(len(data)))
        run(data, workdir, prefix="{}_MMKT_".format(len(data)), pretrained_model='./models/deletion.h5')

def train_tradaboost_model():
    workdir = os.environ["OUTPUT_DIR"] + "model_training/model/Lindel/tradaboost/"
    os.makedirs(workdir, exist_ok=True)
    target_data_f = os.environ["OUTPUT_DIR"] + "model_training/data/Lindel/Tijsterman_Analyser/0226-PRLmESC-Lib1-Cas9"
    target_data = list(pkl.load(open(target_data_f, 'rb')).values())
    guides = get_guides_from_fasta("../../data/inDelphi/train.fasta")
    target_data = [d for d in target_data if d[5]["ID"] in guides][:100]

    source_data_f = os.environ["OUTPUT_DIR"] + "model_training/data/Lindel/Tijsterman_Analyser/train"
    source_data = list(pkl.load(open(source_data_f, 'rb')).values())

    run_tradaboost(source_data, target_data, workdir, prefix="100_")

def train_NHEJ_model():
    workdir = os.environ["OUTPUT_DIR"] + "model_training/model/Lindel/"
    data_f = os.environ["OUTPUT_DIR"] + "model_training/data/Lindel/Tijsterman_Analyser/0226-PRLmESC-Lib1-Cas9"
    data = list(pkl.load(open(data_f, 'rb')).values())
    guides = get_guides_from_fasta("../../data/inDelphi/train.fasta")
    data = [d for d in data if d[5]["ID"] in guides]
    run(data, workdir, prefix="NHEJ_")

def train_baseline_model():
    workdir = os.environ["OUTPUT_DIR"] + "model_training/model/Lindel/"
    data = load_data()
    run(data, workdir)

def load_data():
    data_f = os.environ["OUTPUT_DIR"] + "model_training/data_{}x/Lindel/Tijsterman_Analyser/train".format(MIN_READS_PER_TARGET)
    all_data = pkl.load(open(data_f, 'rb'))
    samples = get_common_samples(min_reads=MIN_READS_PER_TARGET)
    data = [all_data[s] for s in samples]
    return data

if __name__ == "__main__":
    mp.set_start_method("spawn")
    train_baseline_model()
