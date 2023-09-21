import os, sys, random, logging, random

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd 
import numpy as np

from mpi4py import MPI
from sklearn.model_selection import train_test_split, KFold

from model import read_theta, train_model_parallel, test_model_parallel, write_theta

sys.path.append("../")
from data_loader import get_common_samples
from test_setup import MIN_NUMBER_OF_READS

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

NUM_FOLDS = 5
RANDOM_STATE = 42
REG_CONSTS = [0.001, 0.01, 0.1]
I1_REG_CONST = [0.001, 0.01]
MIN_READS_PER_TARGET = MIN_NUMBER_OF_READS
TRAINING_DATA_DIR = os.environ["OUTPUT_DIR"] + "model_training/data_{}x/train/Tijsterman_Analyser/".format(MIN_READS_PER_TARGET)
MODEL_OUTPUT_FOLDER = os.environ["OUTPUT_DIR"] + "model_training/model/FORECasT/"
os.makedirs(MODEL_OUTPUT_FOLDER, exist_ok=True)
MODEL_THETAS_FILE = MODEL_OUTPUT_FOLDER + "{}x_model_thetas".format(MIN_READS_PER_TARGET)
MODEL_RESULTS_FILE = MODEL_OUTPUT_FOLDER + "{}x_model_results".format(MIN_READS_PER_TARGET)

def init_logging():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

def run_batch(r1, r2, num_folds=NUM_FOLDS):
    cmd = "sbatch train.batch {} {} {}".format(r1, r2, num_folds)
    logging.info(cmd)
    os.system(cmd)

def write_train_and_validation_results(train_results, validation_results, reg_const, i1_reg_const):
    results_f = '%s_l2_%s_l1_%s.txt' % (MODEL_RESULTS_FILE, reg_const, i1_reg_const)
    with open(results_f, "w") as f:
        f.write("l2: {},  l1: {}\n".format(reg_const, i1_reg_const))
        f.write("Fold\tTest\tValidation\n")
        for i in range(NUM_FOLDS):
            f.write("{}\t{}\t{}\n".format(i+1, train_results[i][0], validation_results[i][0]))
        mean_train_Q = np.mean([x[0] for x in train_results])
        mean_val_Q = np.mean([x[0] for x in validation_results])
        f.write("Average:\t{}\t{}\n".format(mean_train_Q, mean_val_Q))

def write_train_results(train_results, reg_const, i1_reg_const):
    results_f = '%s.txt' % (MODEL_RESULTS_FILE)
    with open(results_f, "w") as f:
        f.write("l2: {},  l1: {}\n".format(reg_const, i1_reg_const))
        f.write("Fold\tTest\n")
        f.write("{}\t{}\n".format(0, train_results[0][0]))
        mean_train_Q = np.mean([x[0] for x in train_results])
        f.write("Average:\t{}\n".format(mean_train_Q))

def load_feature_labels(oligo_id):
    data = pd.read_pickle(TRAINING_DATA_DIR + oligo_id)
    return [x for x in data.columns if x not in ['Oligo ID','Indel','Frac Sample Reads','Left','Right','Inserted Seq', "Indel_x", "Indel_y", "Counts"]]

def k_fold_cross_validation(samples, feature_columns, kf, reg_const, i1_reg_const, force_reset=False):
    train_results = []
    validation_results = []
    for cv_fold, (train_idx, validation_idx) in enumerate(kf.split(samples)):
        x = np.array(samples)[train_idx]
        val = np.array(samples)[validation_idx]
        theta0 = None
        tmp_file = '%s_l2_%s_l1_%s_cf%d.txt.tmp' % (MODEL_THETAS_FILE, reg_const, i1_reg_const, cv_fold)

        if os.path.isfile(tmp_file) and not force_reset:
                logging.info('Loading from previous tmp file')
                theta0, x, feature_columns = read_theta(tmp_file)

        logging.info("Training...")
        theta = train_model_parallel(x, feature_columns, theta0, tmp_file, cv_idx=cv_fold, reg_const=reg_const, i1_reg_const=i1_reg_const)   
        write_theta(tmp_file[:-4], feature_columns, theta, x)
        
        logging.info('Testing...')
        tr = test_model_parallel(theta, x, feature_columns)
        vr = test_model_parallel(theta, val, feature_columns)

        if mpi_rank == 0:
            train_results.append(tr)
            validation_results.append(vr)
            
    if mpi_rank == 0:
        write_train_and_validation_results(train_results, validation_results, reg_const, i1_reg_const)

def train_model(samples, feature_columns, reg_const, i1_reg_const, force_reset=False):
    theta0 = None
    tmp_file = '%s_l2%s_l1%s.txt.tmp' % (MODEL_THETAS_FILE, reg_const, i1_reg_const)
    
    if os.path.isfile(tmp_file) and not force_reset:
        logging.info('Loading from previous tmp file')
        theta0, samples, feature_columns = read_theta(tmp_file)

    logging.info("Training...")
    theta = train_model_parallel(samples, feature_columns, theta0, tmp_file, reg_const=reg_const, i1_reg_const=i1_reg_const)    
    write_theta(tmp_file[:-4], feature_columns, theta, samples)
    
    logging.info('Testing...')
    train_results = test_model_parallel(theta, samples, feature_columns)

    if mpi_rank == 0:
        write_train_results(train_results, reg_const, i1_reg_const)

def run(samples, reg_const=0.01, i1_reg_const=0.01, force_reset=False, num_folds=True):
    if mpi_rank == 0: 
        feature_columns = load_feature_labels(samples[0])
        data = (samples, feature_columns)
    else:
        data = None
    samples, feature_columns = comm.bcast(data, root=0)
    logging.info("Rank {}, Train Samples={}".format(mpi_rank, len(samples)))

    if num_folds > 0:
        kf = KFold(n_splits=num_folds, random_state=RANDOM_STATE, shuffle=True)
        k_fold_cross_validation(samples, feature_columns, kf, reg_const, i1_reg_const, force_reset=force_reset)
    else:
        train_model(samples, feature_columns, reg_const, i1_reg_const, force_reset=force_reset)

if __name__ == "__main__":
    init_logging()
    if len(sys.argv) == 1:
        for r1 in REG_CONSTS:
            for r2 in I1_REG_CONST:
                run_batch(r1, r2, NUM_FOLDS)
        sys.exit()

    # train = os.listdir(TRAINING_DATA_DIR)
    train = get_common_samples(min_reads=MIN_READS_PER_TARGET)
    reg_const = eval(sys.argv[1])
    i1_reg_const = eval(sys.argv[2])
    folds = eval(sys.argv[3])
    force_reset = eval(sys.argv[4]) if len(sys.argv) >= 5 else False
    run(train, reg_const=reg_const, i1_reg_const=i1_reg_const, force_reset=force_reset, num_folds=folds)




