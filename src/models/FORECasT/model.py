import io, os, sys, csv, time, logging

import pandas as pd
import numpy as np
import random

from mpi4py import MPI

from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize

from src.models.FORECasT.features import read_features_data

from src.config.test_setup import MIN_NUMBER_OF_READS

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

REG_CONST = 0.01
I1_REG_CONST = 0.01
MIN_READS_PER_TARGET = MIN_NUMBER_OF_READS
TRAINING_DATA_DIR = os.environ["OUTPUT_DIR"] + "model_training/data_{}x/train/Tijsterman_Analyser/".format(MIN_READS_PER_TARGET)

def get_profile_counts(profile):
    total = sum([profile[x] for x in profile])
    if total == 0:
        return []
    indel_total = total
    if '-' in profile:
        indel_total -= profile['-']
        null_perc = profile['-']*100.0/indel_total if indel_total != 0 else 100.0
        null_profile = (profile['-'],'-',profile['-']*100.0/total, null_perc)
    counts = [(profile[x],x, profile[x]*100.0/total, profile[x]*100.0/indel_total) for x in profile if x != '-']
    counts.sort(reverse = True)
    if '-' in profile:
        counts = [null_profile] + counts
    return counts

def load(oligo_id):
    return pd.read_pickle(TRAINING_DATA_DIR + oligo_id)

def write_theta(out_file, feature_columns, theta, train_set):
    fout = io.open(out_file, 'w')
    fout.write(u'%s\n' % ','.join([x for x in train_set]))
    lines = [u'%s\t%s\n' % (x,y) for (x,y) in zip(feature_columns, theta)]
    fout.write(''.join(lines))
    fout.close()

def read_theta(theta_file):
    f = io.open(theta_file)
    train_set = f.readline()[:-1].split(',')
    feature_columns, theta = [], []
    for toks in csv.reader(f, delimiter='\t'):
        feature_columns.append(toks[0])
        theta.append(eval(toks[1]))
    return theta, train_set, feature_columns

def print_and_flush(msg, master_only=True):
    if not master_only or mpi_rank == 0:
        logging.info(msg)
        sys.stdout.flush()

def get_cut_site(features_file):
    f = io.open(features_file); f.readline()
    cut_site = eval(f.readline().split('\t')[1])
    return cut_site

def calc_theta_x(row, theta, feature_columns):
    return sum(theta*row[feature_columns[0]:feature_columns[-1]])

def compute_regularisers(theta, feature_columns, reg_const, i1_reg_const):
    Q_reg = sum([i1_reg_const*val**2.0 if 'I' in name else reg_const*val**2.0 for (val, name) in zip(theta, feature_columns)])
    grad_reg = theta*np.array([i1_reg_const if 'I' in name else reg_const for name in feature_columns])
    return Q_reg, grad_reg

def compute_KL_obj_and_gradients(theta, guideset, d, feature_columns, reg_const, i1_reg_const):
    N = len(feature_columns)
    Q, jac, minQ, maxQ = 0.0, np.zeros(N), 0.0, 1000.0
    Qs = []
    for i, oligo_id in enumerate(guideset):
        _data = d[oligo_id]
        st = time.time()
        Y = _data['Frac Sample Reads']
        _data['ThetaX'] = _data.apply(calc_theta_x, axis=1, args=(theta,feature_columns))
        sum_exp = np.exp(_data['ThetaX']).sum()
        Q_reg, grad_reg =  compute_regularisers(theta, feature_columns, reg_const, i1_reg_const)
        tmpQ = (np.log(sum_exp) + sum(Y*(np.log(Y) - _data['ThetaX'])) + Q_reg)
        Q += tmpQ
        Qs.append(tmpQ)
        jac += np.matmul(np.exp(_data['ThetaX']),_data[feature_columns].astype(int))/sum_exp - np.matmul(Y,_data[feature_columns].astype(int)) + grad_reg
        logging.debug("%s - computed %d of %d in %s seconds" % (mpi_rank, i, len(guideset), time.time() - st))
    return Q, jac, Qs 

def assess_fit(theta, guideset, feature_columns, data, cv_idx=0, reg_const=REG_CONST, i1_reg_const=I1_REG_CONST, test_only=False, tmp_file = None):
    #Send out thetas
    theta, done = comm.bcast((theta, False), root=0)
    while not done:
        start_time = time.time()
        #Compute objective and gradients
        Q, jac, Qs = compute_KL_obj_and_gradients(theta, guideset, data, feature_columns, reg_const, i1_reg_const)

        #Combine all
        full_guideset = comm.gather([x for x in guideset], root=0)
        flatten = lambda l: [item for sublist in l for item in sublist]
        objs_and_grads = comm.gather((Q, jac, Qs), root=0)
        if mpi_rank == 0:
            Q, jac, Qs = sum([x[0] for x in objs_and_grads]), sum([x[1] for x in objs_and_grads]), []
            for x in objs_and_grads: Qs.extend(x[2]) 
            Q, jac, Qs = Q/len(Qs), jac/len(Qs), Qs
            print_and_flush(' '.join(['Q=%.5f' % Q, 'Min=%.3f' % min(Qs), 'Max=%.3f' % max(Qs), 'Num=%d' % len(Qs), 'Lambda=%e' % reg_const, 'I1_Lambda=%e' % i1_reg_const]))
            logging.info("--- %s seconds ---" % (time.time() - start_time))
            if tmp_file is not None:
                write_theta(tmp_file, feature_columns, theta, flatten(full_guideset))

        Q, jac, Qs = comm.bcast((Q, jac, Qs), root=0)
        if mpi_rank == 0 or test_only: done = True
        else:
            done = False
            theta, done = comm.bcast((theta, done), root=0)
    return Q, jac, Qs

def train_model_parallel(guideset, feature_columns, theta0, tmp_file, cv_idx=0, reg_const=REG_CONST, i1_reg_const=I1_REG_CONST):
    guidesubsets = [guideset[i:len(guideset):mpi_size] for i in range(mpi_size)]
    # randomly initialise theta values
    if theta0 is None: theta0 = np.array([np.random.normal(loc=0.0, scale=1.0) for x in feature_columns])
    
    data = {}
    for i, oligo_id in enumerate(guidesubsets[mpi_rank]):
        st = time.time()
        data[oligo_id] = load(oligo_id)
        logging.info("%s - Loading %s, %d of %d in %s" % (mpi_rank, oligo_id, i, len(guidesubsets[mpi_rank]), time.time() - st))

    args=(guidesubsets[mpi_rank], feature_columns, data, cv_idx, reg_const, i1_reg_const, False, tmp_file)
    if mpi_rank == 0:
        result = minimize(assess_fit, theta0, args=args, method='L-BFGS-B', jac=True, tol=1e-4)
        theta = result.x
        print_and_flush("Optimization Result: " + str(result.success))
        done = True
        theta, done = comm.bcast((theta, done), root=0)
    else:
        assess_fit(theta0, guidesubsets[mpi_rank], feature_columns, data, cv_idx, reg_const, i1_reg_const, False, tmp_file)
        theta, done = None, True
    theta, done = comm.bcast((theta, done), root=0)
    return theta

def test_model_parallel(theta, guideset, feature_columns, sample_names=["mESC"]):
    guidesubsets = [guideset[i:len(guideset):mpi_size] for i in range(mpi_size)]
    data = {}
    for i, oligo_id in enumerate(guidesubsets[mpi_rank]):
        st = time.time()
        data[oligo_id] = load(oligo_id)
        logging.info("%s - Loading %s, %d of %d in %s" % (mpi_rank, oligo_id, i, len(guidesubsets[mpi_rank]), time.time() - st))
    Q, jac, Qs = assess_fit(theta, guidesubsets[mpi_rank], feature_columns, data, reg_const=0.0, i1_reg_const=0.0, test_only=True )
    if mpi_rank == 0:
        return (Q, Qs)
    else:
        return None
        
def compute_predicted_profile(data, theta, feature_columns):
    data['expThetaX'] = np.exp(data.apply(calc_theta_x, axis=1, args=(theta,feature_columns)))
    sum_exp = data['expThetaX'].sum()
    profile = {x: expthetax*1000/sum_exp for (x,expthetax) in zip(data['Indel'],data['expThetaX'])}
    counts = get_profile_counts(profile)
    return profile, counts
