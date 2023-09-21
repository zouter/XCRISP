from __future__ import division
import sys, os, datetime, subprocess, math, pickle, imp, fnmatch
import random
import numpy as np
from collections import defaultdict
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


out_dir = os.environ["OUTPUT_DIR"] + "/model_training/model/inDelphi/"

##
# Functions
##
def convert_oh_string_to_nparray(input):
    input = input.replace('[', '').replace(']', '')
    nums = input.split(' ')
    return np.array([int(s) for s in nums])

def featurize(rate_stats, Y_nm):
    fivebases = np.array([convert_oh_string_to_nparray(s) for s in rate_stats['Fivebase_OH']])
    threebases = np.array([convert_oh_string_to_nparray(s) for s in rate_stats['Threebase_OH']])

    ent = np.array(rate_stats['Entropy']).reshape(len(rate_stats['Entropy']), 1)
    del_scores = np.array(rate_stats['Del Score']).reshape(len(rate_stats['Del Score']), 1)
    print(ent.shape, fivebases.shape, del_scores.shape)

    Y = np.array(rate_stats[Y_nm])
    print(Y_nm)
    
    Normalizer = [(np.mean(fivebases.T[2]),
                      np.std(fivebases.T[2])),
                  (np.mean(fivebases.T[3]),
                      np.std(fivebases.T[3])),
                  (np.mean(threebases.T[0]),
                      np.std(threebases.T[0])),
                  (np.mean(threebases.T[2]),
                      np.std(threebases.T[2])),
                  (np.mean(ent),
                      np.std(ent)),
                  (np.mean(del_scores),
                      np.std(del_scores)),
                 ]

    fiveG = (fivebases.T[2] - np.mean(fivebases.T[2])) / np.std(fivebases.T[2])
    fiveT = (fivebases.T[3] - np.mean(fivebases.T[3])) / np.std(fivebases.T[3])
    threeA = (threebases.T[0] - np.mean(threebases.T[0])) / np.std(threebases.T[0])
    threeG = (threebases.T[2] - np.mean(threebases.T[2])) / np.std(threebases.T[2])
    gtag = np.array([fiveG, fiveT, threeA, threeG]).T

    ent = (ent - np.mean(ent)) / np.std(ent)
    del_scores = (del_scores - np.mean(del_scores)) / np.std(del_scores)

    X = np.concatenate(( gtag, ent, del_scores), axis = 1)
    X = np.concatenate(( gtag, ent, del_scores), axis = 1)
    feature_names = ['5G', '5T', '3A', '3G', 'Entropy', 'DelScore']
    print('Num. samples: %s, num. features: %s' % X.shape)

    return X, Y, Normalizer

def generate_models(X, Y, bp_stats, Normalizer):
  # Train rate model
  model = KNeighborsRegressor()
  model.fit(X, Y)
  with open(out_dir + 'rate_model_v2.pkl', 'wb') as f:
    pickle.dump(model, f)

  # Obtain bp stats
  bp_model = dict()
  ins_bases = ['A frac', 'C frac', 'G frac', 'T frac']
  t_melt = pd.melt(bp_stats, 
                   id_vars = ['Base'], 
                   value_vars = ins_bases, 
                   var_name = 'Ins Base', 
                   value_name = 'Fraction')
  for base in list('ACGT'):
    bp_model[base] = dict()
    mean_vals = []
    for ins_base in ins_bases:
      crit = (t_melt['Base'] == base) & (t_melt['Ins Base'] == ins_base)
      mean_vals.append(float(np.mean(t_melt[crit])))
    for bp, freq in zip(list('ACGT'), mean_vals):
      bp_model[base][bp] = freq / sum(mean_vals)

  with open(out_dir + 'bp_model_v2.pkl', 'wb') as f:
    pickle.dump(bp_model, f)

  with open(out_dir + 'Normalizer_v2.pkl', 'wb') as f:
    pickle.dump(Normalizer, f)

  return

##
# Main
##
def main(data_nm = ''):
  OUTPUT_DIR = os.environ['OUTPUT_DIR']
  out_dir = OUTPUT_DIR + "/model_training/model/InDelphi/"
  import ins_ratio
  import ins_1bp

  exps = ['train']

  all_rate_stats = pd.DataFrame()
  all_bp_stats = pd.DataFrame()  
  for exp in exps:
    rate_stats = ins_ratio.load_statistics(exp)
    rate_stats = rate_stats[rate_stats['Entropy'] > 0.01]
    bp_stats = ins_1bp.load_statistics(exp)
    exps = rate_stats['_Experiment']

    all_rate_stats = all_rate_stats.append(rate_stats, ignore_index = True)
    all_bp_stats = all_bp_stats.append(bp_stats, ignore_index = True)

    print(exp, len(all_rate_stats))

  X, Y, Normalizer = featurize(all_rate_stats, 'Ins1bp/Del Ratio')
  generate_models(X, Y, all_bp_stats, Normalizer)

  return


if __name__ == '__main__':
  if len(sys.argv) == 2:
    main(data_nm = sys.argv[1])
  else:
    main()
