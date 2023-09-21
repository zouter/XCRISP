from __future__ import division
import sys, os, datetime, subprocess, math, pickle, imp, fnmatch
sys.path.append('/cluster/mshen/')
import numpy as np
from collections import defaultdict
import pandas as pd
import matplotlib
import predict
matplotlib.use('Pdf')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from pandas.errors import EmptyDataError
from prepare import get_Tijsterman_Analyser_datafile, get_valid_indels, is_insert_at_cutsite
from tqdm import tqdm
sys.path.append("../")
from data_loader import get_details_from_fasta

OUTPUT_DIR = os.environ['OUTPUT_DIR']
out_dir = OUTPUT_DIR + "/model_training/data_100x/inDelphi/ins_ratio_"

##
# Going wide: experiments to analyze
##
exps = ['train', 'test']

##
# Run statistics
##
def calc_statistics(orig_df, exp, alldf_dict):
    df = get_valid_indels(orig_df)
    # Calculate statistics on df, saving to alldf_dict
    # Deletion positions

    # Denominator is ins
    if sum(df["countEvents"]) < 100:
        return

    # editing_rate = sum(_lib.crispr_subset(df)['Count']) / sum(_lib.notnoise_subset(df)['Count'])
    # alldf_dict['Editing Rate'].append(editing_rate)

    ins_criteria = (df['Type'] == 'INSERTION') & (df['insSize'] == 1) & df.apply(is_insert_at_cutsite, axis=1)
    ins_count = sum(df[ins_criteria]['countEvents'])

    del_criteria = (df['Type'] == 'DELETION') 
    del_count = sum(df[del_criteria]['countEvents'])
    if del_count == 0:
        return
    alldf_dict['Ins1bp/Del Ratio'].append(ins_count / (del_count + ins_count))

    mhdel_crit = (df['Type'] == 'DELETION') & (df['homologyLength'] > 0)
    mhdel_count = sum(df[mhdel_crit]['countEvents'])
    try:
        alldf_dict['Ins1bp/MHDel Ratio'].append(ins_count / (mhdel_count + ins_count))
    except ZeroDivisionError:
        alldf_dict['Ins1bp/MHDel Ratio'].append(0)

    ins_ratio = ins_count / sum(df["countEvents"])
    alldf_dict['Ins1bp Ratio'].append(ins_ratio)

    seq = exp["TargetSequence"]
    cutsite = exp["PAM Index"] - 3
    fivebase = seq[cutsite - 1]
    alldf_dict['Fivebase'].append(fivebase)

    predict.init_model()
    del_score, dlpred = predict.predict_dels(seq, cutsite)
    alldf_dict['Del Score'].append(del_score)

    from scipy.stats import entropy
    norm_entropy = entropy(dlpred) / np.log(len(dlpred))
    alldf_dict['Entropy'].append(norm_entropy)

    local_seq = seq[cutsite - 4 : cutsite + 4]
    gc = (local_seq.count('C') + local_seq.count('G')) / len(local_seq)
    alldf_dict['GC'].append(gc)

    if fivebase == 'A':
        fivebase_oh = np.array([1, 0, 0, 0])
    if fivebase == 'C':
        fivebase_oh = np.array([0, 1, 0, 0])
    if fivebase == 'G':
        fivebase_oh = np.array([0, 0, 1, 0])
    if fivebase == 'T':
        fivebase_oh = np.array([0, 0, 0, 1])
    alldf_dict['Fivebase_OH'].append(fivebase_oh)

    threebase = seq[cutsite]
    alldf_dict['Threebase'].append(threebase)
    if threebase == 'A':
        threebase_oh = np.array([1, 0, 0, 0])
    if threebase == 'C':
        threebase_oh = np.array([0, 1, 0, 0])
    if threebase == 'G':
        threebase_oh = np.array([0, 0, 1, 0])
    if threebase == 'T':
        threebase_oh = np.array([0, 0, 0, 1])
    alldf_dict['Threebase_OH'].append(threebase_oh)

    alldf_dict['_Experiment'].append(exp["ID"])

    return alldf_dict

def prepare_statistics(data_nm):
    alldf_dict = defaultdict(list)
    if data_nm in ["train", "test"]:
        guides = list(get_details_from_fasta("../../data/FORECasT/{}.fasta".format(data_nm)).values())
    else:
        guides = list(get_details_from_fasta("../../data/inDelphi/LibA.fasta").values())

    for g in tqdm(guides):    
    # for g in [{"ID": "Oligo_46064"}]:
        datafile = get_Tijsterman_Analyser_datafile(data_nm, g["ID"])
        if datafile is None: continue
        try:
            df = pd.read_csv(datafile, sep="\t")
        except EmptyDataError:
            continue

        print("loaded datafile", g)
        calc_statistics(df, g, alldf_dict)

    # Input: Dataset
    # Output: Uniformly processed dataset, requiring minimal processing for plotting but ideally enabling multiple plots
    # Calculate statistics associated with each experiment by name

    # # Return a dataframe where columns are positions and rows are experiment names, values are frequencies
    alldf = pd.DataFrame(alldf_dict)
    return alldf


##
# Load statistics from csv, or calculate 
##
def load_statistics(data_nm, redo=False):
  print(data_nm)
  stats_csv_fn = out_dir + '%s.csv' % (data_nm)
  if not os.path.isfile(stats_csv_fn) or redo:
    print('Running statistics from scratch...')
    stats_csv = prepare_statistics(data_nm)
    stats_csv.to_csv(stats_csv_fn)
  else:
    print('Getting statistics from file...')
    stats_csv = pd.read_csv(stats_csv_fn, index_col = 0)
  print('Done')
  return stats_csv

##
# Plotters
##
def plot():
  # Frequency of deletions by length and MH basis.

  return


##
# nohups
##
# def gen_nohups():
#   # Generate qsub shell scripts and commands for easy parallelization
#   print 'Generating nohup scripts...'
#   qsubs_dir = _config.QSUBS_DIR + NAME + '/'
#   util.ensure_dir_exists(qsubs_dir)
#   nh_commands = []

#   num_scripts = 0
#   for exp in exps:
#     script_id = NAME.split('_')[0]
#     command = 'nohup python -u %s.py %s redo > nh_%s_%s.out &' % (NAME, exp, script_id, exp)
#     nh_commands.append(command)

#   # Save commands
#   with open(qsubs_dir + '_commands.txt', 'w') as f:
#     f.write('\n'.join(nh_commands))

#   return


##
# Main
##
def main(data_nm = '', redo_flag = ''):
    OUTPUT_DIR = os.environ['OUTPUT_DIR']
    out_dir = OUTPUT_DIR + "/model_training/data_100x/inDelphi/"

    redo = False
    if redo_flag == 'redo':
        redo = True

    #   if data_nm == '':
    #     gen_nohups()
    #     return

    if data_nm == 'plot':
        plot()

    else:
        load_statistics(data_nm, redo=True)

    return


if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(data_nm = sys.argv[1])
    elif len(sys.argv) == 3:
        main(data_nm = sys.argv[1], redo_flag = sys.argv[2])
    else:
        main()
