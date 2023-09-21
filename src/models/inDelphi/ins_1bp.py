from __future__ import division
import sys, os, datetime, subprocess, math, pickle, imp, fnmatch
sys.path.append('/cluster/mshen/')
import numpy as np
from collections import defaultdict
import pandas as pd
import matplotlib
matplotlib.use('Pdf')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from pandas.errors import EmptyDataError
from prepare import get_Tijsterman_Analyser_datafile, get_valid_indels, is_insert_at_cutsite
from tqdm import tqdm
sys.path.append("../")
from data_loader import get_details_from_fasta

# Default params
OUTPUT_DIR = os.environ['OUTPUT_DIR']
out_dir = OUTPUT_DIR + "/model_training/data_100x/inDelphi/1bpins_"

##
# Going wide: experiments to analyze
##
exps = ['train', 'test']

##
# Run statistics
##
def calc_statistics(orig_df, exp, alldf_dict):
    # Calculate statistics on df, saving to alldf_dict
    # Deletion positions

    # Denominator is crispr activity
    df = get_valid_indels(orig_df)
    if sum(df['countEvents']) <= 100:
        return
    df['Frequency'] = df["countEvents"]/sum(df["countEvents"])

    criteria = (df['Type'] == 'INSERTION') & (df['insSize'] == 1)

    freq = sum(df[criteria]['Frequency'])
    alldf_dict['Frequency'].append(freq)

    s = df[criteria]

    try:
        a_frac = sum(s[s['insertion'] == 'A']['Frequency']) / freq
    except:
        a_frac = 0
    alldf_dict['A frac'].append(a_frac)

    try:
        c_frac = sum(s[s['insertion'] == 'C']['Frequency']) / freq
    except:
        c_frac = 0
    alldf_dict['C frac'].append(c_frac)

    try:
        g_frac = sum(s[s['insertion'] == 'G']['Frequency']) / freq
    except:
        g_frac = 0
    alldf_dict['G frac'].append(g_frac)

    try:
        t_frac = sum(s[s['insertion'] == 'T']['Frequency']) / freq
    except:
        t_frac = 0
    alldf_dict['T frac'].append(t_frac)

    seq = exp["TargetSequence"]
    cutsite = exp["PAM Index"] - 3
    fivebase = seq[cutsite-1]
    alldf_dict['Base'].append(fivebase)

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
