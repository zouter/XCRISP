import sys, os
import pandas as pd
import numpy as np
import pickle as pkl
from tqdm import tqdm
from pandas.errors import EmptyDataError
from src.data.data_loader import get_details_from_fasta
from pandas.api.types import is_number

OUTPUT_DIR = os.environ['OUTPUT_DIR']
out_dir = OUTPUT_DIR + "/model_training/data_100x/inDelphi/"

class NoDeletionsError(Exception):
    pass

def init_featurized_data():
  good_exps, mh_lengths, gc_fracs, del_lens, freqs, dl_freqs, gt_poss = [], [], [], [], [], [], []
  all_data = [good_exps, mh_lengths, gc_fracs, del_lens, freqs, dl_freqs, gt_poss]
  return all_data

def pickle_featurized_data(featurized_data, nm):
  print("Pickling...")
  os.makedirs(out_dir, exist_ok=True)
  with open(out_dir + '%s.pkl' % (nm), 'wb') as f:
    pkl.dump(featurized_data, f)
  print("Done")
  return

def get_gc_frac(seq):
  return (seq.count('C') + seq.count('G')) / len(seq)

def normalize_frequency(data):
    countEvents = data.loc[:,"countEvents"]
    return countEvents/sum(countEvents)

def get_Tijsterman_Analyser_datafile(dataset, sample_name):
    if dataset in ["train", "test"]:
        filename = OUTPUT_DIR + "processed_data/Tijsterman_Analyser/FORECasT/{}".format(sample_name)
    else:
    
        filename = OUTPUT_DIR + "processed_data/Tijsterman_Analyser/{}/{}_indels".format(dataset, sample_name)
    print(filename)
    if os.path.exists(filename):
        return filename
    else:
        return None

# example to test {'ID': 'Oligo_10096', 'PAM Index': 42, 'Strand': 'FORWARD', 'TargetSequence': 'GATCTTTGGGGACTCTAAAT...ATTTTATGCC'}
def is_insert_at_cutsite(row):
    if row["delRelativeStart"] == 0:
        return True

    ins_seq = row["insertion"]
    leftFlank = row["leftFlank"]
    position = row["delRelativeStart"]
    for i in leftFlank[::-1]:
        if i == ins_seq:
            position -= 1
            if position == 0: return True
        else:
            break

    return False

def get_valid_indels(orig_df):
    DELLEN_LIMIT = 30
    del_df = orig_df[orig_df["Type"] == "DELETION"] # must be deletion
    del_df = del_df[del_df['delSize'] <= DELLEN_LIMIT] # less than del limit

    del_df = del_df.loc[del_df["delRelativeStart"].apply(is_number), :] # some rows seem to have strings here? bug maybe?

    del_df = del_df[del_df["delRelativeStart"] < 0] # left side starting before the cutsite
    del_df = del_df[del_df["delRelativeStart"] + del_df['delSize'] >= 0] # right side touching or going beyond 

    ins_df = orig_df[orig_df["Type"] == "INSERTION"]
    ins_df = ins_df[orig_df['insSize'] == 1]
    ins_df = ins_df[ins_df.apply(is_insert_at_cutsite, axis=1)]
    return pd.concat([del_df, ins_df])

def featurize(orig_df):    
    try:
        df = get_valid_indels(orig_df)
    except:
        return None

    if sum(df['countEvents']) < 100:
        return None

    criteria = (orig_df["Type"] == "DELETION") & (orig_df["delSize"] <= 28)
    s = orig_df[criteria].copy()
    s["Frequency"] = normalize_frequency(s)
    dl_freqs = []
    for del_len in range(1, 28+1):
        dl_freq = sum(s[s['delSize'] == del_len]['Frequency'])
        dl_freqs.append(dl_freq)

    df['Frequency'] = normalize_frequency(df) 

    criteria = (df["homologyLength"] > 0)
    mh_lens = df[criteria]["homologyLength"].astype('int').to_list()
    gc_fracs = [get_gc_frac(seq) for seq in df[criteria]["homology"].to_list()]
    del_lens = df[criteria]["delSize"].astype('int').to_list()
    freqs = df[criteria]["Frequency"].to_list()
    gt_poss = (df[criteria]["delRelativeStart"] + df[criteria]["delSize"]).to_list()

    if sum(criteria) < 2: return None
    if len(freqs) == 0: return None
    if len(dl_freqs) == 0: return None

    return mh_lens, gc_fracs, del_lens, freqs, dl_freqs, gt_poss


def prepare_dataset(data_nm):
    good_exps, mh_lengths, gc_fracs, del_lens, freqs, dl_freqs, gt_poss = init_featurized_data()
    if data_nm in ["train", "test"]:
        guides = list(get_details_from_fasta("../../data/FORECasT/{}.fasta".format(data_nm)).values())
    else:
        guides = list(get_details_from_fasta("../../data/inDelphi/LibA.fasta").values())

    for g in tqdm(guides):    
    # for g in [{"ID": "Oligo_46064"}]:
        datafile = get_Tijsterman_Analyser_datafile(data_nm, g["ID"])
        print(g["ID"])
        if datafile is None: continue
        try:
            data = pd.read_csv(datafile, sep="\t")
        except EmptyDataError:
            continue
        ans = featurize(data)
        if ans is None:
            continue
        mh_len, gc_frac, del_len, freq, dl_freq, gt_pos = ans

        if len(mh_len) == 0:
            raise NoDeletionsError()

        good_exps.append(g["ID"])
        mh_lengths.append(mh_len)
        gc_fracs.append(gc_frac)
        del_lens.append(del_len)
        freqs.append(freq)
        dl_freqs.append(dl_freq)
        gt_poss.append(gt_pos)
    print("Found {} good targets.".format(len(good_exps)))
    featurized_data = [good_exps, mh_lengths, gc_fracs, del_lens, freqs, dl_freqs, gt_poss]
    return featurized_data


def main(data_nm = 'train'):
  featurised_data = prepare_dataset(data_nm)
  pickle_featurized_data(featurised_data, data_nm)
  return

if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(data_nm = sys.argv[1])
  else:
    main()
