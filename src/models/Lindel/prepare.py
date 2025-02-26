import sys, os
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from Bio.Seq import Seq
from features import create_feature_array, onehotencoder
from src.models.Lindel.indels import gen_indel

from src.data.data_loader import load_Tijsterman_data, get_details_from_fasta, load_FORECasT_data
from src.config.test_setup import MIN_NUMBER_OF_READS

OUTPUT_DIR = os.environ['OUTPUT_DIR']
MIN_READS_PER_TARGET = MIN_NUMBER_OF_READS

def map_indels_to_dataframe(indels):
    # df columns: Type, Start, Size, InsSeq
    values = list(indels.values())
    mapped_indels = []
    for x in values:
        if x == '3':
            mapped_indels.append((x, "ins", 0, 3, "X"))
            continue
        a = x.split("+")
        if a[1][0].isalpha():
            mapped_indels.append((x, "ins", 0, a[0], a[1]))
        else:
            mapped_indels.append((x, "del", a[0], a[1], ""))
    return pd.DataFrame(mapped_indels\
        , columns=["Indel", "Type", "Start", "Size", "InsSeq"])\
        .astype({'Start': 'int64', 'Size': 'int64'})

def get_FORECasT_counts(dataset, sample_name, all_indels):
    FORECasT = load_FORECasT_data(sample_name, dataset)
    if type(FORECasT) == bool: return False
    FORECasT = FORECasT.rename(columns = {"F_InsSeq": "InsSeq", "F_counts": "Counts"})
    ge3_counts = FORECasT[(FORECasT.InsSeq.apply(len) > 2) & (FORECasT.Start == 0)].Counts.sum()
    FORECasT.loc[len(FORECasT.index)] = ["ins", 0, 3, "X", ge3_counts]
    counts = pd.merge(all_indels, FORECasT, how="left", on=["Type", "Start", "Size", "InsSeq"]).Counts.fillna(0)
    return counts

def get_Tijsterman_counts(dataset, sample_name, all_indels):
    Tijsterman = load_Tijsterman_data(dataset, sample_name, multi_index=False)
    if type(Tijsterman) == bool: return False
    Tijsterman = Tijsterman.rename(columns = {"T_InsSeq": "InsSeq", "T_counts": "Counts"})
    ge3_counts = Tijsterman[(Tijsterman.InsSeq.apply(len) > 2) & (Tijsterman.Start == 0)].Counts.sum()
    Tijsterman.loc[len(Tijsterman.index)] = ["ins", 0, 3, "X", ge3_counts]
    counts = pd.merge(all_indels, Tijsterman, how="left", on=["Type", "Start", "Size", "InsSeq"]).Counts.fillna(0)
    return counts

def get_features(seq, pam_index):
    cutsite = pam_index - 3
    seq = seq[cutsite-30:cutsite+35]
    pam = {'AGG':0,'TGG':0,'CGG':0,'GGG':0}
    guide = seq[13:33]
    if seq[33:36] not in pam:
        return ('Error: No PAM sequence is identified.')
    indels = gen_indel(seq, 30)
    input_indel = onehotencoder(guide)
    input_ins   = onehotencoder(guide[-6:])
    input_del   = np.concatenate((create_feature_array(features,indels),input_indel),axis=None)
    return input_indel, input_ins, input_del

def correct_inDelphi(o):
    o["TargetSequence"] = "GTCAT" + o["TargetSequence"] + "AGATCGGAAG"
    o["PAM Index"] = o["PAM Index"] + 5

    if o["Strand"] == "REVERSE":
        o["TargetSequence"] = str(Seq(o["TargetSequence"]).reverse_complement())
        o["PAM Index"] = o["PAM Index"] + 5
    return o

if __name__ == "__main__":
    FORECasT = ["test", "train", "HAP1_test", "HAP1_train", "TREX_A_train", "TREX_A_test", "2A_TREX_A_test"]
    LUMC = ["WT", "POLQ", "KU80", "LIG4"]
    inDelphi = ["052218-U2OS-+-LibA-postCas9-rep1_transfertrain", "052218-U2OS-+-LibA-postCas9-rep1_transfertest", "0226-PRLmESC-Lib1-Cas9_transfertrain", "0226-PRLmESC-Lib1-Cas9_transfertest", "0105-mESC-Lib1-Cas9-Tol2-BioRep2-techrep1"]
    DATASETS = FORECasT + inDelphi + LUMC
    DATASETS = ["052218-U2OS-+-LibA-postCas9-rep1_transfertrain", "0226-PRLmESC-Lib1-Cas9_transfertrain",\
        "052218-U2OS-+-LibA-postCas9-rep1_transfertest", "0226-PRLmESC-Lib1-Cas9_transfertest"]
    output = OUTPUT_DIR + "/model_training/data_{}x/Lindel/Tijsterman_Analyser/".format(MIN_READS_PER_TARGET)
    os.makedirs(output, exist_ok=True)
    for t in DATASETS:
        filename = t
        data = {}
        labels, rev_index, features, frame_shift = pkl.load(open("model_prereq.pkl", 'rb'))
        all_indels = map_indels_to_dataframe(rev_index)

        if t in FORECasT:
            parts = t.split("_")
            guides = get_details_from_fasta("../../data/FORECasT/{}.fasta".format(parts[-1]))
            t = "_".join(parts[:-1])
        elif t in inDelphi:
            parts = t.split("_")
            if len(parts) == 1:
                guides = get_details_from_fasta("../../data/inDelphi/LibA.forward.fasta")
            else:
                guides = get_details_from_fasta("../../data/inDelphi/{}.fasta".format(parts[-1]))
                t = parts[0]
            for o in guides:
                guides[o] = correct_inDelphi(guides[o])
        else:    
            guides = get_details_from_fasta("../../data/LUMC/{}.forward.fasta".format(t))

        for g in tqdm(guides.values()):
            counts = get_Tijsterman_counts(t, g["ID"], all_indels)
            if type(counts) == bool: continue
            if counts.sum() < MIN_READS_PER_TARGET: continue
            sequence = g["TargetSequence"]
            pam_index = g["PAM Index"]
            input_indel, input_ins, input_del = get_features(sequence, pam_index)
            data[g["ID"]] = (all_indels.Indel.to_list(), counts, input_indel, input_ins, input_del, g)
        print(len(list(data.keys())))
        pkl.dump(data , open(output + filename, "wb"))          
