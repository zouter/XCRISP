import os, sys
import numpy as np
import pandas as pd
import pickle as pkl
from functools import reduce
import socket

hostname = socket.gethostname()

COUNTS_DIR = os.environ['OUTPUT_DIR'] 

def get_Tijsterman_Analyser_datafile(dataset, sample_name, parsed=True):
    if parsed:
        if dataset in ["test", "train"]:
            return COUNTS_DIR + "processed_data/Tijsterman_Analyser/FORECasT/{}.tij.sorted.tsv".format(sample_name)
        elif dataset in ["HAP1", "TREX_A", "2A_TREX_A"]:
            return COUNTS_DIR + "processed_data/Tijsterman_Analyser/{}/{}.tij.sorted.tsv".format(dataset, sample_name)
        return COUNTS_DIR + "processed_data/Tijsterman_Analyser/{}/{}_indels.tij.sorted.tsv".format(dataset, sample_name)
    else:
        if dataset in ["test", "train", "2A_TREX_A"]:
            return COUNTS_DIR + "processed_data/Tijsterman_Analyser/FORECasT/{}".format(sample_name)
        elif dataset in ["HAP1", "TREX_A", "HAP1_test", "TREX_A_test"]:
            return COUNTS_DIR + "processed_data/Tijsterman_Analyser/{}/{}".format(dataset, sample_name)
        return COUNTS_DIR + "processed_data/Tijsterman_Analyser/{}/{}_indels".format(dataset, sample_name)

def load_Tijsterman_data(dataset, sample_name, multi_index=True):
    f = get_Tijsterman_Analyser_datafile(dataset, sample_name)
    if not os.path.exists(f): 
        print("path does not exist " + f)
        return False
    r = pd.read_csv(f, sep="\t")
    if r.empty: return False
    r["Type"] = r.Type.apply(lambda x: "ins" if x[0] == "I" else "del")
    r["T_counts"] = r.countEvents
    r["T_InsSeq"] = r.InsSeq.fillna("")
    r = r[["Type", "Start", "Size", "T_InsSeq", "T_counts"]].groupby(["Type", "Size", "Start", "T_InsSeq"]).sum().reset_index()
    if multi_index:
        return r[["Type", "Start", "Size", "T_InsSeq", "T_counts"]].set_index(["Type", "Start", "Size"])
    else:
        return r[["Type", "Start", "Size", "T_InsSeq", "T_counts"]]


def load_Tijsterman_data_V2(dataset, sample_name):
    f = get_Tijsterman_Analyser_datafile(dataset, sample_name)
    if not os.path.exists(f): 
        print("path does not exist " + f)
        return False
    r = pd.read_csv(f, sep="\t")
    r = r[r.Type.isin(["DELETION", "INSERTION"])]
    return r

def read_target_sequence_from_file(target):
    sequence_file = os.environ["DATA_DIR"] + "LUMC/" + target
    seq = None
    with open(sequence_file, 'r') as f:
        f.readline()
        lines = f.readlines()
        seq = "".join([l.strip() for l in lines]).upper()
    return seq

def get_guides_from_fasta(oligo_f):
    guides = []
    with open(oligo_f, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            guides.append(lines[i].split()[0][1:])
    return guides

def get_details_from_fasta(oligo_f):
    guides = {}
    with open(oligo_f, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            details = lines[i].split()
            ID = details[0][1:]
            PAM_Index = eval(details[1])
            Direction = details[2]
            TargetSequence = lines[i+1].strip()
            guides[ID] = {
                "ID": ID,
                "PAM Index": PAM_Index,
                "Strand": Direction,
                "TargetSequence": TargetSequence
            }
    return guides

def get_FORECasT_datafile(sample_name, genotype):
    g = "FORECasT" if genotype in ["test", "train"] else genotype
    return "/Users/colm/repos/output/local/model_training/data/profiles/FORECasT/{0}/{1}_gen_indel_reads.txt".format(g, sample_name)

def convert_FORECasT_indel_to_Normal(indel):
    delins, pos = indel.split("_")
    t = "del" if delins[0] == "D" else "ins"
    size = eval(delins[1:])
    c_idx = pos.find("C")
    r_idx = pos.find("R")
    idx = c_idx if c_idx != -1 else r_idx
    homology = 0 if c_idx == -1 else eval(pos[c_idx+1:r_idx])
    start = eval(pos[1:idx]) + 1 + homology
    return t, start, size, homology

def load_FORECasT_data(sample_name, genotype):
    f = get_FORECasT_datafile(sample_name, genotype)
    r = pd.read_csv(f, sep="\t", skiprows=1).iloc[1:,:].reset_index()
    indels = pd.DataFrame(list(r.Indel.apply(convert_FORECasT_indel_to_Normal)), columns = ["Type", "Start", "Size", "homologyLength"])
    indels["F_counts"] = list(r.mESC)
    indels["F_InsSeq"] = r.apply(get_ins_seq, axis=1)
    indels["Indel"] = r.Indel

    # insertions = indels[indels.Type == "ins"]
    
    # insertions = indels[indels.Type == "ins"].apply(lambda x: pd.Series(x.InsSeq), axis=1).stack().reset_index(level=1, drop=1).to_frame("InsSeq").join(indels[["Type", "Start", "Size", "F_counts"]])
    # insertions["InsSeq"] = insertions.InsSeq.apply(lambda x: correct_FORECasT_reverse_input_sequences(x, is_reverse=is_reverse))

    # deletions = indels[indels.Type == "del"]
    # indels = pd.concat([deletions, insertions], ignore_index=True, sort=False)
    return indels[["Indel", "Type", "Start", "Size", "F_InsSeq", "F_counts"]]

def get_ins_seq(row):
    t = row.Indel[0]
    if t == "D":
        return ""

    x = row.Details
    A,T,G,C = 'A','T','G','C'
    AA,AT,AC,AG,CG,CT,CA,CC = 'AA','AT','AC','AG','CG','CT','CA','CC'
    GT,GA,GG,GC,TA,TG,TC,TT = 'GT','GA','GG','GC','TA','TG','TC','TT'
    return [a[2] for a in eval(x)]


def get_common_samples(genotype="train", min_reads=100, include_FORECasT = True):
    # load Lindel data
    lindel_data_f = os.environ["OUTPUT_DIR"] + "model_training/data_{}x/Lindel/Tijsterman_Analyser/{}".format(min_reads, genotype)
    lindel_d = list(pkl.load(open(lindel_data_f, 'rb')).keys())

    # load X-CRISP data
    ourmodel_data_f = os.environ["OUTPUT_DIR"] + "/model_training/data_{}x/X-CRISP/{}.pkl".format(min_reads, genotype) 
    ourmodel_d = list(pd.read_pickle(ourmodel_data_f)["counts"].index.levels[0])

    if include_FORECasT:
        # load FORECasT data
        forecast_data_f = os.environ["OUTPUT_DIR"] + "model_training/data_{}x/{}/Tijsterman_Analyser/".format(min_reads, genotype)
        forecast_d = list(os.listdir(forecast_data_f))

        common_samples = reduce(np.intersect1d, (lindel_d, forecast_d, ourmodel_d))
    else:
        common_samples = reduce(np.intersect1d, (lindel_d, ourmodel_d))

    return common_samples

if __name__ == "__main__":
    gt = sys.argv[1]
    cs = get_common_samples(gt, min_reads=100)
    print(len(cs))
