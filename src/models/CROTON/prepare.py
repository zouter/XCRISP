# mpiexec -n 6 python3 prepare.py
 
import sys, os, pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio.Seq import Seq

from src.models.XCRISP.indels import gen_indels_v3
from src.data.data_loader import get_details_from_fasta, get_Tijsterman_Analyser_datafile
from src.config.test_setup import MIN_NUMBER_OF_READS

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

OUTPUT_DIR = os.environ['OUTPUT_DIR']
MIN_NUMBER_INDELS = MIN_NUMBER_OF_READS

print("Prepping data")

def load_Tijsterman_data(dataset, sample_name):
    filename = get_Tijsterman_Analyser_datafile(dataset, sample_name)
    if not os.path.exists(filename): 
        if VERBOSE: print("path does not exist " + filename)
        return None
    T = pd.read_csv(filename, sep="\t")
    if T.empty:
        if VERBOSE: print("dataframe is empty " + filename)
        return None
    return T

def get_stats(dataset, sample_name, indels):
    allonebpframeshifts = np.array([1, 4, 7, 10, 13, 16, 19, 22, 25, 28])
    alltwobpframeshifts = allonebpframeshifts + 1
    T = load_Tijsterman_data(dataset, sample_name)
    if T is None: return None
    counts = indels.copy().reset_index().set_index(["Type", "Size", "Start", "InsSeq"]).join(T.set_index(["Type", "Size", "Start", "InsSeq"])[["countEvents"]]).reset_index().fillna(0)

    total_counts = counts.countEvents.sum()

    del_freq = counts.loc[counts.Type == "DELETION", "countEvents"].sum()/total_counts
    prob_1bpins = counts.loc[(counts.Type == "INSERTION") & (counts.Size == 1), "countEvents"].sum()/total_counts
    prob_1bpdel = counts.loc[(counts.Type == "DELETION") & (counts.Size == 1), "countEvents"].sum()/total_counts
    one_bp_frameshift = counts.loc[counts.Size.isin(allonebpframeshifts), "countEvents"].sum()/total_counts
    two_bp_frameshift = counts.loc[counts.Size.isin(alltwobpframeshifts), "countEvents"].sum()/total_counts
    frameshift = counts.loc[counts.Size % 3 != 0, "countEvents"].sum()/total_counts

    stats = [del_freq, prob_1bpins, prob_1bpdel, one_bp_frameshift, two_bp_frameshift, frameshift]

    if np.isnan(stats).any(): 
        return None

    return stats

def correct_inDelphi(o):
    o["TargetSequence"] = "GTCAT" + o["TargetSequence"] + "AGATCGGAAG"
    o["PAM Index"] = o["PAM Index"] + 5

    if o["Strand"] == "REVERSE":
        o["TargetSequence"] = str(Seq(o["TargetSequence"]).reverse_complement())
        o["PAM Index"] = o["PAM Index"] + 5
    return o

def one_hot_encode(seq, base_map):
    seq = seq.upper()
    mapping = dict(zip(base_map, range(4))) 
    seq2 = [mapping[i] for i in seq]
    return np.eye(4)[seq2]

if __name__ == "__main__":
    output = OUTPUT_DIR + "/model_training/data_{}x/CROTON/".format(MIN_NUMBER_INDELS)
    os.makedirs(output, exist_ok=True)
    VERBOSE = True

    FORECasT = ["train", "test", "HAP1_test", "TREX_A_test"]
    LUMC = ["WT", "POLQ", "KU80", "LIG4"]
    inDelphi = ["0226-PRLmESC-Lib1-Cas9_train", "0226-PRLmESC-Lib1-Cas9_test", "0105-mESC-Lib1-Cas9-Tol2-BioRep2-techrep1", "052218-U2OS-+-LibA-postCas9-rep1_transfertest", "0226-PRLmESC-Lib1-Cas9_transfertest"]
    # DATASETS = FORECasT + LUMC + inDelphi
    # DATASETS = ["WT", "train", "test", "0105-mESC-Lib1-Cas9-Tol2-BioRep2-techrep1"]
    DATASETS = ["HAP1_test", "TREX_A_test", "052218-U2OS-+-LibA-postCas9-rep1_transfertest", "0226-PRLmESC-Lib1-Cas9_transfertest"]

    for d in DATASETS:
        print(d)
        filename = d
        if rank == 0:
            if d in FORECasT:
                parts = d.rsplit("_", 1)
                guides = list(get_details_from_fasta("./src/data/FORECasT/{}.fasta".format(parts[-1])).values())
                d = parts[0]
            if d in inDelphi:
                parts = d.split("_")
                if len(parts) == 1:
                    guides = list(get_details_from_fasta("./src/data/inDelphi/LibA.forward.fasta").values())
                    guides = [correct_inDelphi(g) for g in guides]
                elif parts[-1] == "transfertest":
                    guides = list(get_details_from_fasta("./src/data/inDelphi/transfertest.fasta".format(parts[-1])).values())
                    guides = [correct_inDelphi(g) for g in guides]
                    d = parts[0]
                else:
                    guides = list(get_details_from_fasta("./src/data/inDelphi/LibA_{}.fasta".format(parts[1])).values())
                    guides = [correct_inDelphi(g) for g in guides]
                    d = parts[0]
            if d in LUMC:
                guides = list(get_details_from_fasta("./src/data/LUMC/{}.forward.fasta".format(d)).values())
            guides = [guides[i:len(guides):size] for i in range(size)]
        else:
            guides = None
        guides = comm.scatter(guides, root=0)
        
        X = []
        Y = []
        z = []
        for g in tqdm(guides[:10]):
            cutsite = g["PAM Index"]-3
            seq = g["TargetSequence"][cutsite-30:cutsite+30]
            indels = gen_indels_v3(seq, 30, max_deletion_length=30).set_index(["Indel"])
            y = get_stats(d, g["ID"], indels)
            if y is None: continue
            x = one_hot_encode(seq, "ACGT")
            X.append(x)
            Y.append(y)
            z.append(g["ID"])

        data = {
            "stats": np.array(Y), 
            "input_seq": np.array(X),
            "samples": np.array(z)
        }
        pickle.dump(data, open(output + filename + ".pkl", "wb"))
    print("Done.")



            

