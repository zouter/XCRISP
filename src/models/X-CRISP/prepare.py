# mpiexec -n 6 python3 prepare.py
 
import sys, os, pickle
import pandas as pd
from tqdm import tqdm
from Bio.Seq import Seq
from indels import gen_indels_v3
from features import get_features, get_insertion_features

sys.path.append("../")
from data_loader import get_details_from_fasta, get_Tijsterman_Analyser_datafile
from test_setup import MIN_NUMBER_OF_READS

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

OUTPUT_DIR = os.environ['OUTPUT_DIR']
MIN_NUMBER_INDELS = MIN_NUMBER_OF_READS

def load_Tijsterman_data(dataset, sample_name):
    filename = get_Tijsterman_Analyser_datafile(dataset, sample_name)
    print(filename)
    if not os.path.exists(filename): 
        if VERBOSE: print("path does not exist " + filename)
        return None
    T = pd.read_csv(filename, sep="\t")
    if T.empty:
        if VERBOSE: print("dataframe is empty " + filename)
        return None
    return T

def get_counts(dataset, sample_name, indels):
    T = load_Tijsterman_data(dataset, sample_name)
    if T is None:
        indels["countEvents"] = 0
        return indels[["countEvents"]]

    T = T[T.Type.isin(indels.Type.unique())]
    # add counts > 3
    ge3counts = T[(T.Type == "INSERTION") & (T.Size > 2)]["countEvents"].sum()
    T.loc[len(T)] = ["INSERTION", 3, 0, "X", 0, ge3counts]

    counts = indels.copy().reset_index().set_index(["Type", "Size", "Start", "InsSeq"]).join(T.set_index(["Type", "Size", "Start", "InsSeq"])[["countEvents"]])
    counts["countEvents"] = counts["countEvents"].fillna(0)
    counts["fraction"] = counts["countEvents"]/counts["countEvents"].sum()
    counts["Sample_Name"] = sample_name
    return counts.reset_index().set_index(["Sample_Name", "Indel"])[["Type", "countEvents", "fraction"]]

def correct_inDelphi(o):
    o["TargetSequence"] = "GTCAT" + o["TargetSequence"] + "AGATCGGAAG"
    o["PAM Index"] = o["PAM Index"] + 5

    if o["Strand"] == "REVERSE":
        o["TargetSequence"] = str(Seq(o["TargetSequence"]).reverse_complement())
        o["PAM Index"] = o["PAM Index"] + 5
    return o

if __name__ == "__main__":
    output = OUTPUT_DIR + "/model_training/data_{}x/X-CRISP/".format(MIN_NUMBER_INDELS)
    os.makedirs(output, exist_ok=True)
    VERBOSE = True

    FORECasT = ["train", "test", "HAP1_train", "HAP1_test", "TREX_A_test", "TREX_A_train"]
    LUMC = ["WT", "POLQ", "KU80", "LIG4"]
    inDelphi = ["0226-PRLmESC-Lib1-Cas9_transfertrain", "0105-mESC-Lib1-Cas9-Tol2-BioRep2-techrep1", \
        "0226-PRLmESC-Lib1-Cas9_transfertest", "052218-U2OS-+-LibA-postCas9-rep1_transfertrain", \
        "052218-U2OS-+-LibA-postCas9-rep1_transfertest"]

    d = sys.argv[1]
    print(d)
    filename = d
    
    if d in FORECasT:
        parts = d.split("_")
        guides = list(get_details_from_fasta("../../data/FORECasT/{}.fasta".format(parts[-1])).values())
        d = "_".join(parts[:-1])
    if d in inDelphi:
        parts = d.split("_")
        if len(parts) == 1:
            guides = list(get_details_from_fasta("../../data/inDelphi/LibA.forward.fasta").values())
            guides = [correct_inDelphi(g) for g in guides]
        else:
            guides = list(get_details_from_fasta("../../data/inDelphi/{}.fasta".format(parts[-1])).values())
            guides = [correct_inDelphi(g) for g in guides]
            d = parts[0]
    if d in LUMC:
        guides = list(get_details_from_fasta("../../data/LUMC/{}.forward.fasta".format(d)).values())
    guides = [guides[i:len(guides):size] for i in range(size)]
    
    if rank != 0:
        guides = None
    guides = comm.scatter(guides, root=0)
    
    all_ins_features = []
    all_del_features = []
    all_counts = []
    for g in tqdm(guides):
        cutsite = g["PAM Index"]-3
        seq = g["TargetSequence"][cutsite-32:cutsite+32]
        indels = gen_indels_v3(seq, 32, max_deletion_length=30).set_index(["Indel"])
        print("getting counts from:", d)
        counts = get_counts(d, g["ID"], indels)
        if sum(counts.countEvents) < MIN_NUMBER_INDELS: 
            if VERBOSE: print("{} has less than {} reads".format(g["ID"], MIN_NUMBER_INDELS))
            continue
        # counts
        all_counts.append(counts)

        # deletion features
        del_features = get_features(indels[indels["Type"] == "DELETION"])
        del_features["Sample_Name"] = g["ID"]
        all_del_features.append(del_features)

        # insertion features
        ins_features = get_insertion_features(g["TargetSequence"], g["PAM Index"] - 3)
        ins_features["Sample_Name"] = g["ID"]
        all_ins_features.append(ins_features)

    all_counts = pd.concat(all_counts).reset_index().set_index(["Sample_Name", "Indel"])
    all_del_features = pd.concat(all_del_features).reset_index().set_index(["Sample_Name", "Indel"])
    all_ins_features = pd.DataFrame(all_ins_features).set_index("Sample_Name")
    
    all_counts = comm.gather(all_counts, root=0)
    all_del_features = comm.gather(all_del_features, root=0)
    all_ins_features = comm.gather(all_ins_features, root=0)

    if rank == 0:
        all_counts = pd.concat(all_counts)
        all_del_features = pd.concat(all_del_features)
        all_ins_features = pd.concat(all_ins_features)

        data = {
            "counts": all_counts, 
            "del_features": all_del_features, 
            "ins_features": all_ins_features
            }
        pickle.dump(data, open(output + filename + ".pkl", "wb"))
        print("Outputting file to", output + filename + ".pkl")
        print("Done.", rank)
    else:
        print("Done: ", rank)
    print(all_counts.index.get_level_values(0).unique().shape)
