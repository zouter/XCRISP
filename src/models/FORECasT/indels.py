import os, io, csv
import pandas as pd
import time
from tqdm import tqdm
import sys
from Bio.Seq import Seq

from profile import read_summary_to_profile

sys.path.append("../")
from data_loader import load_Tijsterman_data, read_target_sequence_from_file, get_guides_from_fasta

OUTPUT_DIR = os.environ['OUTPUT_DIR']

def fasta(x):
    return ">{} {} {}\n{}\n".format(x["ID"], x["PAM Index"], x["Strand"], x["TargetSequence"])

def write_oligo_file(guides, t, d):
    filename = "{}/{}.tmp.fasta".format(OUTPUT_DIR, t)
    with open(filename, "w") as f:
        if d == "FORECasT":
            guides.apply(lambda x: f.write(fasta(x)), axis=1)
        else:
            used_sequences = set()
            for x in guides.iterrows():
                x = x[1]
                sequence = read_target_sequence_from_file(x[4].split("/")[-1])
                right_flank = x[6]
                left_flank = x[5]
                if right_flank[4:6] != "GG":
                    sequence = str(Seq(sequence).reverse_complement())
                    right_flank = str(Seq(left_flank).reverse_complement())    
                pam_index = sequence.find(right_flank) + 3
                intended_pam_index = 40
                sequence = sequence[pam_index-intended_pam_index:pam_index+45]
                # sometimes we have duplicated sequence for which we have analysed data. Take one
                if sequence not in used_sequences:
                    details = {"ID": x[2], "TargetSequence": sequence, "PAM Index": intended_pam_index, "Strand": "FORWARD"}
                    f.write(fasta(details))
                    used_sequences.add(sequence)
    return filename

def indelgen(oligo_f, output):    
    print("indelgen {} {} 4".format(oligo_f, output))
    os.system("indelgen {} {} 4".format(oligo_f, output))
    return None

def load_FORECasT_indels(oligo_id, t):
    f = OUTPUT_DIR + "model_training/data/FORECasT/indels/" + t + "/" + oligo_id + "_genindels.txt"
    p = pd.read_csv(f, sep="\t", skiprows=1, names=["Indel", "#Alts", "Details"])
    p["F_InsSeq"] = p.apply(get_ins_seq, axis=1)
    return p[["Indel", "F_InsSeq"]]

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

def get_ins_seq(row):
    t = row.Indel[0]
    if t == "D":
        return ""

    x = row.Details
    A,T,G,C = 'A','T','G','C'
    AA,AT,AC,AG,CG,CT,CA,CC = 'AA','AT','AC','AG','CG','CT','CA','CC'
    GT,GA,GG,GC,TA,TG,TC,TT = 'GT','GA','GG','GC','TA','TG','TC','TT'
    return [a[2] for a in eval(x)]

def get_T_counts(x, multi_B):
    try:
        y = multi_B.loc[x.Type, x.Start, x.Size, :]
    except KeyError:
        return 0
    if y.empty: return 0
    if x.Type == "ins":
        y = y[y.T_InsSeq.apply(lambda z: z in x.F_InsSeq)]
    return y.T_counts.sum() 

def load_FORECasT_indel_summary(oligo_id, t):
    t = get_FORECasT_or_LUMC_dir(t)
    f = OUTPUT_DIR + "model_training/data/profiles/FORECasT/{}/{}_gen_indel_reads.txt".format(t, oligo_id)
    # f = OUTPUT_DIR + "processed_data/FORECasT/{}/{}/{}_gen_indel_reads.txt".format(t, oligo_id, oligo_id)
    indels = []
    if not os.path.exists(f): 
        print(f)
        return None

    with open(f, 'r') as indel_f:
        indel_f.readline()
        indel_f.readline()
        indel_f.readline()
        for l in indel_f.readlines():
            a = l.split("\t")
            indels.append(( a[0], eval(a[2].strip()) ))
    return pd.DataFrame(indels, columns=["Indel", "Counts"])

def get_mapped_indel_summary_files(oligo_id, dataset, test_file):
    if dataset == "FORECasT":
        return OUTPUT_DIR + '/processed_data/FORECasT/FORECasT/' + oligo_id + '/' + oligo_id + "_mappedindelsummary.txt"
    elif dataset == "LUMC":
        return OUTPUT_DIR + '/processed_data/FORECasT/' + test_file + '/' + oligo_id + '/' + oligo_id + "_mappedindelsummary.txt"
    elif dataset == "inDelphi":
        return OUTPUT_DIR + '/processed_data/FORECasT/inDelphi/c_mapping/' + test_file + '/' + oligo_id + '/' + oligo_id + "_mappedindelsummary.txt"

def compile_reads_for_FORECasT(oligo_id, dataset, test_file):  
    d = test_file if dataset != "FORECasT" else dataset

    gen_indel_file = OUTPUT_DIR + "model_training/data/FORECasT/indels/" + d + '/'  + oligo_id + "_genindels.txt"
    sum_file = get_mapped_indel_summary_files(oligo_id, dataset, test_file)
    out_dir = OUTPUT_DIR + 'model_training/data/profiles/FORECasT/' + d + '/'
    #Read all profiles for this oligo
    profile, mut_read_totals = {}, []
   
    print(sum_file)

    if not os.path.isfile(sum_file): 
        print("file does not exist")
        return        

    stats = read_summary_to_profile(sum_file, profile, oligoid=oligo_id)
    mut_read_totals.append('%d' % (stats[0]-stats[2]))

    #Compile reads for each indel across all samples
    f = io.open(gen_indel_file)
    fout = io.open(out_dir + '%s_gen_indel_reads.txt' % (oligo_id), 'w')
    fout.write(f.readline())    #Git commit
    fout.write(u'Indel\tDetails\t%s\n' % '\t'.join(["mESC"]))
    fout.write(u'All Mutated\t[]\t%s\n' % '\t'.join(mut_read_totals))
    print("%s: %s" % (oligo_id, mut_read_totals))
    for toks in csv.reader(f,delimiter='\t'):
        indel, indel_details = toks[0], toks[2]
        read_str = '\t'.join(['%d' % (profile[indel] if indel in profile else 0)])
        fout.write(u'%s\t%s\t%s\n' % (indel, indel_details, read_str))
    fout.close()
    f.close()

def get_FORECasT_or_LUMC_dir(t):
    if t in ["test", "train"]:
        return "FORECasT"
    return t

if __name__ == "__main__":
    start_time = time.time()

    datasets = [
        ("FORECasT", "train"),
        ("FORECasT", "test"),
        ("FORECasT", "TREX_A_test"),
        ("FORECasT", "HAP1_test"),
        ("LUMC", "WT"),
        ("LUMC", "POLQ"),
        ("LUMC", "KU80"),
        ("LUMC", "LIG4"),
        ("inDelphi", "0226-PRLmESC-Lib1-Cas9"),
        ("inDelphi", "0105-mESC-Lib1-Cas9-Tol2-BioRep2-techrep1"),
        ("inDelphi", "052218-U2OS-+-LibA-postCas9-rep1")
    ]
    
    d = sys.argv[1]
    t = sys.argv[2]

    if (d, t) not in datasets:
        print("dataset does not exist", d, t)
        exit()

    if d == "FORECasT":
        parts = t.split("_")
        oligo_f = "../../data/FORECasT/{}.fasta".format(parts[-1])
        t = "_".join(parts[:-1])
    elif d == "inDelphi":
        oligo_f = "../../data/inDelphi/LibA.forward.fasta"
    else:
        oligo_f = "../../data/{}/{}.forward.fasta".format(d, t)

    # Uncomment below if you need to generate indel files from scratch
    genotype = t 
    output = OUTPUT_DIR + "model_training/data/FORECasT/indels/{}/".format(genotype)
    os.makedirs(output, exist_ok=True)
    # indelgen(oligo_f, output)
    guides = get_guides_from_fasta(oligo_f)
    print("Run time: " + str(time.time() - start_time))

    for g in tqdm(guides):
        # load forecast indels 
        FORECasT = load_FORECasT_indels(g, t)
        # map to Tijsterman profiles
        mappings = FORECasT.Indel.apply(convert_FORECasT_indel_to_Normal)
        mappings = pd.DataFrame(list(mappings), columns=["Type", "Start", "Size", "Homology Length"])
        # load Tijsterman profiles
        Tijsterman = load_Tijsterman_data(t, g)
        if type(Tijsterman) == bool: 
            print("Tijsterman data does not exist for " + g)
            continue
        # get counts
        combined = pd.concat([FORECasT, mappings], axis=1)
        combined["Counts"] = combined.apply(lambda x: get_T_counts(x, Tijsterman), axis=1)
        combined["-"] = "-"
        # save as FORECasT indel profiles
        o = OUTPUT_DIR + "model_training/data/FORECasT/profiles/Tijsterman_Analyser/" + genotype + "/"
        os.makedirs(o, exist_ok=True)
        combined[["Indel", "-", "Counts"]].to_csv(o + g, sep="\t", index=False)
    print("Finished {}.".format(t))

    # old FORECasT processing code
    # if sys.argv[1] == "FORECasT":
    #     for g in tqdm(guides):
    #         compile_reads_for_FORECasT(g, d, t)

    #         valid_indels = load_FORECasT_indels(g, t)
    #         indel_summary = load_FORECasT_indel_summary(g, t)
    #         if indel_summary is None: 
    #             print("No summary for " + g)
    #             continue
    #         combined = pd.merge(valid_indels, indel_summary, how="left", on="Indel").fillna(0)
    #         combined["-"] = "-"
    #         # save as FORECasT indel profiles
    #         o = OUTPUT_DIR + "model_training/data/profiles/FORECasT/" + genotype + "/"
    #         os.makedirs(o, exist_ok=True)
    #         combined[["Indel", "-", "Counts"]].to_csv(o + g, sep="\t", index=False)
            
