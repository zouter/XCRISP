import sys, os
import pandas as pd
from Bio.Seq import Seq
from tqdm import tqdm

DOWNLOADS_DIR = os.environ["OUTPUT_DIR"] + "processed_data/Tijsterman_Analyser/{}/"
CHECK_STATS = False


def check_stats(f):
    stats_f = f + "_indels_stats.txt"
    count = 0
    try:
        with open(stats_f) as sf:
            count = len(sf.readlines())
    except IOError:
        print(stats_f)
        return False
    return count == 14

def read_fasta_file(fasta_f):
    samples = {}
    with open(fasta_f, 'r') as f:
        lines = f.readlines()
        for l in range(0, len(lines), 2):
            details = lines[l].split()
            name = details[0][1:]
            pam_idx = eval(details[1])
            direction = details[2]
            samples[name] = (name, pam_idx, direction)
    return samples

def get_samples(genotype):
    if genotype in ["test", "train"]:
        samples = read_fasta_file("../../../data/FORECasT/{}.fasta".format(genotype))
    else:
        samples = read_fasta_file("../../../data/LUMC/{}.fasta".format(genotype))
    return samples

def parse_Tijsterman_output(f, is_reverse=False):    
    df = pd.read_csv(f, sep="\t")
    if df.empty: return

    sample_name = f
    
    output_f = "{}.tij.sorted.tsv".format(sample_name)
    if os.path.exists(output_f): return

    indels = df.loc[df.Type.isin(["DELETION", "INSERTION"]), :]
    indels["Size"] = indels["delSize"] + indels["insSize"]
    indels["Start"] = indels["delRelativeStart"]
    indels["InsSeq"] = indels["insertion"].fillna("")

    if is_reverse:
        indels["Start"] = indels.apply(reverse_start_loc, axis=1)
        indels["InsSeq"] = indels["InsSeq"].apply(reverse_complement)

    indels = indels[["Type", "Size", "Start", "InsSeq", "homologyLength", "countEvents"]]

    indels = indels.groupby(["Type", "Size", "Start", "InsSeq", "homologyLength"]).sum().reset_index()
    indels = indels.sort_values(by=["countEvents"])
    indels.to_csv(output_f, sep="\t", index=False)

def reverse_complement(x):
    return str(Seq(x).reverse_complement())

def reverse_start_loc(x):
    if x["Type"] == "DELETION":
        return reverse_del_start_loc(x)
    else:
        return reverse_ins_start_loc(x)

def check_reverse_repeat_length(x):
    leftFlankReversed = x["leftFlank"][::-1]
    delSeq = x["del"]
    homology = x["homology"]
    homologyLength = len(homology)
    numRepeats = 1
    for i in range(1, len(leftFlankReversed), homologyLength):
        seq = leftFlankReversed[homologyLength*i:(homologyLength*i + homologyLength)]
        if seq != delSeq:
            break    
        numRepeats += 1
    return numRepeats
        
def reverse_del_start_loc(x):
    numRepeats = 1
    if int(x["countEvents"]) == 31577:
        if len(x["homology"]) == len(x["del"]):
            numRepeats = check_reverse_repeat_length(x)
    return -x["Size"] - x["Start"] + (x["homologyLength"] * numRepeats)

def reverse_ins_start_loc(x):
    insseq = x["insertion"]
    start = int(x["delStart"])

    is_rpt = x["Raw"][start-len(insseq):start] == insseq
    rep_length = x["insSize"] if is_rpt else 0
    return -x["Start"] + rep_length

if __name__ == "__main__":
    genotypes = ["WT", "POLQ", "LIG4", "KU80"]
    genotypes = ["WT2"]
    for t in genotypes:
        d = DOWNLOADS_DIR.format(t)
        os.system("rm {}*.tij.sorted.tsv".format(d))
        samples = get_samples(t)

        for f in tqdm(samples):
            if CHECK_STATS and not check_stats(DOWNLOADS_DIR.format(t) + f):
                continue

            is_reverse = samples[f][2] == "REVERSE"
            parse_Tijsterman_output(DOWNLOADS_DIR.format(t) + f + "_indels", is_reverse=is_reverse)

