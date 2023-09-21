# find cromwell-executions/ -name '*indels*' -exec cp "{}" $PROTONDDR/local/processed_data/Tijsterman_Analyser/052218_U2OS_+_LibA_postCas9_rep1  \;

import sys, os
import pandas as pd
from tqdm import tqdm

out_dir = os.environ["OUTPUT_DIR"] + "processed_data/Tijsterman_Analyser/{}/"


def check_stats(f):
    stats_f = f + "_stats.txt"
    count = 0
    try:
        with open(stats_f) as sf:
            count = len(sf.readlines())
    except IOError:
        return False
    return count == 13

def parse_Tijsterman_output(f):    
    df = pd.read_csv(f, sep="\t")
    if df.empty: return

    sample_name = f
    
    output_f = "{}.tij.sorted.tsv".format(sample_name)
    if os.path.exists(output_f): return

    indels = df.loc[df.Type.isin(["DELETION", "INSERTION"]), :]
    indels["Size"] = indels["delSize"] + indels["insSize"]
    indels["Start"] = indels["delRelativeStart"]
    indels["InsSeq"] = indels["insertion"].fillna("")
    indels = indels[["Type", "Size", "Start", "InsSeq", "homologyLength", "countEvents"]]

    indels = indels.groupby(["Type", "Size", "Start", "InsSeq", "homologyLength"]).sum().reset_index()
    indels = indels.sort_values(by=["countEvents"])
    indels.to_csv(output_f, sep="\t", index=False)

if __name__ == "__main__":
    # for sample in ["0226-PRLmESC-Lib1-Cas9", "0105-mESC-Lib1-Cas9-Tol2-BioRep2-techrep1"]:
    for sample in ["052218-U2OS-+-LibA-postCas9-rep2"]:
        o = out_dir.format(sample)
        # os.system("rm {}*.tij.sorted.tsv".format(o))
        files = os.listdir(o)
        files = [x for x in files if ("tij" not in x) and ("stats" not in x)]

        for f in tqdm(files):
            
            if not check_stats(o + f):
                print(o)
                continue

            parse_Tijsterman_output(o + f)
