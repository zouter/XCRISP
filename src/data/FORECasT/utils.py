import os, random, logging
import pandas as pd
from Bio import SeqIO

from data import get_reads_folder, get_forecast_target_details

def get_guides():
    target_details = pd.read_csv(get_forecast_target_details(), sep="\t")
    guides = target_details[target_details.Subset == "Explorative gRNA-Targets"]
    guides = guides[(guides["PAM Index"] != 56) & (guides["Strand"] == "FORWARD")]
    guides["ID"] = guides["ID"].apply(lambda x: x[:5] + "_" + x[5:])
    return guides

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    dir = os.path.expanduser(dir)
    if not os.path.exists(dir):
        os.mkdir(dir)

def load_fastq_reads_by_id( filename ):
    lookup = {}
    for record in SeqIO.parse(filename,'fastq'):
        lookup[str(record.id)] = str(record.seq)
    return lookup

def get_reads_file(guide, dataset):
    if dataset == "forecast":
        return "%s/%s/%s_gen_indel_reads.txt" %  (get_reads_folder("forecast"), guide, guide.replace("_", ""))
    else:
        return "%s/%s_gen_indel_reads.txt" %  (get_reads_folder(dataset), guide)

def get_reads_for_guide(guide, dataset):
    reads_file = get_reads_file(guide, dataset)
    with open(reads_file) as myfile:
        reads = eval([next(myfile) for x in range(3)][2].split("\t")[2])
    return reads

def select_n_guides(guides, n, min_reads = 100, random_state=0, dataset="forecast"):
    guides = list(guides)
    random.seed(random_state)
    indices = list(range(0, len(guides)))
    random.shuffle(indices)
    selected = []
    for i in indices:
        if get_reads_for_guide(guides[i], dataset) > min_reads:
            selected.append(guides[i])
            logging.debug("Selected %d candidates" % len(selected))
        if len(selected) >= n: break
    return selected