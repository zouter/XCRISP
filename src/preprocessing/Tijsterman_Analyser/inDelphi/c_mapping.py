
### Run a_split and b_alignment from the FORECasT pipeline first

import os, csv, sys
from datetime import datetime
import pandas as pd
from tqdm import tqdm

if len(sys.argv) == 1:
    print("Provide the date of the last run of splitting the data that you want to align in the format YYYY-MM-DD")

datetime.strptime(sys.argv[1], '%Y-%M-%d')

inp_dir = os.environ["OUTPUT_DIR"] + 'indelphi_b_alignment_{}/'.format(sys.argv[1])
reads_dir = os.environ["OUTPUT_DIR"] + 'indelphi_a_split_{}/'.format(sys.argv[1])
out_dir = os.environ["OUTPUT_DIR"] + 'processed_data/Tijsterman_Analyser/inDelphi/c_mapping_{}/'.format(sys.argv[1])

libf = "../../../data/inDelphi/LibA.fasta"
samplesf = "../../../data/inDelphi/samples.tsv"
os.makedirs(out_dir, exist_ok=True)

def get_genotype(sample_name):
    with open(samplesf) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            if r["Run Accession"] == sample_name:
                return r["Sample Alias"]

def get_targets(libf):
    targets = {}
    with open(libf) as f:
        while True:
            line = f.readline()
            if not line: break

            name = line[1:].split()[0]
            seq = f.readline().strip()
            targets[name] = seq
    return targets

def get_reads(assembled_fn):
    reads = {}
    with open(reads_dir + assembled_fn) as f:
        while True:
            line = f.readline()
            if not line: break

            i = line[1:].strip()
            seq = f.readline().strip()
            f.readline()
            quality = f.readline()
            reads[i] = {"s": seq, "q": quality}
    return reads


def main(sample):
    data = {}
    files = os.listdir(inp_dir)
    files = [f for f in files if sample in f]
    for fn in tqdm(files):
        parts = fn.split("_")
        sample_name = parts[0]
        ext = parts[-1].replace("mapping", "fastq")
        assembled_fn = sample_name + "_2_" + ext

        genotype = get_genotype(sample_name)
        targets = get_targets(libf)
        reads = get_reads(assembled_fn)
        
        # divide per oligo
        with open(inp_dir + fn, 'r') as f:
            f.readline()
            while True:
                line = f.readline()
                if not line: break

                parts = line.split("\t")
                read_id = parts[0]
                name = parts[1]
                if name == "None": continue
                if "," in name:
                    continue # we will discard ambiguities for now
                if name not in data:
                    data[name] = []
                data[name].append("@{}\n{}\n+\n{}".format(read_id, reads[read_id]["s"], reads[read_id]["q"]))
        print(data.keys())

        # write per oligo
        for k in data:
            odir = out_dir + genotype + "/" + k + "/"
            os.makedirs(odir, exist_ok=True)
            with open(odir + k + ".fastq", 'a') as w:
                for d in data[k]:
                    w.write("{}".format(d))
            print(odir + k + ".fastq")

if __name__ == "__main__":
    # os.system("rm -r {}*".format(out_dir))
    samples = pd.read_csv(samplesf, sep="\t")["Run Accession"]

    # run for single sample
    if len(sys.argv) == 3 and sys.argv[-1] in set(samples): 
        print("Running for {} {}".format(sys.argv[1], sys.argv[2]))
        main(sys.argv[2])
        exit()

    # run batch jobs for all samples
    for sample in samples:
        cmd = "sbatch c_mapping.batch {} {}".format(sys.argv[1], sample)
        if sys.argv[-1] == "dry":
            print(cmd)
        else:
            os.system(cmd)
