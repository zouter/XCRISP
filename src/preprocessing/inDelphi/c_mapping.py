import os, csv, sys
from datetime import datetime
import pandas as pd
from tqdm import tqdm

if len(sys.argv) == 1:
    print("Provide the date of the last run of splitting the data that you want to align in the format YYYY-MM-DD")

datetime.strptime(sys.argv[1], '%Y-%M-%d')

inp_dir = os.environ["OUTPUT_DIR"] + 'processed_data/FORECasT/inDelphi/b_alignment_{}/'.format(sys.argv[1])
reads_dir = os.environ["OUTPUT_DIR"] + 'processed_data/FORECasT/inDelphi/a_split_{}/'.format(sys.argv[1])
out_dir = os.environ["OUTPUT_DIR"] + 'processed_data/FORECasT/inDelphi/c_mapping_{}/'.format(sys.argv[1])

libf = "../../../data/inDelphi/LibA.fasta"
samplesf = "../../../data/inDelphi/samples.tsv"
os.makedirs(out_dir, exist_ok=True)

def get_genotype(sample_name):
    with open(samplesf) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            if r["Run Accession"] == sample_name:
                return r["Sample Alias"]

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

        targets = {}
        with open(libf) as f:
            while True:
                line = f.readline()
                if not line: break

                name = line[1:].split()[0]
                seq = f.readline().strip()
                targets[name] = seq


        # read all reads
        reads = {}
        with open(reads_dir + assembled_fn) as f:
            while True:
                line = f.readline()
                if not line: break

                i = line[1:].strip()
                seq = f.readline().strip()
                reads[i] = seq

                f.readline()
                f.readline()

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
                data[name].append("{}\t{}\t{}".format(reads[read_id], targets[name], line))


        # write per oligo
        for k in data:
            odir = out_dir + genotype + "/" + k + "/"
            os.makedirs(odir, exist_ok=True)
            with open(odir + k + ".mapping", 'a') as w:
                for d in data[k]:
                    w.write("{}".format(d))
    for k in tqdm(data):
        odir = out_dir + genotype + "/" + k + "/"
        df = pd.read_csv(odir + k + ".mapping", sep="\t", names=["Read", "Target Sequence", "Read ID", "Target Name", "Indel", "Mutations", "Perc1", "Perc2", "Perc3"])
        # df = df[["Indel", "Mutations"]].groupby(["Indel", "Mutations"]).size().reset_index(name="Count")
        df = df[["Indel"]].groupby(["Indel"]).size().reset_index(name="Count")
        df["-"] = "-"
        with open(odir + k + "_mappedindelsummary.txt", "w") as f:
            f.write('@@@{}\n'.format(k))
            df[["Indel", "-", "Count"]].to_csv(f, sep="\t", header=False, index=False)

if __name__ == "__main__":
    os.system("rm -r {}*".format(out_dir))
    samples = pd.read_csv(samplesf, sep="\t")["Run Accession"]

    # run for single sample
    if len(sys.argv) == 3 and sys.argv[-1] in samples: 
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
        


