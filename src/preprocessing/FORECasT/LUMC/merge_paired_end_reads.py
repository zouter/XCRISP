import sys, os
import pandas as pd

GENOTYPES = [
    "WT",
    "POLQ",
    "LIG4",
    "KU80"
]
TARGETS_F = "../../../data/LUMC/{}.txt"
OUTPUT_DIR = os.environ["OUTPUT_DIR"] + "processed_data/FORECasT/"
DATA_DIR = os.environ["DATA_DIR"] + "/LUMC/"
JOB_DIR = OUTPUT_DIR + "jobs/PEAR/"
COLUMNS = ["R1", "R2", "Name", "Alias", "Reference", "leftFlank", "rightFlank", "leftPrimer", "rightPrimer", "basesPastPrimer"]

def merge_paired_end_reads(name, r1_file, r2_file, output):
    output = "%s/%s" % (output, name)
    os.makedirs(output , exist_ok=True)
    cargs = (DATA_DIR, r1_file, DATA_DIR, r2_file, output, name) 
    cmd = 'pear -f %s/%s -r %s/%s -o %s/%s -n 20 -p 0.01 -u 0' % cargs
    print(cmd)
    os.system(cmd)
    prefix = "%s/%s" % (output, name)
    src = "%s.assembled.fastq" % prefix
    dest = "%s.fastq" % prefix
    os.rename(src, dest)

def executeBatchFile(x, mapped_reads_dir):
    name = x.Name
    r1_file = x.R1.split("/")[-1]
    r2_file = x.R2.split("/")[-1]
    os.makedirs(JOB_DIR, exist_ok=True)
    job_file = os.path.join(JOB_DIR, "%s.job" % name)
    with open(job_file, 'w+') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --job-name=%s.merge\n" % name)
        fh.writelines("#SBATCH --output=/scratch/cfseale/logs/slurm-%j.out\n")
        fh.writelines("#SBATCH --error=/scratch/cfseale/logs/slurm-%j.err\n" )
        fh.writelines("#SBATCH --time=02:00:00\n")
        fh.writelines("#SBATCH --mem=300\n")
        fh.writelines("#SBATCH --qos=short\n")
        fh.writelines("python3 merge_paired_end_reads.py %s %s %s %s" % (name, r1_file, r2_file, mapped_reads_dir))
    return job_file

def write_main_file(jobs):
    job_file = os.path.join(JOB_DIR, "merge_paired_end_reads.sh")
    with open(job_file, "w+") as f:
        for j in jobs:
            f.writelines("sbatch %s\n" % j)
    print("%s file written with %d jobs" % (job_file, len(jobs)))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        name = sys.argv[1]
        r1 = sys.argv[2]
        r2 = sys.argv[3]
        output = sys.argv[4]
        merge_paired_end_reads(name, r1, r2, output)
    else:
        jobs = []
        for genotype in GENOTYPES:
            targets = pd.read_csv(TARGETS_F.format(genotype), sep="\t", header=None, names=COLUMNS)
            output_dir = OUTPUT_DIR + genotype
            jobs += list(targets.apply(lambda  x: executeBatchFile(x, output_dir), axis=1))
        write_main_file(jobs)
