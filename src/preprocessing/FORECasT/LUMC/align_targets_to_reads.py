# for each read
# create tmp file with only one target
# run indelmap to generate alignment output for that target
import os, sys
from Bio import SeqIO

from reformat_indel_profile import reformat_indel_profile

GENOTYPES = [
    "WT",
    "POLQ",
    "LIG4",
    "KU80"
]

TMP_FASTA_FILE = "%s.tmp.fasta"
LOGS_DIR = "/scratch/cfseale/logs/"
OUTPUT_DIR = os.environ["OUTPUT_DIR"] + "processed_data/FORECasT/"
DATA_DIR = os.environ["DATA_DIR"] + "/LUMC/"
JOB_DIR = OUTPUT_DIR + "jobs/align/"

def align_reads_to_target(name, description, seq, genotype):
    # create tmp target file to limit mapping to single target
    tmp = TMP_FASTA_FILE % name
    with(open(tmp, 'w')) as f:
        f.write(">%s\n" % description)
        f.write("%s\n" % seq)
    mapped_reads_output_dir = "%s/%s/%s"  % (OUTPUT_DIR, genotype, name)
    assembled_fastq_file = "%s/%s.fastq" % (mapped_reads_output_dir, name)
    output = assembled_fastq_file.replace(".fastq", "_mappedindels.txt")        
    # map reads to target
    cmd = "indelmap %s %s %s 0 4" % (assembled_fastq_file, tmp, output)
    print(cmd)
    os.system(cmd)
    os.remove(tmp)
    return output

def create_batch_file(record, genotype):
    name = record.description.split()[0]
    cmd = "python3 align_targets_to_reads.py %s %s %s" % (record.description, record.seq, genotype)
    job_dir = "%s/%s" % (JOB_DIR, genotype)
    os.makedirs(job_dir, exist_ok=True)
    job_file = os.path.join(job_dir, "%s.job" % name)
    with open(job_file, 'w+') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --job-name=%s.align\n" % name.replace("#", "_"))
        fh.writelines("#SBATCH --output=/scratch/cfseale/logs/slurm-%j.out\n")
        fh.writelines("#SBATCH --error=/scratch/cfseale/logs/slurm-%j.err\n")
        fh.writelines("#SBATCH --time=10:00:00\n")
        fh.writelines("#SBATCH --qos=long\n")
        fh.writelines("#SBATCH --mem=3GB\n")
        fh.writelines(cmd)
    return job_file

def write_main_file(jobs, genotype):
    job_dir = "%s/%s" % (JOB_DIR, genotype)
    job_file = os.path.join(job_dir, "align_targets_to_reads.sh")
    with open(job_file, "w+") as f:
        for j in jobs:
            f.writelines("sbatch %s\n" % j)
    print("%s file written with %d jobs" % (job_file, len(jobs)))
    os.system("sh " + job_file)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        name = sys.argv[1]
        pam_loc = sys.argv[2]
        direction = sys.argv[3]
        sequence = sys.argv[4]
        genotype = sys.argv[5]
        description = "%s %s %s" % (name, pam_loc, direction)
        print("Mapping Indels...")
        mapped_indels_file = align_reads_to_target(name, description, sequence, genotype)
        print("Reformatting...")
        mapped_indels_prefix = mapped_indels_file[:-17]
        reformat_indel_profile(mapped_indels_prefix)
        print("Done.")
    else:
        for genotype in GENOTYPES:
            records = SeqIO.parse('../../../data/LUMC/{}.fasta'.format(genotype), 'fasta') 
            jobs = []  
            for r in records:
                if r.description.split()[1] != "?":
                    jobs.append(create_batch_file(r, genotype))
                else:
                    print("PAM location unknown, skipping %s" % r.id)
            write_main_file(jobs, genotype)
