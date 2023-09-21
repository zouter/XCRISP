# python3 join_paired_end_reads.py /scratch/cfseale/data/PRJEB29746 
import io, os, sys
import pandas as pd
from datetime import date

sys.path.append("../../../data/FORECasT")
from data import find_ERR_for_ERS, load_sample_accession_ERS

def executeBatchFile(cmd, filename, job_dir, logs_dir):
    job_file = os.path.join(job_dir,"%s.job" %filename)
    with open(job_file, 'w+') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --job-name=%s.job\n" % filename)
        fh.writelines("#SBATCH --output=%s%s.out\n" % (logs_dir, filename))
        fh.writelines("#SBATCH --error=%s%s.err\n"  % (logs_dir, filename))
        fh.writelines("#SBATCH --time=03:00:00\n")
        fh.writelines("#SBATCH --mem=300\n")
        fh.writelines("#SBATCH --qos=short\n")
        fh.writelines(cmd)

    cmd = "sbatch %s" % job_file
    if len(sys.argv) > 1 and sys.argv[1] == "dry":
        print(cmd)
    else:
        os.system(cmd)

if __name__ == "__main__":
    # cell_types = ["NULL", "mESC"]
    cell_types = ["2A_TREX_A"]

    output_dir = os.environ["OUTPUT_DIR"]
    assembled_reads_dir = output_dir + "/assembled_reads_{}".format(date.today()) 
    job_dir = assembled_reads_dir + "/pear_jobs"
    logs_dir = os.environ["LOGS_DIR"]
    fastq_dir = os.environ["DATA_DIR"] + "PRJEB29746/"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(assembled_reads_dir, exist_ok=True)
    os.makedirs(job_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    for ct in cell_types:
        os.makedirs(assembled_reads_dir + "/" + ct, exist_ok=True)
        samples = find_ERR_for_ERS(load_sample_accession_ERS(ct))
        r1_files = [x for x in os.listdir(fastq_dir) if x[-11:] == '_1.fastq.gz' and x[:-11] in samples] 
        for r1_file in r1_files:
            file_prefix = r1_file[:r1_file.index('_1')]
            file_suffix = r1_file[len(file_prefix):].replace('1','2',1)
            if not os.path.isfile(fastq_dir + '/' + file_prefix + file_suffix):
                print('Could not find matching R2 file:', fastq_dir, file_prefix, file_suffix)
                continue
            cargs = (fastq_dir, r1_file, fastq_dir, file_prefix, file_suffix, assembled_reads_dir, ct, file_prefix) 
            cmd = 'pear -f %s/%s -r %s/%s%s -o %s/%s/%s_pear -n 20 -p 0.01' % cargs
            executeBatchFile(cmd, file_prefix, job_dir, logs_dir)


