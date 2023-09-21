import os, sys
import time
from datetime import datetime 

sys.path.append("../../../data/FORECasT")
from data import find_ERR_for_ERS, load_sample_accession_ERS

if len(sys.argv) == 1:
    print("Provide the date of the last run of splitting the data that you want to align in the format YYYY-MM-DD")

datetime.strptime(sys.argv[1], '%Y-%M-%d')

def executeBatchFile(cmds, filename, job_dir, logs_dir, ct):
    job_file = os.path.join(job_dir,"%s.job" %filename)
    with open(job_file, 'w+') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --job-name=mapping_{}_%j.job\n".format(filename))
        fh.writelines("#SBATCH --output={}/slurm-%j.out\n".format(logs_dir))
        fh.writelines("#SBATCH --error={}/slurm-%j.err\n".format(logs_dir))
        fh.writelines("#SBATCH --time=08:00:00\n")
        fh.writelines("#SBATCH --ntasks=2\n")
        fh.writelines("#SBATCH --cpus-per-task=1\n")
        fh.writelines("#SBATCH --mem-per-cpu=580\n")
        fh.writelines("#SBATCH --qos=long\n")
        for cmd in cmds:
            fh.writelines("{}\n".format(cmd))
        fh.writelines("wait\n")
    
    if sys.argv[-1] == "dry":
        print("sbatch %s" % job_file)
    else:
        os.system("sbatch %s" % job_file)

if __name__ == "__main__":
    output_dir = os.environ["OUTPUT_DIR"]
    exp_oligo = output_dir + "/exp_target_pam.fasta"
    partitioned_reads_dir = output_dir + "/partitioned_reads_{}/".format(sys.argv[1])
    map_dir = output_dir + "/mapping_files_{}/".format(sys.argv[1])
    job_dir = map_dir + "mapping_jobs/"
    logs_dir = os.environ["LOGS_DIR"]
    max_cut_dist = 4
    num_files_per_job = 10

    os.makedirs(map_dir, exist_ok=True)
    os.makedirs(job_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    for ct in ["mESC"]:
        samples = find_ERR_for_ERS(load_sample_accession_ERS(ct))

        ct_partitioned_reads_dir = partitioned_reads_dir + ct
        ct_map_dir = map_dir + ct
        ct_job_dir = job_dir + ct

        os.makedirs(ct_map_dir, exist_ok=True)
        os.makedirs(ct_partitioned_reads_dir, exist_ok=True)
        os.makedirs(ct_job_dir, exist_ok=True)

        assembled_files = [x for x in os.listdir(ct_partitioned_reads_dir) if x.split("_")[0] in samples] 
        for i in range(0, len(assembled_files), num_files_per_job):
            start_time = time.time()
            cmds = []
            file_prefix = assembled_files[i].split("_pear")[0] + "_{}_{}".format(i, i+num_files_per_job)
            for afile in assembled_files[i:i+num_files_per_job]:
                cmd_args = (ct_partitioned_reads_dir, afile, exp_oligo, ct_map_dir, afile[:-6], max_cut_dist)
                cmd = 'srun -N 1 -n 1 indelmap %s/%s %s %s/%s_mappings.txt 1 %d &' % cmd_args
                cmds.append(cmd)
            executeBatchFile(cmds, file_prefix, ct_job_dir, logs_dir, ct)
        print("created {} batch jobs".format(len(assembled_files)/num_files_per_job))
