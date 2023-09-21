import sys, os, logging, time, datetime, random, math

sys.path.append("../../common")
from data import get_output_folder, get_logs_folder

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

def get_size(oligo_id):
    return os.path.getsize(mESC_mapped_reads_dir + oligo_id + "/" + oligo_id + ".fastq")

def get_summed_size(oligo_dirs):
    sizes = [get_size(x) for x in oligo_dirs]
    return sum(sizes)

def get_predicted_time(oligo_id):
    x = get_size(oligo_id)
    if (x == 0): return 0
    x = math.log(x)
    a = 3.195269e+01
    b = -2.630928e+00
    b2 = 8.447823e-06
    t = round(a + b*x + b2*math.exp(x))*.75
    return max(t,0)
    # return str(datetime.timedelta(minutes=t))

def execute_batch_file(oligo_ids, time, job_dir, logs_dir):
    filename = "exptarget_%s" % (oligo_ids[0])
    job_file = os.path.join(job_dir,"%s.job" %filename)
    with open(job_file, 'w+') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --job-name=%s.job\n" % filename)
        fh.writelines("#SBATCH --output=%s/%s.out\n" % (logs_dir, filename))
        fh.writelines("#SBATCH --error=%s/%s.err\n"  % (logs_dir, filename))
        fh.writelines("#SBATCH --ntasks=1\n")
        fh.writelines("#SBATCH --time=%s\n" % str(datetime.timedelta(minutes=time)))
        fh.writelines("#SBATCH --cpus-per-task=1\n")
        fh.writelines("#SBATCH --mem=80\n")
        fh.writelines("#SBATCH --qos=short\n")
        for oligo_id in oligo_ids:
            taskname = "%s_%s" % (oligo_id, get_size(oligo_id))
            cmd = "srun -n 1 -c 1 -J %s python3 map_reads_to_expanded_targets.py %s\n" % (taskname, oligo_id)
            fh.writelines(cmd)
            fh.writelines("# predicted time: %d minutes\n" % get_predicted_time(oligo_id)) 
    logging.info("Created %s" % filename)
    # os.system("sbatch %s" % job_file)

if __name__ == "__main__":
    max_cut_dist = 4
    null_mapped_reads_dir = get_output_folder() + "/FORECasT/mapped_reads/NULL/"
    mESC_mapped_reads_dir = get_output_folder() + "/FORECasT/mapped_reads/mESC/"
    job_dir = get_output_folder() + "/FORECasT/exptarget_jobs/"
    if not os.path.exists(job_dir): os.mkdir(job_dir)
    logs_dir = get_logs_folder()

    if len(sys.argv) == 2:
        start_time = time.time()
        oligo_dir = sys.argv[1]
        prefix = oligo_dir + "/" + oligo_dir
        null_exp_filename = null_mapped_reads_dir + prefix + "_exptargets.txt"
        mESC_mapped_reads_filename = mESC_mapped_reads_dir + prefix + ".fastq"             
        output_filename = mESC_mapped_reads_dir + prefix + "_mappedindels.txt" 

        if not os.path.exists(null_exp_filename):
            logging.info("exptargets.txt files do not exist for %s" % oligo_dir)       
        elif not os.path.exists(mESC_mapped_reads_filename):
            logging.info("fastq files do not exist for %s" % oligo_dir)       
        else:
            logging.info("Beginning %s" % oligo_dir)
            cmd = "indelmap %s %s %s 0 4" % (mESC_mapped_reads_filename, null_exp_filename, output_filename)   
            logging.info(cmd)
            os.system(cmd) 
            cmd2 = 'python3 reformat_indel_profile.py %s' % mESC_mapped_reads_dir + prefix 
            logging.info(cmd2)
            os.system(cmd2)
        end_time = time.time()
        logging.info("Completed {} in {} seconds".format(oligo_dir, end_time - start_time))
    else:
        oligo_dirs = [x for x in os.listdir(mESC_mapped_reads_dir)]
        oligo_dirs = random.sample(oligo_dirs, 200)
        os.system("rm %s*" % job_dir)
        created_jobs = 0
        predicted_times = []
        batch_oligos = []
        for oligo_id in oligo_dirs:
            if len(batch_oligos) >=1 and sum(predicted_times) + get_predicted_time(oligo_id) > 100:
                ordered_indices = sorted(range(len(predicted_times)), key=lambda k: predicted_times[k])
                batch_oligos = [batch_oligos[i] for i in ordered_indices[::-1]]
                execute_batch_file(batch_oligos, sum(predicted_times), job_dir, logs_dir)
                batch_oligos = []
                predicted_times = []
                created_jobs += 1
            else:
                batch_oligos.append(oligo_id)
                predicted_times.append(get_predicted_time(oligo_id))

        logging.info("Created %d files" % created_jobs)
