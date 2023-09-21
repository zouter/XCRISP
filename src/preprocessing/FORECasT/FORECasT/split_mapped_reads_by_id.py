import os, io, csv, time, sys, logging
from Bio import SeqIO

sys.path.append("../../../data/FORECasT")
from oligo import get_oligo_ids

from datetime import datetime
if len(sys.argv) == 1:
    print("Provide the date of the last run of splitting the data that you want to align in the format YYYY-MM-DD")
datetime.strptime(sys.argv[1], '%Y-%M-%d')

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

def loadMappings( filename ):
    lookup = {}
    f = io.open(filename)
    rdr = csv.reader(f, delimiter='\t')
    for toks in rdr:
        if toks[0][:3] == '@@@': continue
        lookup[toks[0]] = toks[1].split()[0]
    return lookup

def getFileForOligoIdx(oligo_idx):
    return "Oligo_{}".format(oligo_idx)

def write_to_file(oligo_id, reads, mapped_reads_dir, start_idx, stop_idx):
    logging.info("Writing...")
    o = mapped_reads_dir + '/' + getFileForOligoIdx( oligo_id )
    if not os.path.exists(o): os.makedirs(o)
    filename = "%s_%d_%d.fastq" %  (getFileForOligoIdx( oligo_id ), start_idx, stop_idx)
    fout = io.open(o + '/' + filename, 'w')
    for (read_id, read_seq, oid, phred_values) in reads:
        fout.write(u'@%s.%s\n%s\n+\n%s\n' % (oid.split('_')[0],read_id, read_seq, phred_values))
    fout.close()

def executeBatchFile(ids, processed_date, cell_type, job_dir, logs_dir, start_idx, stop_idx):
    filename = "split_%s_%s_%d_%d" % (ids[0], ids[-1], start_idx, stop_idx)
    job_file = os.path.join(job_dir,"%s.job" %(filename))
    cmd = "python3 split_mapped_reads_by_id.py {} {} {} {} {}\n".format(processed_date, cell_type, start_idx, stop_idx, " ".join(ids))
    with open(job_file, 'w+') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --job-name=%s_%s.job\n" % (filename, cell_type))
        fh.writelines("#SBATCH --output=%s/%s_%s.out\n" % (logs_dir, filename, cell_type))
        fh.writelines("#SBATCH --error=%s/%s_%s.err\n"  % (logs_dir, filename, cell_type))
        fh.writelines("#SBATCH --time=01:05:00\n") 
        fh.writelines("#SBATCH --mem=2000MB\n")
        fh.writelines("#SBATCH --qos=short\n")
        fh.writelines(cmd)
    
    if (sys.argv[-1] == "dry"):
        print("sbatch %s" % job_file)
    else:
        os.system("sbatch %s" % job_file)

# try splitting by oligo id
def split_mappings(oligo_ids, mapping_files_dir, mapped_reads_dir, start_idx, stop_idx):
    start_time = time.time()
    reads = {}
    mapping_files = sorted(os.listdir(mapping_files_dir))[start_idx:stop_idx]
    for i, map_file in enumerate(mapping_files):
        fastq_file = map_file[:-13] + '.fastq'
        lookup = loadMappings(mapping_files_dir + "/" + map_file)
        records = SeqIO.parse(partitioned_reads_dir + '/' + fastq_file,'fastq')

        for _, record in enumerate(records):
            if str(record.description) not in lookup:
                logging.info('Could not find', str(record.description),'in mapping file')
                continue
            oid = lookup[str(record.description)]
            if oid == 'None':
                continue
            if ',' in oid:
                continue # Ignore ambiguously mapped reads
            oidx = eval(oid[5:].split('_')[0])

            if (oidx in oligo_ids):
                if oidx not in reads: reads[oidx] = []
                reads[oidx].append((record.id, str(record.seq), oid, record.format("fastq").split("\n")[3]))        
        logging.info("{}:{} {} of {} mapping files".format(map_file, partitioned_reads_dir + '/' + fastq_file, i, len(mapping_files)))
    
    for oidx in oligo_ids:
        if oidx in reads and len(reads[oidx]) > 0:
            logging.info("Writing {} reads to {}".format(len(reads[oidx]), oidx))
            write_to_file(oidx, reads[oidx], mapped_reads_dir, start_idx, stop_idx)

    logging.info("--- %s seconds ---" % (time.time() - start_time))
    logging.info("Finished Oligo %s" % oligo_ids)

def init(cell_type, processed_date):
    output_dir = os.environ["OUTPUT_DIR"]
    exp_oligo_file = output_dir + "/exp_target_pam.fasta"
    mapped_reads_dir = output_dir + "/mapped_reads_{}/".format(processed_date) + cell_type
    mapping_files_dir = output_dir + "/mapping_files_{}/".format(processed_date) + cell_type
    partitioned_reads_dir = output_dir + "/partitioned_reads_{}/".format(processed_date) + cell_type
    jobs_dir = mapped_reads_dir + "/split_jobs/" + cell_type
    log_dir = os.environ["LOGS_DIR"]
    return output_dir, exp_oligo_file, mapped_reads_dir, mapping_files_dir, partitioned_reads_dir, jobs_dir, log_dir
            
if __name__ == "__main__":     
    if len(sys.argv) > 4:
        processed_date = sys.argv[1]
        cell_type = sys.argv[2]
        start_idx = eval(sys.argv[3])
        stop_idx = eval(sys.argv[4])
        oligo_ids = [eval(x) for x in sys.argv[5:]]
        output_dir, exp_oligo_file, mapped_reads_dir, mapping_files_dir, partitioned_reads_dir, jobs_dir, log_dir = init(cell_type, processed_date)
        start_time = time.time()
        split_mappings(oligo_ids, mapping_files_dir, mapped_reads_dir, start_idx, stop_idx)
        logging.info("--- %s seconds ---" % (time.time() - start_time))
    else:
        # before for testing, remove these in full run
        step = 2000
        cell_types = ["mESC"]
        processed_date = sys.argv[1]
        start_time = time.time()
        for ct in cell_types:
            output_dir, exp_oligo_file, mapped_reads_dir, mapping_files_dir, partitioned_reads_dir, jobs_dir, log_dir = init(ct, processed_date)
            if not os.path.exists(jobs_dir): os.makedirs(jobs_dir)
            if not os.path.exists(mapped_reads_dir): os.makedirs(mapped_reads_dir)
            oligo_ids = [str(x) for x in get_oligo_ids(exp_oligo_file)]
            n = len(oligo_ids)
            m = len(os.listdir(mapping_files_dir))
            batch_size = 1000
            for i in range(0, n, step):
                for start_idx in range(0, m, batch_size):
                    end = n if (i+step > n) else i+step
                    stop_idx = start_idx + batch_size
                    executeBatchFile(oligo_ids[i:end], processed_date, ct, jobs_dir, log_dir, start_idx, stop_idx)
        logging.info("--- %s : total time taken in seconds ---" % (time.time() - start_time))
