import io, os, sys, time
from Bio import SeqIO
from datetime import datetime

sys.path.append("../../../data/FORECasT")
from data import find_ERR_for_ERS, load_sample_accession_ERS

if len(sys.argv) == 1:
    print("Provide the date of the last run of splitting the data that you want to align in the format YYYY-MM-DD")

datetime.strptime(sys.argv[1], '%Y-%M-%d')

output_folder = os.environ["OUTPUT_DIR"]

def partition_file(pear_file, input_dir, output_dir):
    start_time = time.time()

    for filename in os.listdir(output_dir):
        if filename.split("_pear")[0] == pear_file.split("_pear")[0]:
            os.remove(output_dir + '/' + filename)

    fouts = []
    for i in range(nump):
        partition_file = output_dir + "/" + pear_file[:-5] + '_%d.' % i + pear_file[-5:]
        fouts.append(io.open(partition_file,'w'))

    i, j = 0,0
    f = io.open(input_dir + "/" + pear_file)
    for line in f:
        fouts[i%nump].write(line)
        j += 1
        i += (j%4 == 0)
    f.close()
    for i in range(nump):
        fouts[i].close()

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    pear_file_dir = output_folder + "/assembled_reads_{}/".format(sys.argv[1])
    nump = 1000

    partitioned_reads_dir = output_folder + "partitioned_reads_{}/".format(sys.argv[1])
    if not os.path.exists(partitioned_reads_dir): os.mkdir(partitioned_reads_dir)

    for ct in ["2A_TREX_A"]:
        ct_pear_dir = pear_file_dir + ct
        ct_partitioned_reads_dir = partitioned_reads_dir + ct
        if not os.path.exists(ct_partitioned_reads_dir): os.mkdir(ct_partitioned_reads_dir)
        samples = find_ERR_for_ERS(load_sample_accession_ERS(ct))
        pear_files = [x for x in os.listdir(ct_pear_dir) if x.split('_')[-1] == 'pear.assembled.fastq' and x.split('_')[0] in samples]
        num_tasks = len(pear_files)
        for i, pear_file in enumerate(pear_files):
            if sys.argv[-1] == "dry":
                print(pear_file, ct_pear_dir, ct_partitioned_reads_dir)
            else:
                partition_file(pear_file, ct_pear_dir, ct_partitioned_reads_dir)
                sys.stderr.write('%s of %s done' % (i+1, num_tasks))













		

    
