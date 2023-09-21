import os, io, csv, time, sys, logging

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

if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[-1] != "dry":
        start_idx = eval(sys.argv[2])
        stop_idx = eval(sys.argv[3])
        cell_types = ["mESC"]
        output_dir = os.environ["OUTPUT_DIR"]
        for ct in cell_types:
            mapped_reads_dir = output_dir + "/mapped_reads_{}/".format(sys.argv[1]) + ct + "/"
            subdirs = sorted(os.listdir(mapped_reads_dir))[start_idx:stop_idx]
            start_time = time.time()
            for i,s in enumerate(subdirs):
                outfile = mapped_reads_dir + s + "/" + s + ".fastq"
                fastq_files = [(mapped_reads_dir +s + "/" + x) for x in os.listdir(mapped_reads_dir + s) if x[-6:] == ".fastq" and len(x.split("_")) == 4]
                if len(fastq_files) > 0:
                    cmd = "cat %s > %s" % (" ".join(sorted(fastq_files)), outfile)
                    end_time = time.time()
                    os.system(cmd)
                    logging.info("%d/%d created %s in %s seconds" % (i, len(subdirs), outfile, end_time - start_time))
    else:
        batch_size = 5000
        for i in range(0, 45000, batch_size):
            cmd = "sbatch combine_fastq_files.batch %s %d %d" % (sys.argv[1], i, i + batch_size)
            if sys.argv[-1] == "dry":
                print(cmd)
            else:
                os.system(cmd)

