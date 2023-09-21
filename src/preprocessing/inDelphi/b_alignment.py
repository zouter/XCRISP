import os, sys
from datetime import datetime
from Bio.Seq import Seq
from tqdm import tqdm

if len(sys.argv) == 1:
    print("Provide the date of the last run of splitting the data that you want to align in the format YYYY-MM-DD")

datetime.strptime(sys.argv[1], '%Y-%M-%d')

lib_dir = "../../data/inDelphi/"
inp_dir = os.environ["OUTPUT_DIR"] + 'indelphi_a_split_{}/'.format(sys.argv[1])
rev_dir = os.environ["OUTPUT_DIR"] + 'indelphi_a_split_reversed_{}/'.format(sys.argv[1])
os.makedirs(rev_dir, exist_ok=True)
os.system("rm {}*".format(rev_dir))
out_dir = os.environ["OUTPUT_DIR"] + 'indelphi_b_alignment_{}/'.format(sys.argv[1])
os.makedirs(out_dir, exist_ok=True)
os.system("rm {}*".format(out_dir))

def create_tmp_oligo_file(lib):
    target_f = lib_dir + lib + ".fasta"
    tmpf = out_dir + lib + ".tmp.fasta"
    with open(target_f, 'r') as f:
        with open(tmpf, 'w') as w:
            while True:
                l = f.readline()
                if not l: break
                p = l.split()
                nm = p[0]
                val = eval(p[1])
                d = p[2]
                w.write("{} {} {}\n".format(nm, val, d))
                w.write(f.readline())
    return tmpf

def reverse_fastq(fname):
    fastq = inp_dir + fname
    reverse_f = rev_dir + fname
    with open(fastq, 'r') as f:
        with open(reverse_f, 'w') as w:
            while True:
                l = f.readline()
                if not l: break
                w.write(l)
                w.write(str(Seq(f.readline().strip()).reverse_complement() + "\n"))
                w.write(f.readline())
                w.write(f.readline().strip()[::-1] + "\n")
    return reverse_f


if __name__ == "__main__":
    files = [f for f in os.listdir(inp_dir) if "_2_" in f]
    for fname in tqdm(files):
        lib = "LibA"
        sample_name = fname.split("_2_")[0]
        ext = "_" + fname.split("_2_")[-1].replace("fastq", "mapping")

        oname = out_dir + sample_name + ext
        fastq = reverse_fastq(fname)
        target_f = create_tmp_oligo_file(lib)
        cmd = "sbatch b_alignment.batch {} {} {}".format(fastq, target_f, oname)

        if len(sys.argv) > 2 and sys.argv[-1] == "dry":
            print(cmd)
        else:
            os.system(cmd)
