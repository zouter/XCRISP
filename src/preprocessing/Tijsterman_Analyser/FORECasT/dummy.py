import sys


DUMMY_LEFT_PRIMER = "GCTCTTCCGATCTGG"
DUMMY_RIGHT_PRIMER = "CGCTCTTCCGATCTA"
DUMMY_PHRED_LEFT = "AAFFIIIIIIIIIII"
DUMMY_PHRED_RIGHT = "IIIIIIIIIIIFFAA"


if __name__ == "__main__":
    in_file = sys.argv[1]
    outfile = in_file[:-6] + ".dummy" + in_file[-6:]

    with open(in_file, "r") as in_fastq_f:
        with open(outfile, "w") as out_fastq_f:
            while True:
                read_id = in_fastq_f.readline()
                if not read_id:
                    break

                out_fastq_f.write(read_id)

                sequence = in_fastq_f.readline().strip()
                sequence = DUMMY_LEFT_PRIMER + sequence + DUMMY_RIGHT_PRIMER + "\n"
                out_fastq_f.write(sequence)

                out_fastq_f.write(in_fastq_f.readline())

                phred = in_fastq_f.readline().strip()
                phred = DUMMY_PHRED_LEFT + phred + DUMMY_PHRED_RIGHT + "\n"
                out_fastq_f.write(phred)
    print("Dummy file created: ", outfile)
    