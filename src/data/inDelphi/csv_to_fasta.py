import csv
from Bio.Seq import Seq
from sklearn.model_selection import train_test_split

def fasta(x):
    return ">{} {} {}\n{}\n".format(x[0], x[1], x[2], x[3])

def write_oligo_file(guides, t):
    filename = "{}.fasta".format(t)
    with open(filename, "w") as f:
        for x in guides:
            f.write(fasta(x))
    return filename


if __name__ == "__main__":
    for lib in ["LibA"]:
        guides = []
        with open(lib + ".csv", 'r') as f:
            reader = csv.DictReader(f)
            unique_seq = []
            w = open(lib + '.fasta', 'w')
            w_fwd = open(lib + '.forward.fasta', 'w')

            for r in reader:
                name = r["Name"].split(":")[0]
                target = r["Guide"]
                sequence = r["Full target sequence"]
                idx = sequence.find(target)
                forward_idx = idx + 20

                if target not in unique_seq:
                    unique_seq.append(target)
                else:
                    print(target + " is duplicated")
                    continue

                # need to correct for FORWARD and REVERSE sequences
                if (idx != -1) and (sequence[forward_idx+1:forward_idx+3] == "GG"):
                    direction = "FORWARD"
                    w.write(">{}_{} {} {}\n".format(name, target, 30, "FORWARD"))
                    w.write(sequence + "\n")
                    w_fwd.write(">{}_{} {} {}\n".format(name, target, 30, "FORWARD"))
                    w_fwd.write(sequence + "\n")
                # else:
                    # direction = "REVERSE"
                    # w.write(">{}_{} {} {}\n".format(name, target, 25, "REVERSE"))
                    # w.write(sequence + "\n")

                    # fwd_sequence = str(Seq(sequence).reverse_complement())
                    # w_fwd.write(">{}_{} {} {}\n".format(name, target, 30, "FORWARD"))
                    # w_fwd.write(fwd_sequence + "\n")


                guides.append((name + "_" + target, 30, direction, sequence))

            train, test = train_test_split(guides, test_size=1000, random_state=42, shuffle=True)
            write_oligo_file(train, "transfertrain")
            write_oligo_file(test, "transfertest")
            
            w.close()
            w_fwd.close()
            print(len(unique_seq))
