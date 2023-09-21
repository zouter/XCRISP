import sys, os
import pandas as pd
from Bio.Seq import Seq

sys.path.append("../modelling")
from data_loader import load_Tijsterman_data, read_target_sequence_from_file, get_guides_from_fasta

HPC_DATA_DIR = "/scratch/cfseale/data/LUMC/"
DATA_DIR = os.environ["DATA_DIR"]
LOCAL_DATA_DIR = "../data/LUMC/"
OUTPUT_FILE = LOCAL_DATA_DIR + "{}.txt"

TOTAL_SEQUENCE = 159
LEFT_FLANK_FORWARD = 82
LEFT_FLANK_REVERSE = TOTAL_SEQUENCE - LEFT_FLANK_FORWARD

def fasta(x):
    return ">{} {} {}\n{}\n".format(x["ID"], x["PAM Index"], x["Strand"], x["TargetSequence"])

def write_oligo_file(guides, t):
    filename = "../data/LUMC/{}.fasta".format(t)
    with open(filename, "w") as f:
        used_sequences = set()
        for x in guides.iterrows():
            x = x[1]

            if x["alias"] == "polq-1_p333_m-cherry_pos_h_PCR_1":
                print("here")

            sequence = x["reference"]
            right_flank = x["right flank"].upper()
            # left flank = x[5]
            pam_index = sequence.find(right_flank) + 3
            strand = "FORWARD"
            left = LEFT_FLANK_FORWARD
            if check_for_pam(right_flank):
                strand = "REVERSE"   
                pam_index = sequence.find(right_flank) - 3
                left = LEFT_FLANK_REVERSE
            right = TOTAL_SEQUENCE - left
            sequence = sequence[pam_index-left:pam_index+right]
            # sometimes we have duplicated sequence for which we have analysed data. Take one
            if sequence not in used_sequences:
                details = {"ID": x["alias"].replace(" ", ""), "TargetSequence": sequence, "PAM Index": left, "Strand": strand}
                f.write(fasta(details))
                used_sequences.add(sequence)
    return filename

def join_flanks(x):
    return x["left flank"] + x["right flank"]

def fix_path(p):
    name = p.split("\\")[-1]
    return HPC_DATA_DIR + name

def check_for_pam(right_flank):
    return right_flank[4:6].upper() != "GG"

def reverse_complement(x):
    return str(Seq(x).reverse_complement()).upper()

def read_reference_file(f):
    n = f.split("\\")[-1]
    seq = ""
    with open(DATA_DIR + "LUMC/{}".format(n), "r") as r:
        lines = r.readlines()
        for l in lines[1:]:
            seq += l.strip()
    return seq.upper()

all_targets = pd.read_excel(LOCAL_DATA_DIR + "LUMC.xlsx", sheet_name="RunInfo", engine='openpyxl')

for d in ["POLQ", "WT", "LIG4", "KU80"]:
    # remove rows that have the same target sequence
    targets = all_targets.loc[all_targets.genotype.str.upper() == d, :]
    # targets = all_targets.iloc[31:32, :]
    targets["left_and_right_flank"] = targets.apply(lambda  x: join_flanks(x), axis=1)
    targets = targets.loc[~targets.left_and_right_flank.duplicated(),:]
    targets["reference"] = targets["reference"].apply(lambda x: read_reference_file(x))
    write_oligo_file(targets, d)

    df = []
    for row in targets.iterrows():
        target = {}
        row = row[1]
        
        target["is_reverse"] = check_for_pam(row["right flank"])
        target["R1 file"] = fix_path(row["R1 file"])
        target["R2 file"] = fix_path(row["R2 file"])
        target["#bases pastprimer"] = row["#bases pastprimer"]
        target["alias"] = row["alias"].replace(" ", "")

        if target["is_reverse"]:
            target["left flank"] = reverse_complement(row["right flank"])
            target["right flank"] = reverse_complement(row["left flank"])
            target["reference"] = reverse_complement(row["reference"])
            target["left_primer"] = reverse_complement(row["right primer"])
            target["right_primer"] = reverse_complement(row["left primer"])
        else:
            target["left flank"] = row["left flank"]
            target["right flank"] = row["right flank"]
            target["reference"] = row["reference"]
            target["left_primer"] = row["left primer"]
            target["right_primer"] = row["right primer"]
        df.append(target)

    
    df = pd.DataFrame(df)
    df = df[["R1 file", "R2 file", "alias", "reference", "left flank", "right flank", "left_primer", "right_primer", "#bases pastprimer", "is_reverse"]]
    df.to_csv(OUTPUT_FILE.format(d), index=False, header=False, sep="\t")
    write_oligo_file(df, d + ".forward")
print("Done.")
