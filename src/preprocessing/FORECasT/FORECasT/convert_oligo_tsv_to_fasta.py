import os, sys
import pandas as pd 

oligo_tsv_file = "../../../data/FORECasT/targets.txt"
oligo_fasta_file = "{}/exp_target_pam.fasta".format(os.environ["OUTPUT_DIR"])

tsv = pd.read_csv(oligo_tsv_file, sep="\t")

with open(oligo_fasta_file, "w+") as f:
    for index, row in tsv.iterrows():
        oligo_id = row["ID"]
        guide_sequence = row["Guide Sequence"]
        target_sequence = row["TargetSequence"]
        pam_index = row["PAM Index"]
        strand = row["Strand"]
        f.writelines(">%s_%s %s %s\n" % (oligo_id, guide_sequence, pam_index, strand))
        f.writelines("%s\n" % target_sequence)

