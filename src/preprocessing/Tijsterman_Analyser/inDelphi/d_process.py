# expected format
# 
# inFile = sample[0],
# subject = sample[1],
# left = sample[2],
# right = sample[3],
# outputFile = sample[4],
# alias = sample[5],

import os
import sys
import csv
import json
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.append("../../../modelling/")
from data_loader import get_details_from_fasta

if len(sys.argv) == 1:
    print("Provide the date of the last run of splitting the data that you want to align in the format YYYY-MM-DD")

datetime.strptime(sys.argv[1], '%Y-%M-%d')

FASTQ_FILE = os.environ["OUTPUT_DIR"] + 'processed_data/Tijsterman_Analyser/inDelphi/c_mapping_{0}/{1}/{2}/{2}.fastq'

WDL = {
  "NGS_PCR.outputFile": "output.txt"
}

LEFT_PRIMER = "GATGGGTGCGACGCGTCAT"
RIGHT_PRIMER = "AGATCGGAAGAGCACACGTCTGAATATTGTGGA"

if __name__ == "__main__":
    # for s in ["1027-mESC-Lib1-Cas9-Tol2-Biorep1-techrep1"]:
    for s in ["052218-U2OS-+-LibA-postCas9-rep2"]:
        # Step 1. Get Test Sequences
        samples = get_details_from_fasta("../../../data/inDelphi/LibA.fasta")
        samples = pd.DataFrame.from_dict(samples, orient='index')

        samples["inFile"] = samples["ID"].apply(lambda x: FASTQ_FILE.format(sys.argv[1], s, x))
        samples["subject"] = samples["TargetSequence"].apply(lambda x: LEFT_PRIMER + x + RIGHT_PRIMER)
        samples["cutsite"] = samples["PAM Index"] - 3
        samples["left"] = samples.apply(lambda x: x["TargetSequence"][x.cutsite-25:x.cutsite], axis=1)
        samples["right"] = samples.apply(lambda x: x["TargetSequence"][x.cutsite:x.cutsite+25], axis=1)
        samples["leftPrimer"] = LEFT_PRIMER
        samples["rightPrimer"] = RIGHT_PRIMER
        samples["alias"] = samples["ID"]
        samples["outputFile"] = s + "_" + samples["ID"] + "_indels"

        samples = samples[["inFile", "subject", "left", "right", "leftPrimer", "rightPrimer","outputFile", "alias"]]

        # Step 2. Split into multiple frames for parrallel processing
        batches = np.array_split(samples, 20)
        for i, b in enumerate(batches):
            batch_input_file = "./batches/inDelphi_{}_{}.txt".format(s, i)
            b.to_csv(batch_input_file, sep="\t", index=False, header=False)

            batch_json = WDL.copy()
            batch_json["NGS_PCR.samples"] = batch_input_file
            batch_json_file = batch_input_file.replace("txt", "json")
            with open(batch_json_file, 'w') as outfile:
                json.dump(batch_json, outfile)

            cmd = "sbatch d_process.batch {}".format(batch_json_file)
            if sys.argv[-1] == "dry":
                print(cmd)
            else:
                os.system(cmd)
        print("Done.")
