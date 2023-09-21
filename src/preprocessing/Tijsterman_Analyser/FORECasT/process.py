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
import pandas as pd
import numpy as np

from dummy import DUMMY_LEFT_PRIMER, DUMMY_RIGHT_PRIMER

from datetime import datetime
if len(sys.argv) == 1:
    print("Provide the date of the last run of splitting the data that you want to align in the format YYYY-MM-DD")

sys.path.append("../../../data/FORECasT")
from utils import get_guides

CELL_TYPE = "2A_TREX_A"
OUTPUT_DIR = os.environ["OUTPUT_DIR"]
ORIG_FASTQ_FILE = OUTPUT_DIR + "/mapped_reads_{0}/{1}/{2}/{2}.fastq"
DUMMY_FILE = OUTPUT_DIR + "/mapped_reads_{0}/{1}/{2}/{2}.dummy.fastq"

WDL = {
  "NGS_PCR.outputFile": "output.txt"
}

# Step 1. Get Test Sequences
samples = get_guides()
samples["inFile"] = samples.ID.apply(lambda x: ORIG_FASTQ_FILE.format(sys.argv[1], CELL_TYPE, x))
samples["dummyFile"] = samples.ID.apply(lambda x: DUMMY_FILE.format(sys.argv[1], CELL_TYPE, x))
samples["subject"] = samples.TargetSequence.apply(lambda x: DUMMY_LEFT_PRIMER + x + DUMMY_RIGHT_PRIMER)
samples["cutsite"] = samples["PAM Index"] - 3
samples["left"] = samples.apply(lambda x: x.TargetSequence[x.cutsite-15:x.cutsite], axis=1)
samples["right"] = samples.apply(lambda x: x.TargetSequence[x.cutsite:x.cutsite+15], axis=1)
samples["leftPrimer"] = DUMMY_LEFT_PRIMER
samples["rightPrimer"] = DUMMY_RIGHT_PRIMER
samples["alias"] = samples.ID
samples["outputFile"] = samples.ID

samples = samples[["inFile", "dummyFile", "subject", "left", "right", "leftPrimer", "rightPrimer","outputFile", "alias"]]

# Step 2. Split into multiple frames for parrallel processing
batches = np.array_split(samples, 500)
os.makedirs("./batches/", exist_ok=True)
for i, b in enumerate(batches):
  batch_input_file = "./batches/FORECasT_{}.txt".format(i)
  b.to_csv(batch_input_file, sep="\t", index=False, header=False)

  batch_json = WDL.copy()
  batch_json["NGS_PCR.samples"] = batch_input_file
  batch_json_file = batch_input_file.replace("txt", "json")
  with open(batch_json_file, 'w') as outfile:
      json.dump(batch_json, outfile)

  cmd = "sbatch process.batch {}".format(batch_json_file)
  if sys.argv[-1] == "dry":
    datetime.strptime(sys.argv[1], '%Y-%M-%d')
    print(cmd)
  else:
    os.system(cmd)
print("Done.")

