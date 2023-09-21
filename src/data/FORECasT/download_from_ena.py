# rewrite this file to use wget, it is much easier (but slower)

import os, sys
import pandas as pd
import multiprocessing
import platform

sys.path.append("../../../common")
from data import find_ERR_for_ERS, load_sample_accession_ERS, get_aspera_file, get_data_folder, get_output_folder
from utils import mkdir_p

usage = """Usage: download_from_ena.py path/to/aspera_download_file.txt path/to/sample_accessions_file.txt path/to/output_folder

Example:
download_from_ena.py \
/Users/colm/repos/repair-outcome-prediction/indel_analysis/data/PRJEB29746/aspera/PRJEB29746.txt \
/Users/colm/repos/repair-outcome-prediction/indel_analysis/data/PRJEB29746/sample_accessions.txt \
/Users/colm/repos/data  

The download_file.txt contains the following columns from ENA:
- study_accession	
- sample_accession	
- experiment_accession	
- run_accession	
- scientific_name	
- fastq_aspera
"""

def write_commands_to_file(cmds):
    output_dir = get_output_folder() + "/aspera"
    mkdir_p(output_dir)
    job_file = os.path.join(os.path.expanduser(output_dir), "aspera_downloads.sh")
    with open(job_file, 'w+') as fh:
        for c in cmds:
            fh.writelines("{}\n".format(c))

def run_command(cmd):
    if len(sys.argv) > 1 and sys.argv[1] == "dry":
        print(cmd)
    else:
        os.system(cmd)

def get_aspera_exec():
    if platform.system() == "Darwin":
        return "~/Applications/Aspera\ CLI"
    else:
        return "~/.aspera/cli"

if __name__ == '__main__':
    # if len(sys.argv) != 4:
    #    sys.exit(usage)

    aspera = get_aspera_exec()

    download_file = get_aspera_file()
    data_folder = get_data_folder("forecast")
    mkdir_p(data_folder)

    print("Downloading files from " + download_file)

    df = pd.read_csv(download_file, sep="\t")
    aspera_links = []

    # mESC_samples = find_ERR_for_ERS(load_sample_accession_ERS("mESC"))
    # NULL_samples = find_ERR_for_ERS(load_sample_accession_ERS("NULL"))
    # samples = mESC_samples + NULL_samples

    samples = find_ERR_for_ERS(load_sample_accession_ERS("2A_TREX_A"))

    for a in list(df[df["run_accession"].isin(samples)]["fastq_aspera"]):
        R = a.split(";")
        aspera_links.append(R[0])
        aspera_links.append(R[1])

    cmds = []
    for link in aspera_links:
        cmds.append('{}/bin/ascp -QT -l 300m -P33001 -i {}/etc/asperaweb_id_dsa.openssh era-fasp@{} {}'.format(aspera, aspera, link, data_folder))
    write_commands_to_file(cmds)


    num_tasks = len(cmds)
    pool = multiprocessing.Pool(processes=4)
    for i, _ in enumerate(pool.imap_unordered(run_command, cmds), 1):
        sys.stderr.write('\rdone {0:%}'.format(i/num_tasks))
