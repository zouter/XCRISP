import os, sys
import pandas as pd
import multiprocessing
import platform

usage = """Usage: download_from_ena.py path/to/aspera_download_file.txt path/to/output_folder

Example:
download_from_ena.py \
/Users/colm/repos/repair-outcome-prediction/indel_analysis/data/PRJEB29746/aspera/PRJEB29746.txt \
/Users/colm/repos/data  

The download_file.txt contains the following columns from ENA:
- Run Accession   
- Sample Accession    
- FASTQ Aspera    
- Sample Alias
"""

def run_command(cmd):
    print(cmd)
    os.system(cmd)

def get_aspera_exec():
    if platform.system() == "Darwin":
        return "~/Applications/Aspera\ CLI"
    else:
        return "~/.aspera/cli"

if __name__ == '__main__':
    aspera = get_aspera_exec()
    download_file = "samples.tsv"
    data_folder = os.environ["DATA_DIR"] + "inDelphi/"
    os.makedirs(data_folder, exist_ok=True)

    print("Downloading files from " + download_file)

    df = pd.read_csv(download_file, sep="\t")
    aspera_links = []

    print("Done.")

    # filter step for new data
    df = df[df["Sample Alias"].str.contains("U2OS")]

    for a in df["FASTQ Aspera"]:
        R = a.split(";")
        aspera_links.append(R[0])
        aspera_links.append(R[1])

    cmds = []
    for link in aspera_links:
        cmd = '{}/bin/ascp -QT -l 300m -P33001 -i {}/etc/asperaweb_id_dsa.openssh era-fasp@{} {}'.format(aspera, aspera, link, data_folder)
        cmds.append(cmd)

    num_tasks = len(cmds)
    pool = multiprocessing.Pool(processes=4)
    for i, _ in enumerate(pool.imap_unordered(run_command, cmds), 1):
        sys.stderr.write('\rdone {0:%}'.format(i/num_tasks))
