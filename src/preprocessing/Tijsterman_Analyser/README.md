# Indel Analysis

### Requirements
- Cromwell v53 -  https://github.com/broadinstitute/cromwell/releases/tag/53
- Java 
- FLASH2 - https://github.com/dstreett/FLASH2
- SLURM
- Tijsterman Analyzer

### Analysis
 - Run basic analysis from git@gitlab.ewi.tudelft.nl:cfseale/selftarget.git 
    ```
    cd indel_anlayis/compute_indels 
    python3 run_example.py
    ```
 - Run the outputted csv files through Tijsterman analysis pipeline
    ```
    java -jar  -Dconfig.file=sge.conf cromwell-53.jar run NGS_PCR.wdl -i NGS_PCR_104091_MB.wdl.json
    ```
