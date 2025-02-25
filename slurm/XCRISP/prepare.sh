# FORECasT data
sbatch $PROTONDDR/repos/x-crisp/slurm/XCRISP/prepare.batch train
sbatch $PROTONDDR/repos/x-crisp/slurm/XCRISP/prepare.batch test

sbatch $PROTONDDR/repos/x-crisp/slurm/XCRISP/prepare.batch HAP1_train
sbatch $PROTONDDR/repos/x-crisp/slurm/XCRISP/prepare.batch HAP1_test

sbatch $PROTONDDR/repos/x-crisp/slurm/XCRISP/prepare.batch TREX_A_test
sbatch $PROTONDDR/repos/x-crisp/slurm/XCRISP/prepare.batch TREX_A_train


# inDelphi Data
sbatch $PROTONDDR/repos/x-crisp/slurm/XCRISP/prepare.batch 0105-mESC-Lib1-Cas9-Tol2-BioRep2-techrep1

sbatch $PROTONDDR/repos/x-crisp/slurm/XCRISP/prepare.batch 0226-PRLmESC-Lib1-Cas9_transfertrain
sbatch $PROTONDDR/repos/x-crisp/slurm/XCRISP/prepare.batch 0226-PRLmESC-Lib1-Cas9_transfertest

sbatch $PROTONDDR/repos/x-crisp/slurm/XCRISP/prepare.batch 052218-U2OS-+-LibA-postCas9-rep1_transfertrain
sbatch $PROTONDDR/repos/x-crisp/slurm/XCRISP/prepare.batch 052218-U2OS-+-LibA-postCas9-rep1_transfertest
