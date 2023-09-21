for MODE in pretrained baseline pretrainedsamearch pretrainedplusonelayer pretrainedonefrozenlayer weightinit
    do for NUM_SAMPLES in 2 5 10 20 50 100 200 500 
        do 
        sbatch ./slurm/1.transfer.batch $MODE $NUM_SAMPLES 0226-PRLmESC-Lib1-Cas9_transfertrain mESC-NHEJ-deficient
        sbatch ./slurm/1.transfer.batch $MODE $NUM_SAMPLES 052218-U2OS-+-LibA-postCas9-rep1_transfertrain U2OS
        sbatch ./slurm/1.transfer.batch $MODE $NUM_SAMPLES HAP1_train HAP1
        sbatch ./slurm/1.transfer.batch $MODE $NUM_SAMPLES TREX_A_train TREX_A
    done
done