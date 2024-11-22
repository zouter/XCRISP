for MODE in pretrainedplusonelayer pretrainedonefrozenlayer weightinit pretrained baseline pretrainedsamearch 
    do for NUM_SAMPLES in 2 5 10 20 50 100 200 500   
        do sbatch ./slurm/1b.test_transfer.batch $MODE $NUM_SAMPLES
    done
done
