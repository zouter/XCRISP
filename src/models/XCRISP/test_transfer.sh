# for MODE in pretrainedplusonelayer pretrainedonefrozenlayer weightinit pretrained baseline pretrainedsamearch 
#     do for NUM_SAMPLES in 2 5 10 20 50 100 200 500 1000  
#         do sbatch test_transfer.batch $MODE $NUM_SAMPLES
#     done
# done

for MODE in baseline
    do for NUM_SAMPLES in 2 5 10 20 50 100 200 500 1000
        do sbatch test_transfer.batch $MODE $NUM_SAMPLES
    done
done
