# IMPORTANT: prepare dataset before running training

# run all experiments
# python3 transfer.py gen_oligos train

for MODE in pretrained baseline pretrainedsamearch pretrainedplusonelayer pretrainedonefrozenlayer weightinit
    do for NUM_SAMPLES in 2 5 10 20 50 100 200 500 
        do 
        sbatch transfer.batch $MODE $NUM_SAMPLES 0226-PRLmESC-Lib1-Cas9_transfertrain mESC-NHEJ-deficient
        sbatch transfer.batch $MODE $NUM_SAMPLES 052218-U2OS-+-LibA-postCas9-rep1_transfertrain U2OS
        sbatch transfer.batch $MODE $NUM_SAMPLES HAP1_train HAP1
        sbatch transfer.batch $MODE $NUM_SAMPLES TREX_A_train TREX_A
    done
done

# for MODE in baseline
#    do for NUM_SAMPLES in 2 5 10 20 50 100 200 500 1000 max
#        do 
#        sbatch transfer.batch $MODE $NUM_SAMPLES train train
#    done
# done

# python3 transfer.py gen_oligos 0226-PRLmESC-Lib1-Cas9_transfertrain 
# python3 transfer.py gen_oligos 052218-U2OS-+-LibA-postCas9-rep1_transfertrain 
# python3 transfer.py gen_oligos HAP1_train 
# python3 transfer.py gen_oligos TREX_A_train 
# python3 transfer.py gen_oligos train  
