for GENOTYPE in 0226-PRLmESC-Lib1-Cas9_transfertrain 052218-U2OS-+-LibA-postCas9-rep1_transfertrain HAP1_train TREX_A_train
do 
    # sbatch LR_insertion.batch $GENOTYPE baseline
    sbatch LR_insertion.batch $GENOTYPE pretrained 0.001
    sbatch LR_insertion.batch $GENOTYPE pretrained 0.0001
done
