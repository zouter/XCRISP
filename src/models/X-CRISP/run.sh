# IMPORTANT: prepare dataset before running training

# run all experiments

# do for EXPERIMENT in full full+numrepeat inDelphi_features v2 v2+ranked
    # do for LOSS in Base BinsOnly Combined
for SEED in 1 
do for EXPERIMENT in v4
    do for LOSS in Base KL_Div 
        do
            sbatch train.batch model $EXPERIMENT $LOSS $SEED
        done
    done
done

# 2NN baselines
# sbatch train.batch indelphi full
# sbatch train.batch indelphi full+numrepeat
# sbatch train.batch indelphi inDelphi_features


