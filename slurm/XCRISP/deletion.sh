
for LEARNING_RATE in 0.1 0.05 0.01 0.001;
    do for LOSS in kld mse; 
        do for L2 in 0.01 0.001 0.0001 0.000075 0.00005 0.000025 0.00001 0.000001;
            do
                sbatch $PROTONDDR/repos/x-crisp/slurm/XCRISP/deletion.batch $LEARNING_RATE $LOSS $L2
            done
        done
    done
done




