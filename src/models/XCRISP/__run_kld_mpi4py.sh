
for LEARNING_RATE in 0.1 0.05 0.01 0.001
   do for LOSS in kld mse 
        do
            sbatch train_mpi4py.batch $LEARNING_RATE $LOSS
        done
   done




