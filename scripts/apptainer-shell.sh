apptainer shell -C --nv \
-H $PROTONDDR/repos/x-crisp/ \
--env PROTONDDR=$PROTONDDR \
-B $PROTONDDR:$PROTONDDR \
containers/lab.sif \
