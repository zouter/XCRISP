apptainer shell -C --nv \
-H $PROTONDDR/repos/x-crisp/ \
--env OUTPUT_DIR=$OUTPUT_DIR \
--env LOGS_DIR=$LOGS_DIR \
-B $OUTPUT_DIR:$OUTPUT_DIR,$LOGS_DIR:$LOGS_DIR \
containers/lab.sif \
