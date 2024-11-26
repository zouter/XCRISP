apptainer shell -C --nv \
-H $PROTONDDR/repos/x-crisp/ \
--env OUTPUT_DIR=$OUTPUT_DIR \
-B $OUTPUT_DIR:$OUTPUT_DIR \
containers/lab.sif \
