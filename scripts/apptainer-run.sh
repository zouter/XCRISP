# update to make sure all requirements are installed
# echo "updating dependencies"
# apptainer exec -C  \
# -H $PROTONDDR \
# containers/lab.sif \
# /home/dsbpredict/miniconda3/envs/dnabert2/bin/python3 -m pip install -r $PROTONDDR/repos/dsb-gene-function-prediction/requirements.txt

# execute desired program
# options:
# - transfer

echo "executing program: $1"
case "$1" in
    "transfer")
        echo "Running transfer learning training"
        if [[ "$2" == *hpc* ]]; then
            echo "Environment variable contains 'hpc'"
            apptainer exec --nv -C  \
            -H $PROTONDDR/repos/x-crisp/ \
            --env OUTPUT_DIR=$OUTPUT_DIR \
            -B $OUTPUT_DIR:$OUTPUT_DIR \
            containers/lab.sif \
            /home/dsbpredict/miniconda3/envs/X-CRISP/bin/python3 -m src.models.XCRISP.transfer $3 $4 $5 $6 
        else
            echo "Environment variable does not contain 'hpc'"
            apptainer exec -C \
            -H $PROTONDDR/repos/x-crisp/ \
            --env OUTPUT_DIR=$OUTPUT_DIR \
            -B $OUTPUT_DIR:$OUTPUT_DIR \
            containers/lab.sif \
            /home/dsbpredict/miniconda3/envs/X-CRISP/bin/python3 -m src.models.XCRISP.transfer pretrained 5 HAP1_train HAP1
        fi
        ;;
    "test_transfer")
        echo "Running transfer learning testing"
        if [[ "$2" == *hpc* ]]; then
            echo "Environment variable contains 'hpc'"
            apptainer exec --nv -C  \
            -H $PROTONDDR/repos/x-crisp/ \
            --env OUTPUT_DIR=$OUTPUT_DIR \
            -B $OUTPUT_DIR:$OUTPUT_DIR \
            containers/lab.sif \
            /home/dsbpredict/miniconda3/envs/X-CRISP/bin/python3 -m src.models.XCRISP.test_transfer $3 $4 
        else
            echo "Environment variable does not contain 'hpc'"
            apptainer exec -C \
            -H $PROTONDDR/repos/x-crisp/ \
            --env OUTPUT_DIR=$OUTPUT_DIR \
            -B $OUTPUT_DIR:$OUTPUT_DIR \
            containers/lab.sif \
            /home/dsbpredict/miniconda3/envs/X-CRISP/bin/python3 -m src.models.XCRISP.test_transfer pretrained 5 
        fi
        ;;
    *)
        echo "Please run one of the following commands: transfer, test_transfer"
        ;;
esac


