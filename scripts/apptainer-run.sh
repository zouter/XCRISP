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
    "alignment_scores")
        echo "Running alignment scores"
        apptainer exec --nv -C  \
        -H $PROTONDDR/repos/x-crisp/ \
        --env PROTONDDR=$PROTONDDR \
        -B $OUTPUT_DIR:$OUTPUT_DIR,$LOGS_DIR:$LOGS_DIR \
        containers/lab.sif \
        mpiexec -n $2 /home/dsbpredict/miniconda3/envs/xcrisp/bin/python3 -m src.analysis.calculate_homology $3
        ;;
    "prepare")
        echo "Preparing data for $3"
        apptainer exec --nv -C  \
        -H $PROTONDDR/repos/x-crisp/ \
        --env OUTPUT_DIR=$OUTPUT_DIR \
        -B $OUTPUT_DIR:$OUTPUT_DIR \
        containers/lab.sif \
        mpiexec -n $2 /home/dsbpredict/miniconda3/envs/xcrisp/bin/python3 -m src.models.$3.prepare $4
        ;;
    "prepare_lindel")
        echo "Preparing data for Lindel"
        apptainer exec --nv -C  \
        -H $PROTONDDR/repos/x-crisp/ \
        --env OUTPUT_DIR=$OUTPUT_DIR \
        -B $OUTPUT_DIR:$OUTPUT_DIR \
        containers/lab.sif \
        /home/dsbpredict/miniconda3/envs/xcrisp/bin/python3 -m src.models.Lindel.prepare
        ;;
    "deletion")
        echo "Training X-CRISP deletion model"
        apptainer exec --nv -C  \
        -H $PROTONDDR/repos/x-crisp/ \
        --env OUTPUT_DIR=$OUTPUT_DIR \
        --env LOGS_DIR=$LOGS_DIR \
        -B $OUTPUT_DIR:$OUTPUT_DIR,$LOGS_DIR:$LOGS_DIR \
        containers/lab.sif \
        mpiexec -n $2 /home/dsbpredict/miniconda3/envs/xcrisp/bin/python3 -m src.models.XCRISP.deletion $3 $4
        ;;
    "insertion")
        echo "Training Lindel insertion model"
        apptainer exec --nv -C  \
        -H $PROTONDDR/repos/x-crisp/ \
        --env OUTPUT_DIR=$OUTPUT_DIR \
        --env LOGS_DIR=$LOGS_DIR \
        -B $OUTPUT_DIR:$OUTPUT_DIR,$LOGS_DIR:$LOGS_DIR \
        containers/lab.sif \
        /home/dsbpredict/miniconda3/envs/xcrisp/bin/python3 -m src.models.Lindel.LR_insertion train baseline 
        ;;
    "indel")
        echo "Training Lindel indel model"
        apptainer exec --nv -C  \
        -H $PROTONDDR/repos/x-crisp/ \
        --env OUTPUT_DIR=$OUTPUT_DIR \
        --env LOGS_DIR=$LOGS_DIR \
        -B $OUTPUT_DIR:$OUTPUT_DIR,$LOGS_DIR:$LOGS_DIR \
        containers/lab.sif \
        /home/dsbpredict/miniconda3/envs/xcrisp/bin/python3 -m src.models.Lindel.LR_indel train baseline 
        ;;
    "shap")
        echo "Running Shap analysis on XCRISP deletion model for $3 deletions"
        apptainer exec --nv -C  \
        -H $PROTONDDR/repos/x-crisp/ \
        --env OUTPUT_DIR=$OUTPUT_DIR \
        --env LOGS_DIR=$LOGS_DIR \
        -B $OUTPUT_DIR:$OUTPUT_DIR,$LOGS_DIR:$LOGS_DIR \
        containers/lab.sif \
        mpiexec -n $2 /home/dsbpredict/miniconda3/envs/xcrisp/bin/python3 -m src.models.XCRISP.interpretability $3 $4
        ;;
    "test")
        echo "Training XCRISP test"
        apptainer exec --nv -C  \
        -H $PROTONDDR/repos/x-crisp/ \
        --env OUTPUT_DIR=$OUTPUT_DIR \
        --env LOGS_DIR=$LOGS_DIR \
        -B $OUTPUT_DIR:$OUTPUT_DIR,$LOGS_DIR:$LOGS_DIR \
        containers/lab.sif \
        /home/dsbpredict/miniconda3/envs/xcrisp/bin/python3 -m src.models.XCRISP.test $2 $3
        ;;
    "test_lindel")
        echo "Running tests on Lindel"
        apptainer exec --nv -C  \
        -H $PROTONDDR/repos/x-crisp/ \
        --env OUTPUT_DIR=$OUTPUT_DIR \
        --env LOGS_DIR=$LOGS_DIR \
        -B $OUTPUT_DIR:$OUTPUT_DIR,$LOGS_DIR:$LOGS_DIR \
        containers/lab.sif \
        /home/dsbpredict/miniconda3/envs/xcrisp/bin/python3 -m src.models.Lindel.test
        ;;
    "shap_analysis_lindel")
        echo "Running Shap analysis on Lindel $3 model"
        apptainer exec --nv -C  \
        -H $PROTONDDR/repos/x-crisp/ \
        --env OUTPUT_DIR=$OUTPUT_DIR \
        --env LOGS_DIR=$LOGS_DIR \
        -B $OUTPUT_DIR:$OUTPUT_DIR,$LOGS_DIR:$LOGS_DIR \
        containers/lab.sif \
        /home/dsbpredict/miniconda3/envs/xcrisp/bin/python3 -m src.models.Lindel.interpretability $3 
        ;;
    "transfer")
        echo "Running transfer learning training"
        if [[ "$2" == *hpc* ]]; then
            echo "Environment variable contains 'hpc'"
            apptainer exec --nv -C  \
            -H $PROTONDDR/repos/x-crisp/ \
            --env OUTPUT_DIR=$OUTPUT_DIR \
            -B $OUTPUT_DIR:$OUTPUT_DIR \
            containers/lab.sif \
            /home/dsbpredict/miniconda3/envs/xcrisp/bin/python3 -m src.models.XCRISP.transfer $3 $4 $5 $6 
        else
            echo "Environment variable does not contain 'hpc'"
            apptainer exec -C \
            -H $PROTONDDR/repos/x-crisp/ \
            --env OUTPUT_DIR=$OUTPUT_DIR \
            -B $OUTPUT_DIR:$OUTPUT_DIR \
            containers/lab.sif \
            /home/dsbpredict/miniconda3/envs/xcrisp/bin/python3 -m src.models.XCRISP.transfer pretrained 5 HAP1_train HAP1
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
            /home/dsbpredict/miniconda3/envs/xcrisp/bin/python3 -m src.models.XCRISP.test_transfer $3 $4 
        else
            echo "Environment variable does not contain 'hpc'"
            apptainer exec -C \
            -H $PROTONDDR/repos/x-crisp/ \
            --env OUTPUT_DIR=$OUTPUT_DIR \
            -B $OUTPUT_DIR:$OUTPUT_DIR \
            containers/lab.sif \
            /home/dsbpredict/miniconda3/envs/xcrisp/bin/python3 -m src.models.XCRISP.test_transfer pretrained 5 
        fi
        ;;
    *)
        echo "Please run one of the following commands: alignment_scores, transfer, test_transfer, deletion, prepare, prepare_lindel, insertion, indel"
        ;;
esac


