ACCEL_CONFIG="configs/accelerate_config_multi_gpu.yaml" # or accelerate_config_single_gpu.yaml
MODEL="XLSR_Conformer"
VERSION="XLSR_Conformer_CL_DT" # select {XLSR_Conformer, XLSR_Conformer_CL, XLSR_Conformer_CL_DT}
TRACK="DF" # select {DF, LA}
USE_VARIABLE="False" # select {False, True}

if [ "${USE_VARIABLE}" = "True" ]; then
    echo "Using variable length"
    poetry run \
        accelerate launch --config_file ${ACCEL_CONFIG} \
        models/${MODEL}/evaluate.py --config logs/${TRACK}/${VERSION}/config.yaml --track ${TRACK} --variable True
else
    echo "Using fixed length"
    poetry run \
        accelerate launch --config_file ${ACCEL_CONFIG} \
        models/${MODEL}/evaluate.py --config logs/${TRACK}/${VERSION}/config.yaml --track ${TRACK}
fi

if [ "${USE_VARIABLE}" = "True" ]; then
    poetry run python common/eval/main.py \
        --cm-score-file logs/${TRACK}/${VERSION}/Scores_VL/${TRACK}/scores.txt \
        --track ${TRACK} --subset eval
else
    poetry run python common/eval/main.py \
        --cm-score-file logs/${TRACK}/${VERSION}/Scores/${TRACK}/scores.txt \
        --track ${TRACK} --subset eval
fi
