ACCEL_CONFIG="configs/accelerate_config_multi_gpu.yaml" # or accelerate_config_single_gpu.yaml
MODEL="XLSR_Conformer"
VERSION="XLSR_Conformer_CL_DT" # select {XLSR_Conformer, XLSR_Conformer_CL, XLSR_Conformer_CL_DT}
USE_VARIABLE="False"

if [ "${USE_VARIABLE}" = "True" ]; then
    echo "Using variable length"
    poetry run \
        accelerate launch --config_file ${ACCEL_CONFIG} \
        models/${MODEL}/evaluate_itw.py --config logs/LA/${VERSION}/config.yaml --variable True
else
    echo "Using fixed length"
    poetry run \
        accelerate launch --config_file ${ACCEL_CONFIG} \
        models/${MODEL}/evaluate_itw.py --config logs/LA/${VERSION}/config.yaml
fi

if [ "${USE_VARIABLE}" = "True" ]; then
    poetry run python common/eval/evaluate_in_the_wild.py \
        logs/LA/${VERSION}/Scores_VL_ITW/scores.txt \
        datasets/keys/ITW/meta.csv
else
    poetry run python common/eval/evaluate_in_the_wild.py \
        logs/LA/${VERSION}/Scores_ITW/scores.txt \
        datasets/keys/ITW/meta.csv
fi
