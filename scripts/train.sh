
ACCEL_CONFIG="configs/accelerate_config_single_gpu.yaml" # only single GPU for reproducibility
VERSION="XLSR_Conformer_CL_DT" # select {XLSR_Conformer, XLSR_Conformer_CL, XLSR_Conformer_CL_DT}
TRACK="DF" # select {DF, LA}

poetry run \
    accelerate launch --config_file ${ACCEL_CONFIG} \
    models/${VERSION}/train.py --config configs/${VERSION}/base_${TRACK}.yaml

