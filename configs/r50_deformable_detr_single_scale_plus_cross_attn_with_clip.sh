#!/usr/bin/env bash

set -x

EXP_DIR=exps/r50_deformable_detr_single_scale_plus_cross_attn_with_clip_plus_diffusers
PY_ARGS=${@:1}


python -u main.py \
    --num_feature_levels 1 \
    --output_dir ${EXP_DIR} \
    --insert_cross_attn \
    ${PY_ARGS}
