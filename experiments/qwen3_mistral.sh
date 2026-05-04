#!/usr/bin/env bash

# Qwen3-8B ASVD
CUDA_VISIBLE_DEVICES='0' python asvd.py \
  --model_id="Qwen/Qwen3-8B" \
  --act_aware \
  --alpha 0.5 \
  --n_calib_samples 32 \
  --calib_dataset wikitext2_evol-codealpaca_tulu-math \
  --scaling_method abs_mean \
  --param_ratio_target 0.9 \
  --use_cache \
  --use_bos \
  --save_model output/Qwen3-8B-ASVD \
  --skip_eval

# Mistral-7B ASVD
CUDA_VISIBLE_DEVICES='1' python asvd.py \
  --model_id="mistralai/Mistral-7B-v0.1" \
  --act_aware \
  --alpha 0.5 \
  --n_calib_samples 32 \
  --calib_dataset wikitext2_evol-codealpaca_tulu-math \
  --scaling_method abs_mean \
  --param_ratio_target 0.9 \
  --use_cache \
  --use_bos \
  --save_model output/Mistral-7B-ASVD \
  --skip_eval
