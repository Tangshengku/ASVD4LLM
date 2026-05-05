export TRANSFORMERS_CACHE=/nfs/scistore19/alistgrp/huggingface/hub

# Qwen3-8B ASVD
CUDA_VISIBLE_DEVICES='0' python asvd.py \
  --model_id="Qwen/Qwen3-8B" \
  --act_aware \
  --alpha 0.5 \
  --n_calib_samples 32 \
  --ppl_eval_batch_size 8 \
  --calib_dataset wikitext2_evol-codealpaca_tulu-math \
  --scaling_method abs_mean \
  --param_ratio_target 0.8 \
  --use_cache \
  --use_bos \
  --save_model output/Qwen3-8B-ASVD \
  --eval_ppl wikitext2 \
  --eval_ppl_fraction 0.02 \
  --eval_batch_size 1

# Mistral-7B ASVD
CUDA_VISIBLE_DEVICES='0' python asvd.py \
  --model_id="mistralai/Mistral-7B-v0.1" \
  --act_aware \
  --alpha 0.5 \
  --n_calib_samples 32 \
  --ppl_eval_batch_size 8 \
  --calib_dataset wikitext2_evol-codealpaca_tulu-math \
  --scaling_method abs_mean \
  --param_ratio_target 0.9 \
  --use_cache \
  --use_bos \
  --save_model output/Mistral-7B-ASVD \
  --eval_ppl wikitext2 \
  --eval_ppl_fraction 0.02 \
  --eval_batch_size 1
