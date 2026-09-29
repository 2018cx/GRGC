WARM_EXP=lmsys-gpt-qwen3b
ADV_EXP=lmsys-gpt-qwen3b
export PYTHONPATH=/tmp/verl_ours_warmup
bash scripts/train/gpt5-chat-filtered-7b-warmup-lr1e-6.sh \
  --model /tmp/models/Qwen2.5-3B-Instruct \
  --reward_model /tmp/models/Qwen2.5-3B-Instruct \
  --exp_name ${WARM_EXP} \
  --nnodes 1
export PYTHONPATH=/tmp/verl_ours
bash scripts/train/gpt5-chat-filtered-7b-adversarial-lr1e-6.sh \
  --exp_name ${ADV_EXP} \
  --resume_step 1500 \
  --nnodes 1
