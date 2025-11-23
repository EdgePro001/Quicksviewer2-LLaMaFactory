conda activate llamaf

# 设置WandB环境变量
export WANDB_PROJECT=glm4v-cubing
export WANDB_NAME=msrvtt_caption_11221800

# 8卡训练
CUDA_VISIBLE_DEVICES=0,1,2,3 \
llamafactory-cli train examples/train_vision/glm4v_cubing_msrvtt9k.yaml