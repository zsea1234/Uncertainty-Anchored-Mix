#!/bin/bash

# 设置环境变量
export LD_LIBRARY_PATH=/mnt/ssd2/guest25/miniconda3/envs/cyclemix/lib/python3.8/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# 注意：这里删除了全局的 CUDA_VISIBLE_DEVICES 设置，改为在每条命令前单独设置

# 基础参数 (200 Epochs)
BASE_ARGS="--mixup_alpha 0.5 \
  --graph True \
  --n_labels 3 \
  --eta 0.2 \
  --beta 1.2 \
  --gamma 0.5 \
  --neigh_size 4 \
  --transport True \
  --t_size 4 \
  --t_eps 0.8 \
  --pu_loss_coef 0.1 \
  --mc_dropout_iters 5 \
  --epochs 200"

echo "========================================================"
echo "Start Parallel Ablation Studies (GPU 1 & GPU 2)"
echo "Date: $(date)"
echo "========================================================"

# Exp 2 已跳过 (你已经跑完了)
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# 实验 3: Uncertainty + Anchor (效率验证) -> 分配给 GPU 2
# ------------------------------------------------------------------
echo "Launching Exp 3 on GPU 2..."
CUDA_VISIBLE_DEVICES=2 python -u main.py $BASE_ARGS \
  --use_uncertainty_saliency True \
  --anchor_lambda 15.0 \
  --start_urpc_epoch 9999 \
  --output_dir /data/MSCMR_cycleMix_PU/ablation_unc_anchor/ \
  > log_exp3_unc_anchor.txt 2>&1 &  
# 注意末尾的 '&' 符号，表示后台运行

PID_EXP3=$! # 记录进程ID
echo "Exp 3 started with PID: $PID_EXP3"

# ------------------------------------------------------------------
# 实验 4: Uncertainty + URPC (约束验证) -> 分配给 GPU 1
# ------------------------------------------------------------------
echo "Launching Exp 4 on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python -u main.py $BASE_ARGS \
  --use_uncertainty_saliency True \
  --anchor_lambda 0 \
  --start_urpc_epoch 15 \
  --output_dir /data/MSCMR_cycleMix_PU/ablation_unc_urpc/ \
  > log_exp4_unc_urpc.txt 2>&1 & 
# 注意末尾的 '&' 符号

PID_EXP4=$! # 记录进程ID
echo "Exp 4 started with PID: $PID_EXP4"

# ------------------------------------------------------------------
echo "Both tasks are running in parallel."
echo "Waiting for completion..."
wait $PID_EXP3 $PID_EXP4
echo "========================================================"
echo "All Parallel Ablation Studies Completed!"