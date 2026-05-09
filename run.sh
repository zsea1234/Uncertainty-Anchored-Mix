export LD_LIBRARY_PATH=/mnt/ssd2/guest25/miniconda3/envs/cyclemix/lib/python3.8/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=1 nohup python -u main.py \
  --mixup_alpha 0.5 \
  --graph True \
  --n_labels 3 \
  --eta 0.2 \
  --beta 1.2 \
  --gamma 0.5 \
  --neigh_size 4 \
  --transport True \
  --t_size 4 \
  --t_eps 0.8 \
  --use_uncertainty_saliency True \
  --anchor_lambda 15.0 \
  --pu_loss_coef 0.1 \
  --mc_dropout_iters 5 \
  --output_dir /data/MSCMR_cycleMix_PU/ \
  > train_log.txt 2>&1 &