CUDA_VISIBLE_DEVICES=3 nohup sh -c '
# === Seed 888 ===
echo "Starting Seed 888..." && \
python -u main.py \
  --model MSCMR --dataset MSCMR_dataset \
  --resume /data/MSCMR_cycleMix_PU/best_checkpoint.pth \
  --lr 1e-5 --batch_size 8 \
  --Rv 10.0 --Lv 1.0 --Myo 1.5 \
  --mixup_alpha 2.0 --graph True \
  --n_labels 3 --eta 0.2 --beta 1.2 --gamma 0.5 --neigh_size 2 \
  --transport True --t_size 4 --t_eps 0.8 \
  --use_uncertainty_saliency True --anchor_lambda 25.0 \
  --pu_loss_coef 0.2 --mc_dropout_iters 10 \
  --seed 888 --epochs 200 \
  --output_dir /data/MSCMR_cycleMix_Finetune_Seed888_Retrain/ \
  > log_seed888_retrain.txt 2>&1 \

&& \

# === Seed 42 ===
echo "Starting Seed 42..." && \
python -u main.py \
  --model MSCMR --dataset MSCMR_dataset \
  --resume /data/MSCMR_cycleMix_PU/best_checkpoint.pth \
  --lr 1e-5 --batch_size 8 \
  --Rv 10.0 --Lv 1.0 --Myo 1.5 \
  --mixup_alpha 2.0 --graph True \
  --n_labels 3 --eta 0.2 --beta 1.2 --gamma 0.5 --neigh_size 2 \
  --transport True --t_size 4 --t_eps 0.8 \
  --use_uncertainty_saliency True --anchor_lambda 25.0 \
  --pu_loss_coef 0.2 --mc_dropout_iters 10 \
  --seed 42 --epochs 200 \
  --output_dir /data/MSCMR_cycleMix_Finetune_Seed42_Retrain/ \
  > log_seed42_retrain.txt 2>&1 \

&& \

# === Seed 666 ===
echo "Starting Seed 666..." && \
python -u main.py \
  --model MSCMR --dataset MSCMR_dataset \
  --resume /data/MSCMR_cycleMix_PU/best_checkpoint.pth \
  --lr 1e-5 --batch_size 8 \
  --Rv 10.0 --Lv 1.0 --Myo 1.5 \
  --mixup_alpha 2.0 --graph True \
  --n_labels 3 --eta 0.2 --beta 1.2 --gamma 0.5 --neigh_size 2 \
  --transport True --t_size 4 --t_eps 0.8 \
  --use_uncertainty_saliency True --anchor_lambda 25.0 \
  --pu_loss_coef 0.2 --mc_dropout_iters 10 \
  --seed 666 --epochs 200 \
  --output_dir /data/MSCMR_cycleMix_Finetune_Seed666_Retrain/ \
  > log_seed666_retrain.txt 2>&1
' &