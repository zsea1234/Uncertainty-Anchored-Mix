import os
import torch
import numpy as np
import nibabel as nib
from skimage import transform, measure
from medpy.metric.binary import dc
import argparse
import time
from models import build_model
import shutil
from scipy import ndimage

# ================= 🏆 最终决战：双尺度鲁棒版 🏆 =================
CHECKPOINTS = [
    '/data/MSCMR_cycleMix_Finetune_Seed888_Retrain/best_checkpoint.pth',
    '/data/MSCMR_cycleMix_Finetune_Seed42_Retrain/best_checkpoint.pth',
    '/data/MSCMR_cycleMix_PU/best_checkpoint.pth'
]
# 权重策略
WEIGHTS = [0.35, 0.25, 0.40] 

TEST_FOLDER = "/home/guest25/zyy/UnceternMix/MSCMR_dataset/val/images"
LABEL_FOLDER = "/home/guest25/zyy/UnceternMix/MSCMR_dataset/val/labels"
OUTPUT_FOLDER = "/home/guest25/zyy/UnceternMix/dataset_results/MSCMR_Ensemble_Final"
# ===============================================================

def makefolder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def load_nii(img_path):
    nimg = nib.load(img_path)
    return np.asanyarray(nimg.dataobj), nimg.affine, nimg.header

def save_nii(img_path, data, affine, header):
    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)

def keep_largest_connected_components_3d(mask):
    if mask.sum() == 0: return mask
    original_mask = mask.copy()
    try:
        # 1. Heart
        heart_vol = np.where((mask > 0), 1, 0)
        out_heart = np.zeros(heart_vol.shape, dtype=np.uint8)
        blobs = measure.label(heart_vol, connectivity=1)
        props = measure.regionprops(blobs)
        if props:
            area = [ele.area for ele in props]
            largest_blob_ind = np.argmax(area)
            largest_blob_label = props[largest_blob_ind].label
            out_heart[blobs == largest_blob_label] = 1
        
        # 2. Structures
        out_img = np.zeros(mask.shape, dtype=np.uint8)
        for struc_id in [1, 2, 3]:
            binary_vol = mask == struc_id
            blobs = measure.label(binary_vol, connectivity=1)
            props = measure.regionprops(blobs)
            if not props: continue
            area = [ele.area for ele in props]
            largest_blob_ind = np.argmax(area)
            largest_blob_label = props[largest_blob_ind].label
            out_img[blobs == largest_blob_label] = struc_id

        final_img = out_heart * out_img 
        if final_img.sum() == 0 and mask.sum() > 0: return original_mask
        return final_img
    except:
        return original_mask

def get_args_parser():
    tasks = {'MR': {'lab_values': [0, 1, 2, 3, 4, 5], 'out_channels': 4}}
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--model', default='MSCMR', required=False)
    parser.add_argument('--in_channels', default=1, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--frozen_weights', type=str, default=None)
    parser.add_argument('--tasks', default=tasks, type=dict)
    # Dummy args
    parser.add_argument('--mixup_alpha', type=float, default=0.5)
    parser.add_argument('--box', type=bool, default=False)
    parser.add_argument('--graph', type=bool, default=True)
    parser.add_argument('--neigh_size', type=int, default=2)
    parser.add_argument('--n_labels', type=int, default=3)
    parser.add_argument('--beta', type=float, default=1.2)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--eta', type=float, default=0.2)
    parser.add_argument('--transport', type=bool, default=True)
    parser.add_argument('--t_eps', type=float, default=0.8)
    parser.add_argument('--t_size', type=int, default=4)
    parser.add_argument('--adv_eps', type=float, default=10.0)
    parser.add_argument('--adv_p', type=float, default=0.0)
    parser.add_argument('--clean_lam', type=float, default=0.0)
    parser.add_argument('--mp', type=int, default=8)
    parser.add_argument('--anchor_lambda', type=float, default=20.0)
    parser.add_argument('--use_uncertainty_saliency', type=bool, default=True)
    parser.add_argument('--mc_dropout_iters', type=int, default=5)
    parser.add_argument('--uncertainty_mode', type=str, default='entropy')
    parser.add_argument('--start_urpc_epoch', default=0, type=int)
    parser.add_argument('--multiDice_loss_coef', default=0, type=float)
    parser.add_argument('--CrossEntropy_loss_coef', default=1, type=float)
    parser.add_argument('--pu_loss_coef', default=0.2, type=float)
    parser.add_argument('--Rv', default=1, type=float)
    parser.add_argument('--Lv', default=1, type=float)
    parser.add_argument('--Myo', default=1, type=float)
    parser.add_argument('--Avg', default=1, type=float)
    parser.add_argument('--dataset', default='MSCMR_dataset', type=str)
    parser.add_argument('--output_dir', default='', help='')
    parser.add_argument('--dist_url', default='env://', help='')
    return parser

def calculate_single_dice(gt, pred):
    dices = []
    for struc in [1, 2, 3]:
        gt_binary = (gt == struc) * 1
        pred_binary = (pred == struc) * 1
        if np.sum(gt_binary) == 0 and np.sum(pred_binary) == 0:
            d = 1.0
        elif np.sum(pred_binary) > 0 and np.sum(gt_binary) == 0 or np.sum(pred_binary) == 0 and np.sum(gt_binary) > 0:
            d = 0.0
        else:
            d = dc(gt_binary, pred_binary)
        dices.append(d)
    return np.mean(dices)

def predict_on_tensor(tensor, models, device):
    """辅助函数：对给定 Tensor 进行集成 + TTA 预测"""
    tensor = tensor.to(device)
    ensemble_prob = 0.0
    for model_idx, model in enumerate(models):
        p1 = torch.softmax(model(tensor, 'MR')['pred_masks'], dim=1)
        p2 = torch.flip(torch.softmax(model(torch.flip(tensor, [3]), 'MR')['pred_masks'], dim=1), [3])
        p3 = torch.flip(torch.softmax(model(torch.flip(tensor, [2]), 'MR')['pred_masks'], dim=1), [2])
        model_pred = (p1 + p2 + p3) / 3.0
        ensemble_prob += model_pred * WEIGHTS[model_idx]
    return ensemble_prob.detach().cpu().numpy().squeeze()

@torch.no_grad()
def main_ensemble():
    parser = argparse.ArgumentParser('MSCMR Ensemble', parents=[get_args_parser()])
    args = parser.parse_args()
    device = torch.device(args.device)
    
    models = []
    print(f"========== Loading {len(CHECKPOINTS)} Models ==========")
    for i, ckpt_path in enumerate(CHECKPOINTS):
        print(f"Loading Model {i+1}: {ckpt_path}")
        model, _, _, _ = build_model(args)
        model.to(device)
        model.eval()
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            state_dict = checkpoint['model']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False) 
            models.append(model)
        else:
            print(f"Error: Checkpoint not found: {ckpt_path}")
            return

    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    makefolder(OUTPUT_FOLDER)
    dir_pred = os.path.join(OUTPUT_FOLDER, "predictions")
    makefolder(dir_pred)
    dir_gt = os.path.join(OUTPUT_FOLDER, "masks")
    makefolder(dir_gt)

    test_files = sorted(os.listdir(TEST_FOLDER))
    print(f"========== Start Dual-Scale Robust Inference ==========")
    
    all_dices = []
    
    for file_index in range(len(test_files)):
        test_file = test_files[file_index]
        label_file = test_file.replace(".nii.gz", "_manual.nii.gz")
        if not os.path.exists(os.path.join(LABEL_FOLDER, label_file)): continue
            
        print(f"\n--- Processing {test_file} ---")
        img_dat = load_nii(os.path.join(TEST_FOLDER, test_file))
        mask_dat = load_nii(os.path.join(LABEL_FOLDER, label_file))
        img = img_dat[0].copy().astype(np.float32)
        mask = mask_dat[0] 
        
        # 1. 鲁棒归一化 (Percentile Clip)
        p05 = np.percentile(img, 0.5)
        p995 = np.percentile(img, 99.5)
        img = np.clip(img, p05, p995)
        img = np.divide((img - np.mean(img)), np.std(img) + 1e-8)
        
        predictions_3d_accum = np.zeros((4, mask.shape[0], mask.shape[1], mask.shape[2]))
        
        for slice_index in range(img.shape[2]):
            img_slice = img[:,:,slice_index]
            h, w = img_slice.shape
            
            # === Scale 1: Simple Resize (212x212) - 全局视野 ===
            img_212 = transform.resize(img_slice, (212, 212), order=1, preserve_range=True)
            t_212 = torch.from_numpy(img_212).float().view(1, 1, 212, 212)
            prob_212 = predict_on_tensor(t_212, models, device)
            
            # 还原 Scale 1
            prob_212_orig = transform.resize(prob_212, (4, h, w), order=3, preserve_range=True)
            
            # === Scale 2: Center Crop from Resize (256x256) - 高清视野 ===
            # 先缩放到 256
            img_256 = transform.resize(img_slice, (256, 256), order=1, preserve_range=True)
            # Center Crop 212
            c_start = (256 - 212) // 2
            img_256_crop = img_256[c_start:c_start+212, c_start:c_start+212]
            t_256 = torch.from_numpy(img_256_crop).float().view(1, 1, 212, 212)
            prob_256_crop = predict_on_tensor(t_256, models, device)
            
            # 还原 Scale 2 (先放到 256 画布，再缩放回原图)
            prob_256_full = np.zeros((4, 256, 256))
            prob_256_full[:, c_start:c_start+212, c_start:c_start+212] = prob_256_crop
            prob_256_orig = transform.resize(prob_256_full, (4, h, w), order=3, preserve_range=True)
            
            # === 平均融合 ===
            # Scale 1 权重略高，因为它保证不漏
            final_prob = 0.6 * prob_212_orig + 0.4 * prob_256_orig
            predictions_3d_accum[:, :, :, slice_index] = final_prob
            
        # Argmax
        prediction_mask = np.uint8(np.argmax(predictions_3d_accum, axis=0))
        
        # 3D LCC
        prediction_mask = keep_largest_connected_components_3d(prediction_mask)
        
        save_nii(os.path.join(dir_pred, label_file), prediction_mask, mask_dat[1], mask_dat[2])
        save_nii(os.path.join(dir_gt, label_file), mask_dat[0], mask_dat[1], mask_dat[2])
        
        current_dice = calculate_single_dice(mask_dat[0], prediction_mask)
        all_dices.append(current_dice)
        print(f"-> Dice = {current_dice:.4f}")

    print(f"\n🏆 FINAL DICE SCORE: {np.mean(all_dices):.5f} 🏆")

if __name__ == '__main__':
    main_ensemble()