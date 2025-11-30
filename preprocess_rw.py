import os
import glob
import numpy as np
import nibabel as nib
from skimage.segmentation import random_walker
from skimage.exposure import rescale_intensity
from tqdm import tqdm
from pathlib import Path

# ================= 配置区域 =================
# 1. 获取当前脚本绝对路径
CURRENT_DIR = Path(__file__).resolve().parent

# 2. 智能定位数据集根目录
# 优先检查当前目录下是否存在 MSCMR_dataset (针对脚本在根目录的情况)
if (CURRENT_DIR / 'MSCMR_dataset').exists():
    DATA_ROOT = CURRENT_DIR / 'MSCMR_dataset'
# 其次检查上级目录下是否存在 (针对脚本在 data/ 目录的情况)
elif (CURRENT_DIR.parent / 'MSCMR_dataset').exists():
    DATA_ROOT = CURRENT_DIR.parent / 'MSCMR_dataset'
else:
    # 默认回退到当前目录，让后续报错提示更直观
    DATA_ROOT = CURRENT_DIR / 'MSCMR_dataset'

# 3. 图像和标签路径
TRAIN_IMG_DIR = DATA_ROOT / 'train' / 'images'
TRAIN_LAB_DIR = DATA_ROOT / 'train' / 'labels'
# ===========================================

def process_case(img_path_obj):
    img_path = str(img_path_obj)
    filename = img_path_obj.name
    
    # 动态处理后缀
    if filename.endswith('.nii.gz'):
        suffix = '.nii.gz'
    elif filename.endswith('.nii'):
        suffix = '.nii'
    else:
        return 

    scribble_filename = filename.replace(f'_DE{suffix}', f'_DE_scribble{suffix}')
    save_filename = filename.replace(f'_DE{suffix}', f'_DE_dense{suffix}')
    
    scribble_path = TRAIN_LAB_DIR / scribble_filename
    save_path = TRAIN_LAB_DIR / save_filename

    if not scribble_path.exists():
        return

    # 如果需要强制重新生成，请注释掉下面这两行
    if save_path.exists():
        print(f"[Skip] Already exists: {save_filename}")
        return

    try:
        # 1. 加载数据
        img_obj = nib.load(img_path)
        scr_obj = nib.load(str(scribble_path))
        
        img_data = img_obj.get_fdata()
        scr_data = scr_obj.get_fdata().astype(np.int32)
        
        dense_mask = np.zeros_like(scr_data)
        depth = img_data.shape[2]
        
        for i in range(depth):
            slice_img = img_data[:, :, i]
            slice_scr = scr_data[:, :, i]
            
            unique_labels = np.unique(slice_scr)
            if len(unique_labels) < 2:
                continue
            
            # === 改进点 1: 归一化到 [0, 1] ===
            # Random Walker 对 diff 的计算更敏感，0-1 范围比 -1,1 更容易控制 beta
            img_norm = rescale_intensity(slice_img, in_range='image', out_range=(0, 1))
            
            # === 改进点 2: 智能背景种子 (关键修复) ===
            # MRI 背景通常是黑色的。我们将亮度低于 0.05 (5%) 的区域强制设为背景种子。
            # 这就像给背景加了一道“防洪堤”，防止前景泄漏。
            bg_threshold = 0.05
            background_mask = img_norm < bg_threshold
            
            # 保留原有的边缘强制约束 (双重保险)
            background_mask[:5, :] = True
            background_mask[-5:, :] = True
            background_mask[:, :5] = True
            background_mask[:, -5:] = True
            
            # 确保不覆盖已有的 scribble (虽然 scribble 应该在亮处，但以防万一)
            background_mask = background_mask & (slice_scr == 0)
            
            # 构建 Markers
            rw_markers = np.zeros_like(slice_scr)
            rw_markers[background_mask] = 1 # 背景设为 1
            
            mask_fg = slice_scr > 0
            rw_markers[mask_fg] = slice_scr[mask_fg] + 1 # 前景设为 2, 3...
            
            # === 改进点 3: 调整 beta ===
            # 归一化改为 0-1 后，diff 变小了，需要适当调大 beta 以保持边界敏感度
            # beta=130 在 0-1 范围内可能偏弱，可以尝试提高到 1000，或者保持 130 观察效果
            # 有了上面的背景阈值保护，beta=130 应该也足够安全了。
            labels = random_walker(img_norm, rw_markers, beta=130, mode='bf')
            
            # 还原标签
            labels_restored = labels.copy()
            labels_restored[labels == 1] = 0
            labels_restored[labels > 1] -= 1
            
            dense_mask[:, :, i] = labels_restored
            
        # 保存结果
        new_nii = nib.Nifti1Image(dense_mask, img_obj.affine, img_obj.header)
        nib.save(new_nii, str(save_path))
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")
    img_path = str(img_path_obj)
    filename = img_path_obj.name
    
    # --- 关键修复：动态处理 .nii 和 .nii.gz 后缀 ---
    if filename.endswith('.nii.gz'):
        suffix = '.nii.gz'
    elif filename.endswith('.nii'):
        suffix = '.nii'
    else:
        return # 跳过其他文件

    # 构造文件名
    # 原图: subjectXX_DE.nii.gz
    # 涂鸦: subjectXX_DE_scribble.nii.gz
    # 目标: subjectXX_DE_dense.nii.gz
    scribble_filename = filename.replace(f'_DE{suffix}', f'_DE_scribble{suffix}')
    save_filename = filename.replace(f'_DE{suffix}', f'_DE_dense{suffix}')
    
    scribble_path = TRAIN_LAB_DIR / scribble_filename
    save_path = TRAIN_LAB_DIR / save_filename

    # 检查涂鸦是否存在
    if not scribble_path.exists():
        # print(f"[Skip] No scribble found for: {filename}")
        return

    # 检查是否已处理过
    if save_path.exists():
        print(f"[Skip] Already exists: {save_filename}")
        return

    try:
        # 1. 加载数据
        img_obj = nib.load(img_path)
        scr_obj = nib.load(str(scribble_path))
        
        img_data = img_obj.get_fdata()
        scr_data = scr_obj.get_fdata().astype(np.int32)
        
        # 2. 准备输出
        dense_mask = np.zeros_like(scr_data)
        
        # 3. 逐层处理 (Slice-by-Slice)
        depth = img_data.shape[2]
        
        for i in range(depth):
            slice_img = img_data[:, :, i]
            slice_scr = scr_data[:, :, i]
            
            unique_labels = np.unique(slice_scr)
            
            # 如果该层没有有效标注（只有0），跳过
            if len(unique_labels) < 2:
                continue
                
            # 图像归一化
            img_norm = rescale_intensity(slice_img, in_range='image', out_range=(-1, 1))
            
            # --- 构建 Markers ---
            # Random Walker 需要确定的背景种子和前景种子
            
            # A. 自动添加背景种子 (假设图像边缘是背景)
            # 这里的逻辑是防止 RW 把整个未标注区域都填成前景
            background_mask = np.zeros_like(slice_img, dtype=bool)
            background_mask[:5, :] = True
            background_mask[-5:, :] = True
            background_mask[:, :5] = True
            background_mask[:, -5:] = True
            # 确保不覆盖已有的 scribble
            background_mask = background_mask & (slice_scr == 0)
            
            rw_markers = np.zeros_like(slice_scr)
            # 将背景种子设为 1 (RW 中 1 代表第一个 Label)
            rw_markers[background_mask] = 1
            
            # B. 添加前景种子
            # 原标签 (1, 2, 3) -> RW标签 (2, 3, 4)
            mask_fg = slice_scr > 0
            rw_markers[mask_fg] = slice_scr[mask_fg] + 1
            
            # 执行 Random Walker (beta越大越难跨越边缘)
            labels = random_walker(img_norm, rw_markers, beta=130, mode='bf')
            
            # C. 还原标签 (1->0, 2->1, 3->2 ...)
            labels_restored = labels.copy()
            labels_restored[labels == 1] = 0
            labels_restored[labels > 1] -= 1
            
            dense_mask[:, :, i] = labels_restored
            
        # 4. 保存结果 (.nii.gz)
        new_nii = nib.Nifti1Image(dense_mask, img_obj.affine, img_obj.header)
        nib.save(new_nii, str(save_path))
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")

def main():
    print(f"--- Random Walker Preprocessing ---")
    print(f"Images Dir: {TRAIN_IMG_DIR}")
    print(f"Labels Dir: {TRAIN_LAB_DIR}")
    
    if not TRAIN_IMG_DIR.exists():
        print(f"Error: Image directory not found!")
        return

    # 同时查找 .nii 和 .nii.gz
    img_files = list(TRAIN_IMG_DIR.glob('*_DE.nii')) + list(TRAIN_IMG_DIR.glob('*_DE.nii.gz'))
    
    print(f"Found {len(img_files)} images.")
    
    if len(img_files) == 0:
        print("Error: No images found. Please check if the path contains .nii or .nii.gz files.")
        return

    print("Starting processing...")
    # 使用 tqdm 显示进度条
    for img_path in tqdm(img_files):
        process_case(img_path)
        
    print("Done! Dense masks generated.")

if __name__ == '__main__':
    main()