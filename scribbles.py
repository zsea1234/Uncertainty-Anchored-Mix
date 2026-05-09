import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import binary_dilation

# ================= 配置区 (请根据你的环境修改这里) =================

# 1. 定义要处理的病人 ID 列表
PATIENT_IDS = ["patient078", "patient079"]

# 2. 基础文件夹路径 (代码会自动在后面拼接文件名)
# 请确保这些路径指向的是包含 .nii.gz 文件的文件夹，而不是具体文件
BASE_IMG_DIR = "/home/guest25/zyy/nnUNet_raw/Dataset504_ACDC_Small_FullSup/imagesTr"
BASE_GT_DIR = "/home/guest25/zyy/Uncertainty-Anchored-Mix/dataset_results/ACDC_Ensemble_Final/masks"
BASE_SCRIBBLE_DIR = "/home/guest25/zyy/Uncertainty-Anchored-Mix/ACDC_dataset/val/weak_labels"

# 3. 输出保存的总目录
OUTPUT_DIR = "./ACDC_Visualization_Results"

# 4. 可视化设置
CROP_SIZE = 160
THICKNESS = 2  # 涂鸦线条粗细
dpi_setting = 300

# ACDC 颜色定义
colors = {
    1: [0.0, 1.0, 0.0, 1.0],  # RV: Green
    2: [1.0, 0.0, 1.0, 1.0],  # Myo: Magenta
    3: [1.0, 1.0, 0.0, 1.0]   # LV: Yellow
}

# ================= 工具函数 =================
def crop_center(img, ref_mask, size):
    """ 使用参考掩码的中心进行裁剪 """
    coords = np.argwhere(ref_mask > 0)
    if coords.size == 0:
        center_y, center_x = img.shape[0]//2, img.shape[1]//2
    else:
        center_y, center_x = coords.mean(axis=0).astype(int)
    
    y1, y2 = center_y - size//2, center_y + size//2
    x1, x2 = center_x - size//2, center_x + size//2
    
    pad_y1 = max(0, -y1); pad_y2 = max(0, y2 - img.shape[0])
    pad_x1 = max(0, -x1); pad_x2 = max(0, x2 - img.shape[1])
    
    cropped = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
    return np.pad(cropped, ((pad_y1, pad_y2), (pad_x1, pad_x2)), mode='constant')

def save_plot(img_data, filename, overlay=None):
    """ 通用绘图保存函数 """
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
    ax.set_facecolor('black')
    
    # 绘制底图
    ax.imshow(img_data, cmap='gray')
    
    # 如果有覆盖层（涂鸦），则绘制
    if overlay is not None:
        ax.imshow(overlay)
        
    ax.axis('off')
    plt.savefig(filename, dpi=dpi_setting, bbox_inches='tight', pad_inches=0)
    plt.close(fig) # 关闭图像以释放内存

def process_patient(pid):
    print(f"\n--- Processing {pid} ---")
    
    # 1. 构造文件路径 (根据ACDC常见命名规则)
    # 假设图片是 _0000.nii.gz，GT是 _gt.nii.gz，Scribble是 _scribble.nii.gz
    # 如果你的文件名规则不同，请在这里修改
    f_img = os.path.join(BASE_IMG_DIR, f"{pid}_frame01_0000.nii.gz")
    f_gt = os.path.join(BASE_GT_DIR, f"{pid}_frame01_gt.nii.gz")
    f_scribble = os.path.join(BASE_SCRIBBLE_DIR, f"{pid}_frame01_scribble.nii.gz")
    
    # 检查文件是否存在
    for p in [f_img, f_gt, f_scribble]:
        if not os.path.exists(p):
            print(f"❌ 错误: 文件不存在 -> {p}")
            return

    # 2. 加载数据
    img_data = nib.load(f_img).get_fdata()
    gt_data = nib.load(f_gt).get_fdata()
    scribble_data = nib.load(f_scribble).get_fdata()

    # 3. 确定切片 (使用 GT 最大面积的切片)
    slice_idx = np.argmax(np.sum(gt_data > 0, axis=(0, 1)))
    print(f"Slice Index: {slice_idx}")

    # 4. 旋转和裁剪
    raw_img = np.rot90(img_data[:, :, slice_idx])
    raw_gt = np.rot90(gt_data[:, :, slice_idx])
    raw_scribble = np.rot90(scribble_data[:, :, slice_idx])

    img_cropped = crop_center(raw_img, raw_gt, CROP_SIZE)
    scribble_cropped = crop_center(raw_scribble, raw_gt, CROP_SIZE)

    # 5. 准备保存目录
    save_dir = os.path.join(OUTPUT_DIR, pid)
    os.makedirs(save_dir, exist_ok=True)

    # 6. 生成并保存【原始图像】 (Raw Image)
    save_name_raw = os.path.join(save_dir, f"{pid}_Raw.png")
    save_plot(img_cropped, save_name_raw, overlay=None)
    print(f"✅ Saved Raw: {save_name_raw}")

    # 7. 生成并保存【涂鸦叠加图】 (Scribble Image)
    overlay = np.zeros((*scribble_cropped.shape, 4))
    for cls_idx, color in colors.items():
        mask = (scribble_cropped == cls_idx)
        if np.sum(mask) > 0:
            if THICKNESS > 0:
                mask = binary_dilation(mask, iterations=THICKNESS)
            overlay[mask] = color
            
    save_name_scribble = os.path.join(save_dir, f"{pid}_Scribble.png")
    save_plot(img_cropped, save_name_scribble, overlay=overlay)
    print(f"✅ Saved Scribble: {save_name_scribble}")

if __name__ == "__main__":
    # 创建主输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 循环处理每个病人
    for pid in PATIENT_IDS:
        process_patient(pid)
        
    print("\n🎉 所有处理完成！请查看文件夹:", OUTPUT_DIR)